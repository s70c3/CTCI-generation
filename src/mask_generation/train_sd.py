import os
import os.path as osp
import cv2
import numpy as np
from lightning import Trainer
from omegaconf import OmegaConf

from accelerate import Accelerator

import torch
import torch.nn as nn
import lightning.pytorch as pl
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel

from src.common.callbacks import GenerateImage2ImageCallback, SaveWeightsCallback
from src.mask_generation.train_vae import VQVAE


class SD15Lightning(pl.LightningModule):
    def __init__(
        self,
        vqvae: VQVAE,
        tokenizer,
        text_encoder,
        lr: float = 1e-4,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vqvae", "tokenizer", "text_encoder"])

        self.vqvae = vqvae
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="unet"
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps)

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_e = self.vqvae.encode(x)
            z_q, _ = self.vqvae.quantizer(z_e)
        return z_q

    def training_step(self, batch: dict, batch_idx: int):
        images = batch['pixel_values']
        captions = batch['caption']

        # Получаем latents из VQVAE
        latents = self.forward(images.to(self.device))
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.size(0),),
            device=self.device,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        tokenized = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        ).to(self.device)
        text_embeds = self.text_encoder(**tokenized).last_hidden_state

        # Предсказываем шум
        model_out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
        ).sample

        loss = self.loss_fn(model_out, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        images = batch['pixel_values']
        captions = batch['caption']
        latents = self.forward(images.to(self.device))
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.size(0),),
            device=self.device,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        tokenized = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        ).to(self.device)
        text_embeds = self.text_encoder(**tokenized).last_hidden_state

        model_out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
        ).sample

        loss = self.loss_fn(model_out, noise)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=self.hparams.lr)


class Text2ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, caption: str, width=256, height=256):
        self.width = width
        self.height = height

        self.datadir = images_dir
        self.images_list = os.listdir(self.datadir)

        self.caption = caption

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        prompt = self.caption

        image = cv2.imread(osp.join(self.datadir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))

        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(pixel_values=image, caption=prompt)


def load_vqvae_from_checkpoint(
    checkpoint_path: str,
):
    vqvae = VQVAE(
        in_channels=1,
        hidden_dim=4,
        num_residual_blocks=8,
        num_embeddings=256,
        embedding_dim=4,
        commitment_cost=0.5,
        lr=learning_rate
    )

    ckpt_data = torch.load(checkpoint_path)
    state_dict = ckpt_data.get("state_dict", ckpt_data)
    vqvae.load_state_dict(state_dict, strict=False)
    vqvae.eval()
    return vqvae


if __name__ == "__main__":
    config = OmegaConf.load("configs/mask_sd_config.yaml")

    vqvae_checkpoint = config.base_model.vqvae_checkpoint

    output_dir = config.out_directories.output_dir
    images_logs_dir = config.out_directories.images_logs_dir
    weights_logs_dir = config.out_directories.weights_logs_dir

    train_images_dir = config.datasets_dirs.train_images_dir
    val_images_dir = config.datasets_dirs.val_images_dir

    num_epochs = config.train_params.num_epochs
    learning_rate = config.train_params.learning_rate
    timesteps = config.train_params.timesteps
    num_workers = config.train_params.timesteps
    batch_size = config.train_params.batch_size
    image_size = config.train_params.image_size
    log_images_step = config.train_params.log_images_step
    log_weights_step = config.train_params.log_weights_step

    device = config.hardware.device
    precision = "bf16"

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder"
    )

    vqvae = load_vqvae_from_checkpoint(vqvae_checkpoint)
    model = SD15Lightning(
        vqvae=vqvae, tokenizer=tokenizer,
        text_encoder=text_encoder,
        lr=learning_rate, timesteps=timesteps
    ).to(device)

    train_ds = Text2ImageDataset(
        images_dir=train_images_dir,
        caption="Rocks binary segmentation mask.",
        width=config.train_params.image_size,
        height=config.train_params.image_size
    )
    val_ds = Text2ImageDataset(
        images_dir=val_images_dir,
        caption="Rocks binary segmentation mask.",
        width=config.train_params.image_size,
        height=config.train_params.image_size
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    log_callback = GenerateImage2ImageCallback(
        log_dir=images_logs_dir,
        log_every_n_steps=log_images_step
    )
    save_callback = SaveWeightsCallback(
        log_dir=weights_logs_dir,
        modules_to_save=["ip_adapter"],
        log_every_n_steps=log_weights_step
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        devices=accelerator.num_processes,
        default_root_dir=output_dir,
        log_every_n_steps=10,
        precision='bf16',
        callbacks=[log_callback, save_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete! Model saved to: {weights_logs_dir}")
