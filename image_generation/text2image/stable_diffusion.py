import torch
import torch.nn as nn
from accelerate import Accelerator

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import get_cosine_schedule_with_warmup

import lightning.pytorch as pl


class BaseDiffusionLightningModule(pl.LightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.lr = lr
        self.num_training_steps = num_training_steps
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions)
        loss = self.criterion(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions)
        loss = self.criterion(noise_pred, noise)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def encode_text(self, captions):
        input_ids = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).input_ids.to(self.device)
        return self.text_encoder(input_ids)[0]

    def add_noise(self, latents, noise, timesteps):
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class StableDiffusionLightningModule(BaseDiffusionLightningModule):
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", lr=1e-5, num_training_steps=1000):
        self.accelerator = Accelerator()
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, safety_checker=None, torch_dtype=torch.float32
        )
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        super().__init__(
            vae=pipe.vae,
            unet=pipe.unet,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            noise_scheduler=scheduler,
            lr=lr,
            num_training_steps=num_training_steps,
        )
        self.latent_scaling_factor = 0.18215

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        del pipe

    def forward(self, images, captions):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.latent_scaling_factor
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latents = self.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.encode_text(captions)
        with self.accelerator.autocast():
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred, noise

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    @torch.no_grad()
    def generate(self, captions, num_inference_steps=50, guidance_scale=7.5):
        text_embeddings = self.encode_text(captions)
        batch_size = text_embeddings.shape[0]
        latents = torch.randn(
            batch_size,
            self.unet.config.in_channels,
            self.vae.config.sample_size,
            self.vae.config.sample_size,
            device=self.device,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            latent_model_input = latents
            with self.accelerator.autocast():
                noise_pred = self.unet(latent_model_input, t, text_embeddings).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        return self.decode_latents(latents)
