import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPVisionModelWithProjection

from image_generation.data.image2image import Image2ImageDataset
from image_generation.image2image.controlnet import ControlNetLightningModule
from image_generation.image2image.sd_ipa import IPAdapterLightningModule


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    dataset = Image2ImageDataset(
        images_dir=cfg.data.images_dir,
        conditioning_images_dir=cfg.data.conditioning_images_dir,
        caption=cfg.data.caption,
        width=cfg.data.width,
        height=cfg.data.height,
        conditioning_width=cfg.data.conditioning_width,
        conditioning_height=cfg.data.conditioning_height,
    )
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    if cfg.generate.get("scale") is not None:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.base_model.pretrained_model_name,
            safety_checker=None,
            torch_dtype=None,
        )
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        model = IPAdapterLightningModule(
            vae=pipe.vae,
            unet=pipe.unet,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            image_encoder=image_encoder,
            noise_scheduler=scheduler,
            lr=cfg.train.lr,
            num_training_steps=cfg.train.num_training_steps,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.base_model.pretrained_model_name,
            safety_checker=None,
        )
        model = ControlNetLightningModule(
            vae=pipe.vae,
            unet=pipe.unet,
            controlnet=pipe.controlnet,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            noise_scheduler=pipe.scheduler,
            accelerator=None,
            lr=cfg.train.lr,
            num_training_steps=cfg.train.num_training_steps,
        )

    trainer = pl.Trainer(max_epochs=cfg.train.num_epochs)
    trainer.fit(model, train_loader, val_loader)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config path")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()