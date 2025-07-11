import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from image_generation.data.text2image import Text2ImageDataset
from image_generation.text2image.stable_diffusion import StableDiffusionLightningModule


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    dataset = Text2ImageDataset(
        images_dir=cfg.data.images_dir,
        caption=cfg.data.caption,
        width=cfg.data.width,
        height=cfg.data.height,
    )
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = StableDiffusionLightningModule(
        model_name=cfg.base_model.pretrained_model_name,
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