import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning.pytorch as pl

from image_generation.vae.vqvae import VQVAE


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=cfg.model.in_channels),
        transforms.Resize((cfg.data.height, cfg.data.width)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(cfg.data.images_dir, transform=transform)

    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = VQVAE(
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_residual_blocks=cfg.model.num_residual_blocks,
        num_embeddings=cfg.model.num_embeddings,
        embedding_dim=cfg.model.embedding_dim,
        commitment_cost=cfg.model.commitment_cost,
        lr=cfg.train.lr,
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