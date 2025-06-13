import os
import math
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

import lightning.pytorch as pl
from lightning import Trainer

from accelerate import Accelerator
from diffusers import AutoencoderKL

from src.common.callbacks import GenerateImagesVQVAECallback


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=10, embedding_dim=128):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        nn.init.kaiming_uniform_(self._embedding.weight, a=math.sqrt(5))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        avg_probs = torch.mean(encodings, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, perplexity


class QuantizerLoss(nn.Module):
    def __init__(self, commitment_cost):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.mse = nn.MSELoss()

    def forward(self, z_e, z_q):
        codebook_loss = self.mse(z_q, z_e.detach())
        commitment_loss = self.commitment_cost * self.mse(z_q.detach(), z_e)
        return codebook_loss, commitment_loss

class VQVAELoss(nn.Module):
    def __init__(self, commitment_cost=0.5):
        super().__init__()
        self.recon_loss = nn.MSELoss()
        self.quantizer_loss = QuantizerLoss(commitment_cost)

    def forward(self, x, x_recon, z_e, z_q):
        recon = self.recon_loss(x_recon, x)
        codebook, commitment = self.quantizer_loss(z_e, z_q)
        total = recon + 0.01 * codebook + 0.01 * commitment
        return total, recon, codebook, commitment


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 4,
        num_residual_blocks: int = 4,
        num_embeddings: int = 256,
        embedding_dim: int = 4,
        commitment_cost: float = 0.5,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.residual_enc = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.project_enc = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        self.project_dec = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1)
        self.residual_dec = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.loss_fn = VQVAELoss(commitment_cost)

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.residual_enc(z_e)
        return self.project_enc(z_e)

    def decode(self, z_q):
        d = self.project_dec(z_q)
        d = self.residual_dec(d)
        return self.decoder(d)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, perplexity = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        return x_recon, z_e, z_q, perplexity

    def training_step(self, x, batch_idx):
        x_recon, z_e, z_q, perplexity = self(x)
        total, recon, codebook, commitment = self.loss_fn(x, x_recon, z_e, z_q)
        self.log('train_loss', total, prog_bar=True)
        self.log('train_recon_loss', recon, prog_bar=True)
        self.log('train_codebook_loss', codebook, prog_bar=True)
        self.log('train_commitment_loss', commitment, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)
        return total

    def validation_step(self, x, batch_idx):
        x_recon, z_e, z_q, perplexity = self(x)
        total, recon, codebook, commitment = self.loss_fn(x, x_recon, z_e, z_q)
        self.log('val_loss', total, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.listdir = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.listdir[idx])
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    config = OmegaConf.load(r"F:\\ITMO_ML\\CTCI-generation\\configs\\vqvae_config.yaml")

    output_dir = config.out_directories.output_dir
    images_logs_dir = config.out_directories.images_logs_dir
    # loss_logs_dir = config.out_directories.loss_logs_dir
    # weights_logs_dir = config.out_directories.weights_logs_dir

    train_images_dir = config.datasets_dirs.train_images_dir
    val_images_dir = config.datasets_dirs.val_images_dir

    num_epochs = config.train_params.num_epochs
    learning_rate = config.train_params.learning_rate
    batch_size = config.train_params.batch_size
    num_workers = config.train_params.num_workers
    image_size = config.train_params.image_size
    log_images_step = config.train_params.log_images_step
    log_loss_step = config.train_params.log_loss_step
    log_weights_step = config.train_params.log_weights_step

    device = config.hardware.device
    precision = config.hardware.precision

    model = VQVAE(
        in_channels=1,
        hidden_dim=4,
        num_residual_blocks=8,
        num_embeddings=256,
        embedding_dim=4,
        commitment_cost=0.5,
        lr = learning_rate,
    )

    transform = transforms.Compose([
        transforms.Resize((
            int(image_size * 1.25),
            int(image_size * 1.25)
        )),
        transforms.RandomCrop(image_size),
        transforms.ToTensor()
    ])

    accelerator = Accelerator()

    train_dataset = ImageFolderDataset(train_images_dir, transform=transform)
    val_dataset = ImageFolderDataset(val_images_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    if device == "cuda":
        torch.set_float32_matmul_precision('medium')

    log_callback = GenerateImagesVQVAECallback(
        log_dir=images_logs_dir,
        num_generate=4,
        log_every_n_steps=log_images_step
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        precision=precision,
        default_root_dir=output_dir,
        log_every_n_steps=100,
        # accumulate_grad_batches=2,
        callbacks=[log_callback],
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_loader, val_loader)