import math

import torch
import torch.nn as nn
import lightning.pytorch as pl


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
