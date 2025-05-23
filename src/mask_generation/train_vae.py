import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import lightning.pytorch as pl


from diffusers import AutoencoderKL


class VectorQuantizer(nn.Module):
    '''
    Добросовестно сгенерено гптшкой
    '''
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = torch.argmin(dist, dim=1).unsqueeze(1)
        encodings = torch.zeros(indices.size(0), self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, indices, 1)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # Losses
        e_latent = F.mse_loss(quantized.detach(), z_e)
        q_latent = F.mse_loss(quantized, z_e.detach())
        loss = q_latent + self.commitment_cost * e_latent
        # Straight-through
        quantized = z_e + (quantized - z_e).detach()
        # Perplexity
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity


class SDXLVQVAE(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_name: str = "stabilityai/sdxl-vae",
            in_channels: int = 1,
            num_embeddings: int = 512,
            embedding_dim: int = 4 * 4 * 4,
            commitment_cost: float = 0.25,
            lr: float = 2e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vae = AutoencoderKL.from_pretrained(
            self.hparams.pretrained_model_name,
            subfolder=None
        )

        self.vq = VectorQuantizer(
            num_embeddings=self.hparams.num_embeddings,
            embedding_dim=self.hparams.embedding_dim,
            commitment_cost=self.hparams.commitment_cost,
        )

        if in_channels == 1:
            vae_first_conv = self.vae.encoder.conv_in
            vae_last_conv = self.vae.decoder.conv_out

            # TODO: добавить функцию активации и нормализацию
            self.input_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=vae_first_conv.in_channels,
                kernel_size=3,
                stride=1, padding=1
            )

            with torch.no_grad():
                w = vae_first_conv.weight.mean(dim=1, keepdim=True)
                self.input_conv.weight.copy_(w)
                if vae_first_conv.bias is not None:
                    self.input_conv.bias.copy_(vae_first_conv.bias.mean())

            # TODO: добавить функцию активации и нормализацию
            self.output_conv = nn.Conv2d(
                in_channels=vae_last_conv.out_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1, padding=1
            )

            with torch.no_grad():
                w_out = vae_last_conv.weight.mean(dim=0, keepdim=True)
                self.output_conv.weight.copy_(w_out)
                if vae_last_conv.bias is not None:
                    self.output_conv.bias.copy_(vae_last_conv.bias.mean())

        self.mxe_loss = nn.MSELoss()


    def forward(self, x):
        # изображение должно быть отмасштабировано к [-1, 1]
        if hasattr(self, 'input_conv'):
            x = self.input_conv(x)

        z = self.vae.encode(x).latent_dist.mean
        z_q, vq_loss, perplexity = self.vq(z)

        x = self.vae.decode(z_q).sample
        if hasattr(self, 'output_conv'):
            x = self.output_conv(x)

        return x, vq_loss, perplexity

    def training_step(self, batch, batch_idx):
        new_images, vq_loss, perplexity = self(batch)

        recon_loss = self.mse_loss(new_images, batch)
        loss = recon_loss + vq_loss

        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/perplexity', perplexity, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        new_images, vq_loss, perplexity = self(batch)

        recon_loss = self.mse_loss(new_images, batch)
        loss = recon_loss + vq_loss

        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/vq_loss', vq_loss, prog_bar=True)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/perplexity', perplexity, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    pass