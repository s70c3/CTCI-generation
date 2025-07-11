from image_generation.base import BaseDiffusionLightningModule


class BaseDiffusionImage2ImageLightningModule(BaseDiffusionLightningModule):
    def __init__(self, vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps):
        super().__init__(vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].permute(0, 3, 1, 2)
        conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
        captions = batch["caption"]

        noise_pred, noise = self(images, captions, conditions)
        loss = self.criterion(noise_pred, noise)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss