import torch
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline

from image_generation.image2image.base import BaseDiffusionImage2ImageLightningModule


class ControlNetLightningModule(BaseDiffusionImage2ImageLightningModule):
    def __init__(self, vae, unet, controlnet, text_encoder, tokenizer, noise_scheduler, accelerator, lr, num_training_steps):
        super().__init__(vae, unet, text_encoder, tokenizer, noise_scheduler, lr, num_training_steps)
        self.controlnet = controlnet
        self.vae = vae
        self.unet = unet
        self.criterion = nn.MSELoss()

        self.accelerator = accelerator

    def forward(self, images, captions, conditions):
        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        text_embeddings = self.encode_text(captions)

        noise = torch.randn_like(latents).to(images.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=images.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=conditions,
            return_dict=False,
        )

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=[res.to(dtype=torch.float16) for res in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16),
            return_dict=True
        )['sample']

        return noise_pred, noise

    @torch.no_grad()
    def inference(self, captions, conditions, num_inference_steps=100):
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            requires_safety_checker = False,
            safety_checker=None,
            feature_extractor=None
        )
        # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        device = captions.device
        pipeline.to(device)
        # pipeline.set_progress_bar_config(disable=True)


        generated_images = pipeline(
            prompt=captions,
            image=conditions,
            num_inference_steps=num_inference_steps,
        ).images
        return generated_images
