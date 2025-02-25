import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from diffusers.utils import make_image_grid

from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image


class GenerateText2ImageCallback(pl.Callback):
    def __init__(self, log_dir, log_every_n_steps=1000):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        os.makedirs(log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            pl_module.eval()
            with torch.no_grad():
                captions = batch["caption"]
                captions = captions

                generated_images = pl_module.inference(captions=captions)
                generated_images = np.array(generated_images)

                for i, img in enumerate(generated_images):
                    img_path = os.path.join(self.log_dir, f"step_{global_step}_image_{i}.png")

                    pil_generated_img = self.to_pil(img)
                    grid = make_image_grid([pil_generated_img], rows=1, cols=1)

                    save_image(self.to_tensor(grid).unsqueeze(0), img_path)

                print(f"Logged generated images at step {global_step}")
            pl_module.train()


class GenerateImage2ImageCallback(pl.Callback):
    def __init__(self, log_dir, log_every_n_steps=1000):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        os.makedirs(log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            pl_module.eval()
            with torch.no_grad():
                captions = batch["caption"]
                conditions = batch["conditioning_pixel_values"].permute(0, 3, 1, 2)
                captions = captions
                conditions = conditions

                conditions = conditions.to(pl_module.device)
                generated_images = pl_module.inference(captions=captions, conditions=conditions)
                generated_images = np.array(generated_images)

                for i, img in enumerate(generated_images):
                    img_path = os.path.join(self.log_dir, f"step_{global_step}_image_{i}.png")

                    pil_conditions = self.to_pil(conditions[i])
                    pil_generated_img = self.to_pil(img)
                    grid = make_image_grid([pil_conditions, pil_generated_img], rows=1, cols=2)

                    save_image(self.to_tensor(grid).unsqueeze(0), img_path)

                print(f"Logged generated images at step {global_step}")
            pl_module.train()

class TrainingLossCallback(pl.Callback):
    def __init__(self, log_dir, log_every_n_steps=1000):
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        os.makedirs(log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0:
            train_loss = trainer.callback_metrics['train_loss'].item()
            self.train_losses.append(train_loss)
            val_loss = trainer.callback_metrics['val_loss'].item() if 'val_loss' in trainer.callback_metrics else None
            if val_loss:
                self.val_losses.append(val_loss)

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
            if self.val_losses:
                plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Training and Validation Loss at Step {global_step}')
            plt.savefig(os.path.join(self.log_dir, f'loss_plot_step_{global_step}.png'))
            plt.close()

    def on_train_end(self, trainer, pl_module):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Final Training and Validation Loss')
        plt.savefig(os.path.join(self.log_dir, 'final_loss_plot.png'))
        plt.close()


class SaveWeightsCallback(pl.Callback):
    def __init__(self, log_dir, modules_to_save, log_every_n_steps=1000):
        self.save_dir = log_dir
        self.modules_to_save = modules_to_save
        self.log_every_n_steps = log_every_n_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def save_module(self, module, module_name, global_step):
        """
        Save trainable weights of a specific module.
        """
        trainable_params = {name: param.cpu() for name, param in module.named_parameters() if param.requires_grad}
        if trainable_params:
            save_dir = os.path.join(self.save_dir, f"step_{global_step}")
            save_path = os.path.join(save_dir, f"{module_name}_weights.pt")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(trainable_params, save_path)
            print(f"Saved {module_name} trainable weights to {save_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            for module_name in self.modules_to_save:
                module = getattr(pl_module, module_name, None)
                if module:
                    self.save_module(module, module_name, trainer.global_step)
                else:
                    print(f"Module '{module_name}' not found in the model.")

