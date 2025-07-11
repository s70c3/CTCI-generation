import os
import os.path as osp
from typing import Optional

import numpy as np
import cv2
from torch.utils.data import Dataset


class Image2ImageDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        conditioning_images_dir: str,
        caption: str,
        width: int = 256,
        height: int = 256,
        conditioning_width: Optional[int] = None,
        conditioning_height: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.cond_width = conditioning_width if conditioning_width is not None else width
        self.cond_height = conditioning_height if conditioning_height is not None else height

        self.images_dir = images_dir
        self.conditioning_dir = conditioning_images_dir

        self.images_list = sorted(os.listdir(self.images_dir))
        self.conditioning_list = sorted(os.listdir(self.conditioning_dir))
        assert len(self.images_list) == len(self.conditioning_list), (
            "Number of images and conditioning should be same"
        )

        self.caption = caption

    def __len__(self) -> int:
        return len(self.images_list)

    def _load_image(self, path: str, width: int, height: int) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = (image.astype(np.float32) / 127.5) - 1.0
        return image

    def __getitem__(self, idx: int):
        image_path = osp.join(self.images_dir, self.images_list[idx])
        condition_path = osp.join(self.conditioning_dir, self.conditioning_list[idx])

        image = self._load_image(image_path, self.width, self.height)
        condition = self._load_image(condition_path, self.cond_width, self.cond_height)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": condition,
            "caption": self.caption,
        }