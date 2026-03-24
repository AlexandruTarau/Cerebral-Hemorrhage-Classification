import json
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


class Preprocess:
    def __init__(self, config_path, training=False):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.training = training

        # Training augmentations
        self.augmentations = []
        if self.training:
            aug_cfg = self.config.get('augmentations', {})
            if aug_cfg.get('random_rotation', False):
                self.augmentations.append(T.RandomRotation(aug_cfg.get('degrees', 10), fill=0))
            if aug_cfg.get('horizontal_flip', False):
                self.augmentations.append(T.RandomHorizontalFlip())
            if aug_cfg.get('vertical_flip', False):
                self.augmentations.append(T.RandomVerticalFlip())
            if aug_cfg.get('color_jitter', False):
                self.augmentations.append(
                    T.ColorJitter(brightness=0.1, contrast=0.1)
                )

        # After custom processing
        resize_size = self.config.get('transformations', {}).get('resize', None)

        self.post_transforms = []
        if resize_size:
            self.post_transforms.append(T.Resize(resize_size))

        self.post_transforms.append(T.ToTensor())
        self.post_transforms = T.Compose(self.post_transforms)

    def apply_custom(self, image):
        img = np.array(image.convert('L'))  # Convert to Grayscale
        cfg = self.config['transformations']

        if cfg.get('gaussian_blur', False):
            img = cv2.GaussianBlur(img, (5, 5), img.shape[0] / 100)

        if cfg.get('ben_graham', False):
            img = cv2.addWeighted(img, 2, cv2.GaussianBlur(img, (0, 0), img.shape[0] / 50), -2, 128)

        if cfg.get('clahe', False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        if cfg.get('sobel', False):
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img = cv2.magnitude(sobelx, sobely)
            img = np.uint8(np.clip(img, 0, 255))

        if cfg.get('sharpen', False):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        if cfg.get('normalize', False):
            min_val = img.min()
            max_val = img.max()

            if max_val - min_val < 1e-6:
                img = np.zeros_like(img, dtype=np.float32)
            else:
                img = (img - min_val) / (max_val - min_val)

        return Image.fromarray(img)

    def __call__(self, image):
        if self.training:
            for aug in self.augmentations:
                image = aug(image)

        image = self.apply_custom(image)
        image = self.post_transforms(image)

        return image