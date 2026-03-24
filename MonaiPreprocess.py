import numpy as np
import monai.transforms as mt
import torch
from PIL import Image
import json


class MonaiPreprocess:
    def __init__(self, config_path, augmentation_set="basic"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        resize_size = self.config.get('transformations', {}).get('resize', None)

        # Standard preprocessing steps common to all sets
        self.base_pre = [
            mt.ScaleIntensity(),
            mt.Resize(spatial_size=resize_size),
        ]

        # Basic Set (Baseline - No Augmentation)
        self.basic_transform = mt.Compose(self.base_pre + [mt.ToTensor()])

        # SET 1: Geometric (Spatial Augmentations)
        self.geometric_transform = mt.Compose(self.base_pre + [
            mt.RandRotate(range_x=np.pi/2, prob=0.7, keep_size=True),
            mt.RandFlip(spatial_axis=0, prob=0.5),
            mt.RandFlip(spatial_axis=1, prob=0.5),
            mt.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            mt.ToTensor()
        ])

        # SET 2: Intensity (Pixel Augmentations)
        self.intensity_transform = mt.Compose(self.base_pre + [
            mt.RandAdjustContrast(gamma=(0.5, 2.0), prob=0.7),
            mt.RandScaleIntensity(factors=0.1, prob=0.5),
            mt.RandGaussianNoise(prob=0.3, std=0.05),
            mt.RandGaussianSmooth(sigma_x=(0.25, 1.5), prob=0.3),
            mt.ToTensor()
        ])

        # SET 3: Hybrid (Geometric + Intensity)
        self.hybrid_transform = mt.Compose(self.base_pre + [
            # Geometric
            mt.RandRotate(range_x=np.pi/4, prob=0.5, keep_size=True),
            mt.RandFlip(spatial_axis=0, prob=0.5),
            mt.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),

            # Intensity
            mt.RandAdjustContrast(gamma=(0.5, 2.0), prob=0.5),
            mt.RandScaleIntensity(factors=0.1, prob=0.5),
            mt.RandGaussianNoise(prob=0.3, std=0.05),

            mt.ToTensor()
        ])

        # SET 4: Elastic
        self.elastic_transform = mt.Compose(self.base_pre + [
            mt.RandRotate(range_x=np.pi/4, prob=0.5, keep_size=True),
            mt.Rand2DElastic(
                spacing=(30, 30),
                magnitude_range=(1, 2),
                prob=0.3,
                padding_mode="zeros"
            ),
            mt.ToTensor()
        ])

        # SET 5: Ultimate (Hybrid + Elastic)
        self.ultimate_transform = mt.Compose(self.base_pre + [
            # Elastic
            mt.Rand2DElastic(
                spacing=(30, 30),
                magnitude_range=(1, 2),
                prob=0.3,
                padding_mode="zeros"
            ),

            # Geometric
            mt.RandRotate(range_x=np.pi/4, prob=0.5, keep_size=True),
            mt.RandFlip(spatial_axis=0, prob=0.5),
            mt.RandFlip(spatial_axis=1, prob=0.5),
            mt.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),

            # Intensity
            mt.RandAdjustContrast(gamma=(0.5, 2.0), prob=0.5),
            mt.RandGaussianNoise(prob=0.3, std=0.05),
            mt.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), prob=0.2),

            mt.ToTensor()
        ])

        self.sets = {
            "basic": self.basic_transform,
            "geometric": self.geometric_transform,
            "intensity": self.intensity_transform,
            "hybrid": self.hybrid_transform,
            "elastic": self.elastic_transform,
            "ultimate": self.ultimate_transform
        }

        self.active_transform = self.sets.get(augmentation_set, self.basic_transform)

    def __call__(self, image):
        img = np.array(image.convert('L'))[np.newaxis, ...].astype(np.float32)

        # Apply the selected MONAI Compose chain
        return self.active_transform(img)