"""
Reusable image transforms for training and validation.
"""

import torch
from torchvision import transforms

DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Train-time augmentation to improve generalization.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.90, 1.10)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(0.0, 0.05)], p=0.2),
            transforms.Normalize(mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD),
        ]
    )


def build_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation/eval transforms without augmentation.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD),
        ]
    )
