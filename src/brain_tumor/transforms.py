"""
Reusable image transforms for training and validation.

Anti-Clever-Hans strategies:
- CenterCrop: Forces model to focus on brain center, not edges/skull
- RandomErasing: Prevents over-reliance on specific regions
- Grayscale: Removes color artifacts that might correlate with labels
- Stronger geometric augmentation: Breaks spatial shortcuts
"""

from collections.abc import Callable
import random
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class CenterBiasedCrop(nn.Module):
    """
    Crops with a bias towards the center of the image.
    This helps the model focus on the brain region rather than edges/skull.
    """

    def __init__(
        self,
        size: int = 224,
        center_weight: float = 0.7,
        scale: tuple[float, float] = (0.75, 0.95),
    ) -> None:
        super().__init__()
        self.size = size
        self.center_weight = center_weight
        self.scale = scale

    def forward(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        # Random scale
        scale = random.uniform(self.scale[0], self.scale[1])
        crop_size = int(min(width, height) * scale)

        # Bias towards center
        max_x = width - crop_size
        max_y = height - crop_size

        if random.random() < self.center_weight:
            # Center crop with small jitter
            jitter = int(crop_size * 0.1)
            center_x = (width - crop_size) // 2
            center_y = (height - crop_size) // 2
            x = max(0, min(max_x, center_x + random.randint(-jitter, jitter)))
            y = max(0, min(max_y, center_y + random.randint(-jitter, jitter)))
        else:
            # Random crop (but still not extreme edges)
            margin = int(crop_size * 0.15)
            x = random.randint(min(margin, max_x), max(margin, max_x - margin))
            y = random.randint(min(margin, max_y), max(margin, max_y - margin))

        img = img.crop((x, y, x + crop_size, y + crop_size))
        return img.resize((self.size, self.size))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, center_weight={self.center_weight})"


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation: blends pairs of images and labels.
    Helps prevent memorization and improves generalization.

    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
        # Ensure we keep at least 50% contribution from the primary sample
        lam = max(lam, 1.0 - lam)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable[..., torch.Tensor],
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def build_train_transforms(
    image_size: int = 224, anti_clever_hans: bool = True
) -> transforms.Compose:
    """
    Train-time augmentation to improve generalization.

    Args:
        image_size: Target image size
        anti_clever_hans: If True, applies stronger center-biased augmentation
                         to prevent learning from edge/skull artifacts
    """
    if anti_clever_hans:
        # Anti-Clever-Hans pipeline: focus on center, remove edge shortcuts
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(280),  # Larger initial size for better cropping
                CenterBiasedCrop(size=image_size, center_weight=0.7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),  # MRI can be flipped
                transforms.RandomRotation(20),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)], p=0.3
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.15), ratio=(0.5, 2.0), value="random"  # type: ignore[arg-type]
                ),
                transforms.RandomApply([AddGaussianNoise(0.0, 0.03)], p=0.2),
                transforms.Normalize(mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD),
            ]
        )
    else:
        # Standard pipeline (legacy)
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
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)], p=0.2
                ),
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
    Uses strict center crop to evaluate on the same region the model was trained on.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(280),  # Match training resize
            transforms.CenterCrop(image_size),  # Strict center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGE_MEAN, std=DEFAULT_IMAGE_STD),
        ]
    )
