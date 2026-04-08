"""Augmentation registry for structured classification training."""

from __future__ import annotations

import math
import random
from typing import Final

import torch
from torchvision.transforms import transforms

from app.schemas.parameter_space import TrainHypAugmentation


NormalizationStats = tuple[tuple[float, float, float], tuple[float, float, float]]

CLASSIFICATION_DATASET_NORMALIZATION: Final[dict[str, NormalizationStats]] = {
    "dt": ((0.4707, 0.4707, 0.4707), (0.0587, 0.0587, 0.0587)),
    "neu": ((0.5002, 0.5002, 0.5002), (0.1103, 0.1103, 0.1103)),
    "kdsc": ((0.4303, 0.4303, 0.4303), (0.0922, 0.0922, 0.0922)),
    "nt": ((0.3903, 0.3903, 0.3903), (0.0754, 0.0754, 0.0754)),
}


def resolve_classification_normalization(dataset_name: str | None) -> NormalizationStats | None:
    """Return dataset-specific normalization stats when available."""
    if dataset_name is None:
        return None
    return CLASSIFICATION_DATASET_NORMALIZATION.get(dataset_name.strip().lower())


def build_train_transform(
    image_size: int,
    augmentation: TrainHypAugmentation,
    dataset_name: str | None = None,
) -> transforms.Compose:
    """Build the train transform pipeline from structured params."""
    steps: list[object] = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    steps.append(transforms.ToTensor())
    if augmentation.random_erasing > 0:
        steps.append(
            transforms.RandomErasing(
                p=augmentation.random_erasing,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    normalization = resolve_classification_normalization(dataset_name)
    if normalization is not None:
        mean, std = normalization
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def build_eval_transform(image_size: int, dataset_name: str | None = None) -> transforms.Compose:
    """Build the eval transform pipeline."""
    steps: list[object] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    normalization = resolve_classification_normalization(dataset_name)
    if normalization is not None:
        mean, std = normalization
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def apply_batch_augmentations(
    images: torch.Tensor,
    labels: torch.Tensor,
    augmentation: TrainHypAugmentation,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor, float]]:
    """Apply structured batch augmentations such as mixup and cutmix."""
    mixup_alpha = augmentation.mixup
    cutmix_alpha = augmentation.cutmix
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, labels

    if mixup_alpha > 0 and cutmix_alpha > 0:
        use_cutmix = random.random() < 0.5
    else:
        use_cutmix = cutmix_alpha > 0

    if use_cutmix:
        return _apply_cutmix(images, labels, cutmix_alpha)
    return _apply_mixup(images, labels, mixup_alpha)

def _apply_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    mix_ratio = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed_images = mix_ratio * images + (1.0 - mix_ratio) * images[permutation]
    return mixed_images, (labels, labels[permutation], mix_ratio)


def _apply_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    mix_ratio = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
    permutation = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[permutation]

    _, _, image_height, image_width = images.size()
    cut_ratio = math.sqrt(1.0 - mix_ratio)
    cut_width = int(image_width * cut_ratio)
    cut_height = int(image_height * cut_ratio)

    center_x = random.randint(0, image_width - 1)
    center_y = random.randint(0, image_height - 1)

    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, image_width)
    y2 = min(center_y + cut_height // 2, image_height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]

    patch_area = (x2 - x1) * (y2 - y1)
    adjusted_ratio = 1.0 - (patch_area / float(image_width * image_height))
    return mixed_images, (labels, labels[permutation], adjusted_ratio)
