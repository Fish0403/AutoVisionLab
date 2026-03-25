"""Augmentation registry for structured classification training."""

from __future__ import annotations

import math
import random

import torch
from torchvision.transforms import transforms

from app.schemas.parameter_space import AugmentationParams


def build_train_transform(
    image_size: int,
    augmentation_policy: str,
    augmentation_params: AugmentationParams,
) -> transforms.Compose:
    """Build the train transform pipeline from structured params."""
    steps: list[object] = [transforms.Resize((image_size, image_size))]
    steps.extend(_build_policy_transforms(image_size, augmentation_policy))
    steps.append(transforms.ToTensor())
    if augmentation_params.random_erasing_prob > 0:
        steps.append(
            transforms.RandomErasing(
                p=augmentation_params.random_erasing_prob,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    steps.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    return transforms.Compose(steps)


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Build the eval transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def apply_batch_augmentations(
    images: torch.Tensor,
    labels: torch.Tensor,
    augmentation_params: AugmentationParams,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor, float]]:
    """Apply structured batch augmentations such as mixup and cutmix."""
    mixup_alpha = augmentation_params.mixup_alpha
    cutmix_alpha = augmentation_params.cutmix_alpha
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, labels

    if mixup_alpha > 0 and cutmix_alpha > 0:
        use_cutmix = random.random() < 0.5
    else:
        use_cutmix = cutmix_alpha > 0

    if use_cutmix:
        return _apply_cutmix(images, labels, cutmix_alpha)
    return _apply_mixup(images, labels, mixup_alpha)


def _build_policy_transforms(
    image_size: int,
    augmentation_policy: str,
) -> list[object]:
    if augmentation_policy == "none":
        return []
    if augmentation_policy == "basic":
        return [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    raise ValueError(f"Unsupported augmentation_policy: {augmentation_policy}")


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
