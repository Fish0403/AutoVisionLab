"""MobileNetV2 trainer implementation."""

from torch import nn
from torchvision.models import mobilenet_v2

from app.trainers.classification.base_trainer import BaseClassificationTrainer


class MobileNetTrainer(BaseClassificationTrainer):
    """Trainer for torchvision MobileNetV2."""

    def create_model(self) -> nn.Module:
        """Create a MobileNetV2 classifier with 10 classes."""
        return mobilenet_v2(num_classes=10)
