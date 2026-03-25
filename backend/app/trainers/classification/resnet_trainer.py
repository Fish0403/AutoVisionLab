"""ResNet trainer implementations."""

from torch import nn
from torchvision.models import resnet18, resnet34

from app.trainers.classification.base_trainer import BaseClassificationTrainer


class ResNet18Trainer(BaseClassificationTrainer):
    """Trainer for torchvision ResNet18."""

    def create_model(self) -> nn.Module:
        """Create a ResNet18 classifier with 10 classes."""
        return resnet18(num_classes=10)


class ResNet34Trainer(BaseClassificationTrainer):
    """Trainer for torchvision ResNet34."""

    def create_model(self) -> nn.Module:
        """Create a ResNet34 classifier with 10 classes."""
        return resnet34(num_classes=10)
