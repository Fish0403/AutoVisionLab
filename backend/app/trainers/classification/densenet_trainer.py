"""DenseNet trainer implementations."""

from torch import nn
from torchvision.models import densenet121

from app.trainers.classification.base_trainer import BaseClassificationTrainer


class DenseNet121Trainer(BaseClassificationTrainer):
    """Trainer for torchvision DenseNet121."""

    def create_model(self) -> nn.Module:
        """Create a DenseNet121 classifier with 10 classes."""
        return densenet121(num_classes=10)
