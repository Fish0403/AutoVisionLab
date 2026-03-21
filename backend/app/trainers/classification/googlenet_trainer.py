"""GoogLeNet trainer implementation."""

import torch
from torch import nn
from torchvision.models import googlenet

from app.trainers.classification.base_trainer import BaseClassificationTrainer


class GoogLeNetTrainer(BaseClassificationTrainer):
    """Trainer for torchvision GoogLeNet."""

    def create_model(self) -> nn.Module:
        """Create a GoogLeNet classifier with 10 classes."""
        return googlenet(num_classes=10, aux_logits=bool(self.config.params.aux_logits))

    def compute_loss(self, outputs, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Include auxiliary head losses when aux logits are enabled."""
        if self.config.params.aux_logits and hasattr(outputs, "aux_logits1") and hasattr(outputs, "aux_logits2"):
            main_loss = criterion(outputs.logits, labels)
            aux1_loss = criterion(outputs.aux_logits1, labels)
            aux2_loss = criterion(outputs.aux_logits2, labels)
            return main_loss + 0.3 * (aux1_loss + aux2_loss)
        return super().compute_loss(outputs, labels, criterion)
