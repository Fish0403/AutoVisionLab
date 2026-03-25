"""Base trainer interface for classification experiments."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from app.core.settings import get_settings
from app.schemas.ai import ResultSchema
from app.schemas.common import ArtifactPaths, MetricsSnapshot, ResourceUsage
from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.components import (
    apply_batch_augmentations,
    build_eval_transform,
    build_loss,
    build_train_transform,
)


class TrainingInterruptedError(RuntimeError):
    """Raised when the current experiment is stopped and discarded."""


class BaseClassificationTrainer(ABC):
    """Abstract trainer that only accepts structured experiment config."""

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_id: str,
        run_id: str,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.should_stop = should_stop or (lambda: False)
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.artifact_root = Path(self.settings.artifact_root)
        self.log_path = self.artifact_root / "runs" / f"{self.run_id}.log"
        self.checkpoint_path = self.artifact_root / "checkpoints" / f"{self.experiment_id}.pt"
        self.data_root = Path(self.settings.data_root)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model for this trainer."""

    def train(self) -> ResultSchema:
        """Execute training and return a structured result."""
        train_loader, val_loader = self.build_dataloaders()
        model = self.create_model().to(self.device)
        criterion = build_loss(self.config.params)
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        best_val_loss = float("inf")
        best_epoch = 1
        best_top1_acc = 0.0
        final_train_loss = 0.0
        started_at = time.time()

        with self.log_path.open("a", encoding="utf-8") as log_file:
            self.write_log(log_file, f"Starting training on device={self.device}")
            for epoch in range(1, self.config.params.epochs + 1):
                self.raise_if_stopped(log_file)
                final_train_loss = self.train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, top1_acc = self.evaluate(model, val_loader, criterion)
                if scheduler is not None:
                    scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_top1_acc = top1_acc
                    torch.save({"model_state_dict": model.state_dict()}, self.checkpoint_path)

                self.write_log(
                    log_file,
                    (
                        f"epoch={epoch} train_loss={final_train_loss:.4f} "
                        f"val_loss={val_loss:.4f} top1_acc={top1_acc:.4f}"
                    ),
                )
        training_seconds = int(time.time() - started_at)
        gpu_memory_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0

        return ResultSchema(
            status="success",
            metrics=MetricsSnapshot(
                train_loss=round(final_train_loss, 4),
                val_loss=round(best_val_loss, 4),
                top1_acc=round(best_top1_acc, 4),
                best_epoch=best_epoch,
            ),
            resource=ResourceUsage(gpu_memory_mb=gpu_memory_mb, training_seconds=training_seconds),
            params=self.config.params,
            artifacts=ArtifactPaths(
                log_path=str(self.log_path),
                checkpoint_path=str(self.checkpoint_path),
            ),
        )

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build dataloaders for the configured classification dataset."""
        image_size = self.config.params.image_size
        train_transform = build_train_transform(
            image_size=image_size,
            augmentation_policy=self.config.params.augmentation_policy,
            augmentation_params=self.config.params.augmentation_params,
        )
        eval_transform = build_eval_transform(image_size)
        train_dir, val_dir = self.resolve_classification_dataset_dirs(self.config.dataset.strip().lower())
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=eval_transform)
        if self.settings.is_demo_mode:
            train_dataset = Subset(train_dataset, range(min(len(train_dataset), self.settings.demo_train_samples)))
            val_dataset = Subset(val_dataset, range(min(len(val_dataset), self.settings.demo_val_samples)))
        batch_size = self.config.params.batch_size
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        )

    def resolve_classification_dataset_dirs(self, dataset_name: str) -> tuple[Path, Path]:
        """Resolve ImageFolder train/val directories for one dataset."""
        dataset_root = self.data_root / dataset_name / "classification"
        train_dir = dataset_root / "train"
        val_dir = dataset_root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                "Classification dataset directory not found. Expected: "
                f"{train_dir} and {val_dir}"
            )
        return train_dir, val_dir

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build the optimizer from structured params."""
        common_kwargs = {"lr": self.config.params.learning_rate, "weight_decay": self.config.params.weight_decay}
        if self.config.params.optimizer == "sgd":
            return SGD(model.parameters(), momentum=0.9, **common_kwargs)
        if self.config.params.optimizer == "adam":
            return Adam(model.parameters(), **common_kwargs)
        return AdamW(model.parameters(), **common_kwargs)

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        """Build the scheduler from structured params."""
        if self.config.params.scheduler == "step":
            return StepLR(optimizer, step_size=max(self.config.params.epochs // 3, 1), gamma=0.1)
        if self.config.params.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.config.params.epochs)
        return None

    def train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train the model for one epoch."""
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            self.raise_if_stopped()
            images = images.to(self.device)
            labels = labels.to(self.device)
            images, labels = apply_batch_augmentations(images, labels, self.config.params.augmentation_params)
            optimizer.zero_grad()
            outputs = self.forward_train(model, images)
            loss = self.compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
        return total_loss / max(sample_count, 1)

    def evaluate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """Evaluate the model on the validation split."""
        model.eval()
        total_loss = 0.0
        correct = 0
        sample_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                self.raise_if_stopped()
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward_eval(model, images)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (predictions == labels).sum().item()
                sample_count += batch_size
        return total_loss / max(sample_count, 1), correct / max(sample_count, 1)

    def forward_train(self, model: nn.Module, images: torch.Tensor):
        """Run the train forward pass."""
        return model(images)

    def forward_eval(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Run the eval forward pass and normalize model outputs."""
        outputs = model(images)
        return self.unwrap_logits(outputs)

    def compute_loss(self, outputs, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Compute the loss for one training step."""
        return criterion(self.unwrap_logits(outputs), labels)

    def unwrap_logits(self, outputs) -> torch.Tensor:
        """Normalize model-specific outputs into logits."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def write_log(self, log_file, message: str) -> None:
        """Write one log line."""
        log_file.write(f"[{self.experiment_id}] {message}\n")
        log_file.flush()

    def raise_if_stopped(self, log_file=None) -> None:
        """Abort training immediately when stop is requested."""
        if not self.should_stop():
            return
        if log_file is not None:
            self.write_log(log_file, "Training interrupted by user request.")
        raise TrainingInterruptedError(f"Experiment {self.experiment_id} was discarded by user request")
