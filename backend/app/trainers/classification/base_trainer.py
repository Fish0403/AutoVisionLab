"""Base trainer interface for classification experiments."""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset

from app.core.settings import get_settings
from app.schemas.ai import ResultSchema
from app.schemas.common import ArtifactPaths, MetricsSnapshot, ResourceUsage
from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.data_loading import (
    ManifestClassificationDataset,
    build_class_index,
    resolve_classification_dataset_files,
)
from app.trainers.classification.components import (
    apply_batch_augmentations,
    build_eval_transform,
    build_loss,
    build_train_transform,
)
from app.trainers.classification.model_adapters import (
    ClassificationModelAdapter,
    get_classification_model_adapter,
)


DEMO_SUBSET_SEED = 42


class TrainingInterruptedError(RuntimeError):
    """Raised when the current experiment is stopped and discarded."""


class BaseClassificationTrainer:
    """Generic classification trainer driven by one model adapter."""

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_id: str,
        run_id: str,
        should_stop: Callable[[], bool] | None = None,
        model_adapter: ClassificationModelAdapter | None = None,
    ) -> None:
        self.config = config
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.should_stop = should_stop or (lambda: False)
        self.model_adapter = model_adapter or get_classification_model_adapter(config.model_name)
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.artifact_root = Path(self.settings.artifact_root)
        self.log_path = self.artifact_root / "runs" / f"{self.run_id}.log"
        self.checkpoint_path = self.artifact_root / "checkpoints" / f"{self.experiment_id}.pt"
        self.data_root = Path(self.settings.data_root)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def create_model(self) -> nn.Module:
        """Create the model for this trainer."""
        return self.model_adapter.create_model(self.config)

    def count_model_parameters_million(self, model: nn.Module) -> float:
        """Return trainable parameter count in millions."""
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return round(parameter_count / 1_000_000, 4)

    def measure_inference_latency_ms(self, model: nn.Module) -> float:
        """Run a tiny inference benchmark on the current device."""
        input_channels = self.config.model_recipe.input_channels if self.config.model_recipe is not None else 3
        image_size = self.config.train_hyp.image_size
        dummy_input = torch.randn(1, input_channels, image_size, image_size, device=self.device)
        warmup_steps = 2
        timed_steps = 5

        was_training = model.training
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup_steps):
                _ = model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            started_at = time.perf_counter()
            for _ in range(timed_steps):
                _ = model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / timed_steps
        if was_training:
            model.train()
        return round(elapsed_ms, 4)

    def train(self) -> ResultSchema:
        """Execute training and return a structured result."""
        train_loader = None
        val_loader = None
        model = None
        criterion = None
        optimizer = None
        scheduler = None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        try:
            train_loader, val_loader = self.build_dataloaders()
            model = self.create_model().to(self.device)
            criterion = build_loss(self.config.train_hyp)
            optimizer = self.build_optimizer(model)
            scheduler = self.build_scheduler(optimizer)

            best_val_loss = float("inf")
            best_epoch = 1
            best_top1_acc = 0.0
            final_train_loss = 0.0
            started_at = time.time()

            with self.log_path.open("a", encoding="utf-8") as log_file:
                self.write_log(log_file, f"Starting training on device={self.device}")
                for epoch in range(1, self.config.train_hyp.epochs + 1):
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
            parameter_count_million = self.count_model_parameters_million(model)
            latency_ms = self.measure_inference_latency_ms(model)

            return ResultSchema(
                status="success",
                metrics=MetricsSnapshot(
                    train_loss=round(final_train_loss, 4),
                    val_loss=round(best_val_loss, 4),
                    top1_acc=round(best_top1_acc, 4),
                    best_epoch=best_epoch,
                ),
                resource=ResourceUsage(
                    gpu_memory_mb=gpu_memory_mb,
                    training_seconds=training_seconds,
                    latency_ms=latency_ms,
                    parameter_count_million=parameter_count_million,
                ),
                params=self.config.result_params(),
                artifacts=ArtifactPaths(
                    log_path=str(self.log_path),
                    checkpoint_path=str(self.checkpoint_path),
                ),
            )
        finally:
            del scheduler
            del optimizer
            del criterion
            del model
            del train_loader
            del val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build dataloaders for the configured classification dataset."""
        image_size = self.config.train_hyp.image_size
        train_transform = build_train_transform(
            image_size=image_size,
            augmentation=self.config.train_hyp.augmentation,
            dataset_name=self.config.dataset,
        )
        eval_transform = build_eval_transform(image_size, dataset_name=self.config.dataset)
        train_manifest, val_manifest, source_root = resolve_classification_dataset_files(
            self.data_root,
            self.config.dataset.strip(),
        )
        class_to_idx = build_class_index(train_manifest, val_manifest)
        train_dataset = ManifestClassificationDataset(
            manifest_path=train_manifest,
            source_root=source_root,
            class_to_idx=class_to_idx,
            transform=train_transform,
        )
        val_dataset = ManifestClassificationDataset(
            manifest_path=val_manifest,
            source_root=source_root,
            class_to_idx=class_to_idx,
            transform=eval_transform,
        )
        if self.config.use_demo_mode:
            train_dataset = self.build_demo_subset(
                train_dataset,
                max_samples=self.settings.demo_train_samples,
                seed=DEMO_SUBSET_SEED,
            )
            val_dataset = self.build_demo_subset(
                val_dataset,
                max_samples=self.settings.demo_val_samples,
                seed=DEMO_SUBSET_SEED + 1,
            )
        batch_size = self.config.train_hyp.batch_size
        num_workers = max(int(self.settings.classification_num_workers), 0)
        common_dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            common_dataloader_kwargs["persistent_workers"] = True
        return (
            DataLoader(train_dataset, shuffle=True, **common_dataloader_kwargs),
            DataLoader(val_dataset, shuffle=False, **common_dataloader_kwargs),
        )

    def build_demo_subset(
        self,
        dataset: Dataset,
        *,
        max_samples: int,
        seed: int,
    ) -> Dataset:
        """Build a deterministic demo subset with class coverage when possible."""
        if len(dataset) <= max_samples:
            return dataset
        subset_indices = self.build_demo_subset_indices(dataset, max_samples=max_samples, seed=seed)
        return Subset(dataset, subset_indices)

    def build_demo_subset_indices(
        self,
        dataset: Dataset,
        *,
        max_samples: int,
        seed: int,
    ) -> list[int]:
        """Build deterministic subset indices for demo mode."""
        targets = getattr(dataset, "targets", None)
        if not isinstance(targets, list) or len(targets) != len(dataset):
            shuffled_indices = list(range(len(dataset)))
            random.Random(seed).shuffle(shuffled_indices)
            return shuffled_indices[:max_samples]

        rng = random.Random(seed)
        indices_by_class: dict[int, list[int]] = {}
        for index, target in enumerate(targets):
            indices_by_class.setdefault(int(target), []).append(index)
        for class_indices in indices_by_class.values():
            rng.shuffle(class_indices)

        selected_indices: list[int] = []
        max_class_length = max(len(class_indices) for class_indices in indices_by_class.values())
        for position in range(max_class_length):
            for class_id in sorted(indices_by_class):
                class_indices = indices_by_class[class_id]
                if position >= len(class_indices):
                    continue
                selected_indices.append(class_indices[position])
                if len(selected_indices) >= max_samples:
                    return selected_indices
        return selected_indices[:max_samples]

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build the optimizer from structured params."""
        common_kwargs = {"lr": self.config.train_hyp.lr0, "weight_decay": self.config.train_hyp.weight_decay}
        if self.config.train_hyp.optimizer == "sgd":
            return SGD(model.parameters(), momentum=self.config.train_hyp.momentum, **common_kwargs)
        if self.config.train_hyp.optimizer == "adam":
            return Adam(model.parameters(), **common_kwargs)
        return AdamW(model.parameters(), **common_kwargs)

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        """Build the scheduler from structured params."""
        if self.config.train_hyp.scheduler == "step":
            return StepLR(optimizer, step_size=max(self.config.train_hyp.epochs // 3, 1), gamma=0.1)
        if self.config.train_hyp.scheduler == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.train_hyp.epochs,
                eta_min=self.config.train_hyp.lr0 * self.config.train_hyp.lrf,
            )
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
            images, labels = apply_batch_augmentations(images, labels, self.config.train_hyp.augmentation)
            optimizer.zero_grad()
            outputs = self.forward_train(model, images)
            loss = self.compute_loss(outputs, labels, criterion)
            loss.backward()
            if self.config.train_hyp.runtime.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.train_hyp.runtime.grad_clip_norm)
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
        return self.model_adapter.compute_loss(
            self.config,
            outputs,
            labels,
            criterion,
            self.unwrap_logits,
        )

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
