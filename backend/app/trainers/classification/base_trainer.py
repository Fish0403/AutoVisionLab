"""Base trainer interface for classification experiments."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import default_loader

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


PREPARED_SOURCE_DIRNAME = "classification_source"
DEMO_SUBSET_SEED = 42


class TrainingInterruptedError(RuntimeError):
    """Raised when the current experiment is stopped and discarded."""


class ManifestClassificationDataset(Dataset):
    """Classification dataset backed by a split manifest file."""

    def __init__(
        self,
        manifest_path: Path,
        source_root: Path,
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.manifest_path = manifest_path
        self.source_root = source_root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.samples = self._load_samples()
        self.targets = [target for _, target in self.samples]

    def _load_samples(self) -> list[tuple[Path, int]]:
        """Load manifest lines into path and class index tuples."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Classification manifest not found: {self.manifest_path}")

        samples: list[tuple[Path, int]] = []
        for line_number, raw_line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                relative_path_text, class_name = line.rsplit("\t", maxsplit=1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest line at {self.manifest_path}:{line_number}: {raw_line!r}"
                    )
                relative_path_text, class_name = parts
            if class_name not in self.class_to_idx:
                raise ValueError(
                    f"Unknown class {class_name!r} found in manifest {self.manifest_path}:{line_number}"
                )
            image_path = self.source_root / Path(relative_path_text)
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Manifest points to a missing image: {image_path} "
                    f"(from {self.manifest_path}:{line_number})"
                )
            samples.append((image_path, self.class_to_idx[class_name]))
        if not samples:
            raise ValueError(f"Classification manifest is empty: {self.manifest_path}")
        return samples

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one sample from the manifest."""
        image_path, label = self.samples[index]
        image = default_loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
        train_manifest, val_manifest, source_root = self.resolve_classification_dataset_files(self.config.dataset.strip())
        class_to_idx = self.build_class_index(train_manifest, val_manifest)
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
        batch_size = self.config.params.batch_size
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
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

    def build_class_index(self, train_manifest: Path, val_manifest: Path) -> dict[str, int]:
        """Build a stable class index from the train and val manifests."""
        class_names = sorted(self.collect_manifest_classes(train_manifest) | self.collect_manifest_classes(val_manifest))
        if not class_names:
            raise ValueError(
                f"No classes found in manifests: train={train_manifest}, val={val_manifest}"
            )
        return {class_name: index for index, class_name in enumerate(class_names)}

    def collect_manifest_classes(self, manifest_path: Path) -> set[str]:
        """Collect class names from one manifest file."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Classification manifest not found: {manifest_path}")

        class_names: set[str] = set()
        for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                _, class_name = line.rsplit("\t", maxsplit=1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest line at {manifest_path}:{line_number}: {raw_line!r}"
                    )
                _, class_name = parts
            class_names.add(class_name)
        return class_names

    def resolve_classification_dataset_files(self, dataset_name: str) -> tuple[Path, Path, Path]:
        """Resolve manifest files and source root for one classification dataset."""
        manifest_root = self.resolve_dataset_dir(parent_dir=self.data_root / "classification", dataset_name=dataset_name)
        train_manifest = manifest_root / "train.txt"
        val_manifest = manifest_root / "val.txt"
        if not train_manifest.exists() or not val_manifest.exists():
            raise FileNotFoundError(
                "Classification manifests not found. Expected: "
                f"{train_manifest} and {val_manifest}"
            )

        raw_root = self.resolve_dataset_dir(parent_dir=self.data_root / "raw", dataset_name=dataset_name)
        prepared_source_root = raw_root / PREPARED_SOURCE_DIRNAME
        source_root = prepared_source_root if prepared_source_root.exists() else raw_root
        return train_manifest, val_manifest, source_root

    def resolve_dataset_dir(self, parent_dir: Path, dataset_name: str) -> Path:
        """Resolve one dataset directory with alias and case-insensitive fallback."""
        if not parent_dir.exists():
            raise FileNotFoundError(f"Dataset parent directory not found: {parent_dir}")

        candidate_names = [dataset_name]

        for candidate_name in candidate_names:
            candidate_dir = parent_dir / candidate_name
            if candidate_dir.exists():
                return candidate_dir

        normalized_candidates = {self.normalize_dataset_name(name) for name in candidate_names}
        for child_dir in sorted(path for path in parent_dir.iterdir() if path.is_dir()):
            if self.normalize_dataset_name(child_dir.name) in normalized_candidates:
                return child_dir

        raise FileNotFoundError(
            f"Dataset directory not found under {parent_dir} for dataset={dataset_name!r}"
        )

    def normalize_dataset_name(self, dataset_name: str) -> str:
        """Normalize one dataset name for tolerant directory lookup."""
        return dataset_name.strip().lower().replace("_", "-")

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
