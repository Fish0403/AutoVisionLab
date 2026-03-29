"""Dataset resolution and manifest-backed loading for classification trainers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from app.services.dataset_paths import PREPARED_SOURCE_DIRNAME, resolve_dataset_dir


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


def collect_manifest_classes(manifest_path: Path) -> set[str]:
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


def build_class_index(train_manifest: Path, val_manifest: Path) -> dict[str, int]:
    """Build a stable class index from the train and val manifests."""
    class_names = sorted(collect_manifest_classes(train_manifest) | collect_manifest_classes(val_manifest))
    if not class_names:
        raise ValueError(
            f"No classes found in manifests: train={train_manifest}, val={val_manifest}"
        )
    return {class_name: index for index, class_name in enumerate(class_names)}


def resolve_classification_dataset_files(data_root: Path, dataset_name: str) -> tuple[Path, Path, Path]:
    """Resolve manifest files and source root for one classification dataset."""
    manifest_root = resolve_dataset_dir(
        parent_dir=data_root / "classification",
        dataset_name=dataset_name,
        strict=True,
    )
    train_manifest = manifest_root / "train.txt"
    val_manifest = manifest_root / "val.txt"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            "Classification manifests not found. Expected: "
            f"{train_manifest} and {val_manifest}"
        )

    raw_root = resolve_dataset_dir(
        parent_dir=data_root / "raw",
        dataset_name=dataset_name,
        strict=True,
    )
    prepared_source_root = raw_root / PREPARED_SOURCE_DIRNAME
    source_root = prepared_source_root if prepared_source_root.exists() else raw_root
    return train_manifest, val_manifest, source_root
