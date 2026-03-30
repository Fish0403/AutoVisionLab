"""Tests for classification trainer dataloader configuration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.base_trainer import BaseClassificationTrainer


def _build_config_payload() -> dict[str, object]:
    """Build a minimal classification config payload for dataloader tests."""
    return {
        "task_type": "classification",
        "dataset": "temp-dataset",
        "model_family": "mobilenet",
        "model_name": "mobilenet_v3_small",
        "parameter_space_version": "mobilenet_v3_small@v1",
        "use_demo_mode": False,
        "params": {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 2,
            "image_size": 32,
            "epochs": 1,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "augmentation_policy": "basic",
            "augmentation_params": {
                "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0,
                "random_erasing_prob": 0.0,
            },
            "loss_name": "cross_entropy_with_label_smoothing",
            "loss_params": {"focal_gamma": 2.0},
            "label_smoothing": 0.1,
            "aux_logits": False,
        },
    }


class ClassificationDataloaderConfigTest(unittest.TestCase):
    """Verify dataloader worker settings are applied."""

    def test_build_dataloaders_uses_configured_worker_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            classification_root = temp_root / "classification" / "temp-dataset"
            raw_root = temp_root / "raw" / "temp-dataset"
            classification_root.mkdir(parents=True)
            raw_root.mkdir(parents=True)

            train_manifest = classification_root / "train.txt"
            val_manifest = classification_root / "val.txt"

            samples = [
                ("sample_a.png", "class_a"),
                ("sample_b.png", "class_b"),
                ("sample_c.png", "class_a"),
                ("sample_d.png", "class_b"),
            ]
            for image_name, _ in samples:
                image = Image.new("RGB", (16, 16), color=(128, 128, 128))
                image.save(raw_root / image_name)

            train_manifest.write_text(
                "\n".join(f"{image_name}\t{label}" for image_name, label in samples[:3]) + "\n",
                encoding="utf-8",
            )
            val_manifest.write_text(
                "\n".join(f"{image_name}\t{label}" for image_name, label in samples[1:]) + "\n",
                encoding="utf-8",
            )

            config = ExperimentConfig.model_validate(_build_config_payload())
            trainer = BaseClassificationTrainer(
                config=config,
                experiment_id="exp_test_loader",
                run_id="run_test_loader",
            )
            trainer.data_root = temp_root
            trainer.device = torch.device("cuda")
            trainer.settings.classification_num_workers = 2

            train_loader, val_loader = trainer.build_dataloaders()

        self.assertEqual(train_loader.num_workers, 2)
        self.assertEqual(val_loader.num_workers, 2)
        self.assertTrue(train_loader.pin_memory)
        self.assertTrue(val_loader.pin_memory)
        self.assertTrue(train_loader.persistent_workers)
        self.assertTrue(val_loader.persistent_workers)


if __name__ == "__main__":
    unittest.main()
