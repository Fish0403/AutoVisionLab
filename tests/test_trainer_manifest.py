"""Tests for unified trainer manifest parser, validator, and builder entrypoints."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import (
    ExperimentConfig,
    ExperimentParams,
    build_default_model_recipe,
)
from app.trainers.builder import build_model_from_manifest
from app.trainers.factory import build_trainer_from_config
from app.trainers.parser import load_trainer_manifest, parse_trainer_manifest_payload
from app.trainers.validator import validate_trainer_manifest
from tests.helpers.experiment_config_builders import build_default_dataset_recipe, build_train_hyp_from_params


def _build_config_payload(model_name: str, model_family: str, parameter_space_version: str) -> dict[str, object]:
    """Build one minimal experiment config payload."""
    params = ExperimentParams.model_validate(
        {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 64,
            "image_size": 96,
            "epochs": 10,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "augmentation_policy": "basic",
            "augmentation_params": {
                "mixup_alpha": 0.2,
                "cutmix_alpha": 0.0,
                "random_erasing_prob": 0.1,
            },
            "loss_name": "cross_entropy_with_label_smoothing",
            "loss_params": {"focal_gamma": 2.0},
            "label_smoothing": 0.1,
            "aux_logits": False,
        }
    )
    return {
        "task_type": "classification",
        "dataset": "neu",
        "model_family": model_family,
        "model_name": model_name,
        "parameter_space_version": parameter_space_version,
        "params": params.model_dump(),
        "model_recipe": build_default_model_recipe(
            model_name=model_name,
            task_type="classification",
            model_family=model_family,
        ).model_dump(by_alias=True),
        "train_hyp": build_train_hyp_from_params(
            task_type="classification",
            params=params,
        ).model_dump(),
        "dataset_recipe": build_default_dataset_recipe(
            dataset_name="neu",
            task_type="classification",
        ).model_dump(),
    }


class TrainerManifestTest(unittest.TestCase):
    """Verify the unified manifest entrypoints."""

    def test_load_manifest_parses_short_layer_form(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.yaml"
            manifest_path.write_text(
                """
version: trainer_manifest@v1
task_type: classification
parameter_space_version: mobilenet_v3_small@v1
model:
  version: model_recipe@v1
  task_type: classification
  model_family: mobilenet
  base_model: mobilenet_v3_small
  head_config:
    pooling_type: avg
    classifier_dropout: 0.2
    classifier_type: linear
  backbone:
    - [-1, 1, stem_conv, [16, 3, 2, hardswish], stem]
    - [-1, 1, inverted_residual, [16, 3, 16, 16, true, RE, 2, 1], C1]
  head:
    - [-1, 1, pointwise_tail, [6, hardswish], tail]
    - [-1, 1, global_pool, [], pool]
    - [-1, 1, classifier, [1024], cls]
train:
  version: train_hyp@v1
  task_type: classification
  optimizer: adamw
  lr0: 0.003
  weight_decay: 0.0001
  scheduler: cosine
  epochs: 10
  batch_size: 64
  image_size: 96
data:
  version: dataset_recipe@v1
  task_type: classification
  dataset_name: neu
  source:
    root_dir: data/raw/neu
  splits:
    train_manifest: data/classification/neu/train.txt
    val_manifest: data/classification/neu/val.txt
search:
  allow_model_module_search: true
""".strip(),
                encoding="utf-8",
            )

            manifest = load_trainer_manifest(manifest_path)

        self.assertEqual(manifest.model_name, "mobilenet_v3_small")
        self.assertEqual(manifest.model.backbone[0].from_indices, -1)
        self.assertEqual(manifest.model.backbone[0].repeat, 1)
        self.assertEqual(manifest.model.head[2].tag, "cls")
        self.assertTrue(manifest.search.allow_model_module_search)

    def test_validator_rejects_unsupported_googlenet_architecture_override(self) -> None:
        manifest = parse_trainer_manifest_payload(
            {
                "version": "trainer_manifest@v1",
                "task_type": "classification",
                "parameter_space_version": "googlenet@v1",
                "model": {
                    "version": "model_recipe@v1",
                    "task_type": "classification",
                    "model_family": "googlenet",
                    "base_model": "googlenet",
                    "backbone": [[-1, 1, "inception", [64]]],
                },
                "train": {
                    "version": "train_hyp@v1",
                    "task_type": "classification",
                    "optimizer": "adamw",
                    "lr0": 0.003,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "epochs": 10,
                    "batch_size": 32,
                    "image_size": 96,
                },
                "data": {
                    "version": "dataset_recipe@v1",
                    "task_type": "classification",
                    "dataset_name": "neu",
                    "source": {"root_dir": "data/raw/neu"},
                    "splits": {
                        "train_manifest": "data/classification/neu/train.txt",
                        "val_manifest": "data/classification/neu/val.txt",
                    },
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "custom architecture layers"):
            validate_trainer_manifest(manifest)

    def test_parser_rejects_legacy_model_architecture_section(self) -> None:
        with self.assertRaisesRegex(ValueError, "model.architecture"):
            parse_trainer_manifest_payload(
                {
                    "version": "trainer_manifest@v1",
                    "task_type": "classification",
                    "parameter_space_version": "mobilenet_v3_small@v1",
                    "model": {
                        "version": "model_recipe@v1",
                        "task_type": "classification",
                        "model_family": "mobilenet",
                        "base_model": "mobilenet_v3_small",
                        "architecture": {
                            "backbone": [[-1, 1, "stem_conv", [16, 3, 2, "hardswish"], "stem"]],
                        },
                    },
                    "train": {
                        "version": "train_hyp@v1",
                        "task_type": "classification",
                        "optimizer": "adamw",
                        "lr0": 0.003,
                        "weight_decay": 0.0001,
                        "scheduler": "cosine",
                        "epochs": 10,
                        "batch_size": 32,
                        "image_size": 96,
                    },
                    "data": {
                        "version": "dataset_recipe@v1",
                        "task_type": "classification",
                        "dataset_name": "neu",
                        "source": {"root_dir": "data/raw/neu"},
                        "splits": {
                            "train_manifest": "data/classification/neu/train.txt",
                            "val_manifest": "data/classification/neu/val.txt",
                        },
                    },
                }
            )

    def test_build_model_from_manifest_dispatches_to_resnet_builder(self) -> None:
        config = ExperimentConfig.model_validate(_build_config_payload("resnet18", "resnet", "resnet18@v1"))
        manifest = parse_trainer_manifest_payload(
            {
                "version": "trainer_manifest@v1",
                "task_type": "classification",
                "parameter_space_version": config.parameter_space_version,
                "model": config.model_recipe.model_dump(by_alias=True),
                "train": config.train_hyp.model_dump(),
                "data": config.dataset_recipe.model_dump(),
                "search": config.search_policy.model_dump(),
                "ranking": config.ranking_policy.model_dump(),
                "runtime": {
                    "use_demo_mode": config.use_demo_mode,
                    "participates_in_ranking": config.participates_in_ranking,
                },
            }
        )

        model = build_model_from_manifest(manifest)
        output = model(torch.zeros(1, 3, 96, 96))

        self.assertEqual(model.__class__.__name__, "GenericTorchvisionClassifier")
        self.assertEqual(tuple(output.shape), (1, 6))

    def test_build_model_from_manifest_dispatches_to_mobilenet_v2_builder(self) -> None:
        config = ExperimentConfig.model_validate(_build_config_payload("mobilenet_v2", "mobilenet", "mobilenet_v2@v1"))
        manifest = parse_trainer_manifest_payload(
            {
                "version": "trainer_manifest@v1",
                "task_type": "classification",
                "parameter_space_version": config.parameter_space_version,
                "model": config.model_recipe.model_dump(by_alias=True),
                "train": config.train_hyp.model_dump(),
                "data": config.dataset_recipe.model_dump(),
                "search": config.search_policy.model_dump(),
                "ranking": config.ranking_policy.model_dump(),
                "runtime": {
                    "use_demo_mode": config.use_demo_mode,
                    "participates_in_ranking": config.participates_in_ranking,
                },
            }
        )

        model = build_model_from_manifest(manifest)
        output = model(torch.zeros(1, 3, 96, 96))

        self.assertEqual(model.__class__.__name__, "GenericTorchvisionClassifier")
        self.assertEqual(tuple(output.shape), (1, 6))
        self.assertEqual(model.head[0].p, 0.2)

    def test_build_model_from_manifest_infers_output_classes_from_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_manifest = temp_root / "train.txt"
            val_manifest = temp_root / "val.txt"
            train_manifest.write_text(
                "sample_a.jpg\tclass_a\nsample_b.jpg\tclass_b\nsample_c.jpg\tclass_c\n",
                encoding="utf-8",
            )
            val_manifest.write_text(
                "sample_d.jpg\tclass_a\nsample_e.jpg\tclass_b\n",
                encoding="utf-8",
            )

            manifest = parse_trainer_manifest_payload(
                {
                    "version": "trainer_manifest@v1",
                    "task_type": "classification",
                    "parameter_space_version": "mobilenet_v3_small@v1",
                    "model": {
                        "version": "model_recipe@v1",
                        "task_type": "classification",
                        "model_family": "mobilenet",
                        "base_model": "mobilenet_v3_small",
                    },
                    "train": {
                        "version": "train_hyp@v1",
                        "task_type": "classification",
                        "optimizer": "adamw",
                        "lr0": 0.003,
                        "weight_decay": 0.0001,
                        "scheduler": "cosine",
                        "epochs": 10,
                        "batch_size": 32,
                        "image_size": 96,
                    },
                    "data": {
                        "version": "dataset_recipe@v1",
                        "task_type": "classification",
                        "dataset_name": "temp-dataset",
                        "source": {"root_dir": str(temp_root)},
                        "splits": {
                            "train_manifest": str(train_manifest),
                            "val_manifest": str(val_manifest),
                        },
                    },
                }
            )

            model = build_model_from_manifest(manifest)

        self.assertEqual(model.head[-1].out_features, 3)

    def test_build_trainer_from_config_dispatches_to_trainer_factory(self) -> None:
        payload = _build_config_payload("googlenet", "googlenet", "googlenet@v1")
        payload["model_recipe"] = {
            "version": "model_recipe@v1",
            "task_type": "classification",
            "model_family": "googlenet",
            "base_model": "googlenet",
            "modules": {"aux_logits": True},
        }
        config = ExperimentConfig.model_validate(payload)

        trainer = build_trainer_from_config(config, experiment_id="exp_1", run_id="run_1")

        self.assertEqual(trainer.model_adapter.name, "googlenet")


if __name__ == "__main__":
    unittest.main()
