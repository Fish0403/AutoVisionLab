"""Tests for recipe-aware experiment config defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import ExperimentConfig
from app.trainers.templates import load_builtin_model_recipe_payload


def _build_config_payload() -> dict[str, object]:
    """Build one minimal legacy experiment config payload."""
    return {
        "task_type": "classification",
        "dataset": "neu",
        "model_family": "mobilenet",
        "model_name": "mobilenet_v3_small",
        "parameter_space_version": "mobilenet_v3_small@v1",
        "params": {
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
        },
    }


class ExperimentConfigRecipeTest(unittest.TestCase):
    """Verify recipe objects are attached to experiment config payloads."""

    def test_builtin_mobilenet_template_hides_architecture_details(self) -> None:
        template_payload = load_builtin_model_recipe_payload("mobilenet_v3_small")

        self.assertNotIn("backbone", template_payload)
        self.assertNotIn("head", template_payload)

    def test_legacy_payload_is_backfilled_with_default_recipes(self) -> None:
        config = ExperimentConfig.model_validate(_build_config_payload())

        self.assertIsNotNone(config.model_recipe)
        self.assertEqual(config.model_recipe.base_model, "mobilenet_v3_small")
        self.assertEqual(config.model_recipe.head_config.classifier_dropout, 0.2)
        self.assertEqual(config.model_recipe.components.backbone.name, "mobilenet_v3_small_native")
        self.assertEqual(config.model_recipe.components.neck.name, "avg_pool")
        self.assertEqual(config.model_recipe.components.head.name, "native_classifier")
        self.assertEqual(config.model_recipe.backbone[0].module, "stem_conv")
        self.assertEqual(config.model_recipe.backbone[0].args, [16, 3, 2, "hardswish"])
        self.assertEqual(len(config.model_recipe.backbone), 12)
        self.assertEqual(config.model_recipe.backbone[1].module, "inverted_residual")
        self.assertIsNotNone(config.train_hyp)
        self.assertEqual(config.train_hyp.lr0, 0.003)
        self.assertEqual(config.train_hyp.augmentation.mixup, 0.2)
        self.assertIsNotNone(config.dataset_recipe)
        self.assertEqual(config.dataset_recipe.dataset_name, "neu")
        self.assertEqual(len(config.dataset_recipe.class_names), 6)
        self.assertEqual(config.model_recipe.nc, 6)
        self.assertEqual(
            config.dataset_recipe.splits.train_manifest,
            "data/classification/neu/train.txt",
        )

    def test_default_dataset_recipe_infers_dt_class_names_and_model_output_classes(self) -> None:
        payload = _build_config_payload()
        payload["dataset"] = "DT"
        config = ExperimentConfig.model_validate(payload)

        self.assertEqual(config.dataset_recipe.class_names, ["0", "1", "2"])
        self.assertEqual(config.model_recipe.nc, 3)

    def test_mobilenet_v2_defaults_attach_native_components_and_dropout(self) -> None:
        payload = _build_config_payload()
        payload["model_name"] = "mobilenet_v2"
        payload["parameter_space_version"] = "mobilenet_v2@v1"
        config = ExperimentConfig.model_validate(payload)

        self.assertEqual(config.model_recipe.base_model, "mobilenet_v2")
        self.assertEqual(config.model_recipe.components.backbone.name, "mobilenet_v2_native")
        self.assertEqual(config.model_recipe.components.neck.name, "avg_pool")
        self.assertEqual(config.model_recipe.components.head.name, "native_classifier")
        self.assertEqual(config.model_recipe.head_config.classifier_dropout, 0.2)
        self.assertEqual(config.model_recipe.nc, 6)

    def test_explicit_recipe_payloads_are_preserved(self) -> None:
        payload = _build_config_payload()
        payload["model_recipe"] = {
            "version": "model_recipe@v1",
            "task_type": "classification",
            "model_family": "mobilenet",
            "base_model": "mobilenet_v3_small",
            "width_multiple": 1.25,
            "backbone_config": {
                "stem_variant": "deep_stem",
                "attention_module": "se",
                "last_channel_multiplier": 1.0,
            },
            "head_config": {
                "pooling_type": "gem",
                "classifier_dropout": 0.2,
                "classifier_type": "linear",
            },
            "components": {
                "backbone": {"name": "mobilenet_v3_small_native", "params": {}},
                "neck": {"name": "gem_pool", "params": {}},
                "head": {"name": "native_classifier", "params": {}},
            },
        }
        payload["train_hyp"] = {
            "version": "train_hyp@v1",
            "task_type": "classification",
            "optimizer": "adamw",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "scheduler": "cosine",
            "epochs": 20,
            "batch_size": 32,
            "image_size": 96,
            "augmentation": {
                "policy": "basic",
                "mixup": 0.3,
                "cutmix": 0.1,
                "random_erasing": 0.2,
            },
            "loss": {"name": "focal_loss"},
        }
        payload["dataset_recipe"] = {
            "version": "dataset_recipe@v1",
            "task_type": "classification",
            "dataset_name": "neu",
            "class_names": ["crazing"],
            "source": {
                "root_dir": "data/raw/neu",
                "prepared_source_dir": "data/raw/neu/classification_source",
            },
            "splits": {
                "train_manifest": "data/classification/neu/train.txt",
                "val_manifest": "data/classification/neu/val.txt",
                "test_manifest": "data/classification/neu/test.txt",
            },
        }

        config = ExperimentConfig.model_validate(payload)

        self.assertEqual(config.model_recipe.width_multiple, 1.25)
        self.assertEqual(config.model_recipe.backbone_config.stem_variant, "deep_stem")
        self.assertEqual(config.model_recipe.head_config.pooling_type, "gem")
        self.assertEqual(config.model_recipe.components.neck.name, "gem_pool")
        self.assertEqual(len(config.model_recipe.backbone), 12)
        self.assertEqual(config.train_hyp.lr0, 0.001)
        self.assertEqual(config.train_hyp.loss.name, "focal_loss")
        self.assertEqual(config.dataset_recipe.class_names, ["crazing"])
        self.assertEqual(
            config.dataset_recipe.source.prepared_source_dir,
            "data/raw/neu/classification_source",
        )

    def test_result_params_are_derived_from_train_hyp(self) -> None:
        payload = _build_config_payload()
        payload["model_family"] = "googlenet"
        payload["model_name"] = "googlenet"
        payload["parameter_space_version"] = "googlenet@v1"
        payload["model_recipe"] = {
            "version": "model_recipe@v1",
            "task_type": "classification",
            "model_family": "googlenet",
            "base_model": "googlenet",
            "modules": {"aux_logits": True},
        }
        payload["train_hyp"] = {
            "version": "train_hyp@v1",
            "task_type": "classification",
            "optimizer": "sgd",
            "lr0": 0.01,
            "lrf": 0.05,
            "momentum": 0.95,
            "weight_decay": 0.0003,
            "scheduler": "step",
            "epochs": 30,
            "batch_size": 16,
            "image_size": 128,
            "label_smoothing": 0.05,
            "augmentation": {
                "policy": "basic",
                "mixup": 0.4,
                "cutmix": 0.2,
                "random_erasing": 0.1,
            },
            "loss": {"name": "focal_loss"},
            "fl_gamma": 1.5,
        }

        config = ExperimentConfig.model_validate(payload)
        result_params = config.result_params()

        self.assertTrue(config.use_aux_logits())
        self.assertEqual(result_params.optimizer, "sgd")
        self.assertEqual(result_params.learning_rate, 0.01)
        self.assertEqual(result_params.augmentation_params.mixup_alpha, 0.4)
        self.assertEqual(result_params.loss_name, "focal_loss")
        self.assertEqual(result_params.loss_params.focal_gamma, 1.5)
        self.assertTrue(result_params.aux_logits)

    def test_legacy_architecture_payload_is_migrated_to_flat_backbone_head_lists(self) -> None:
        payload = _build_config_payload()
        payload["model_recipe"] = {
            "version": "model_recipe@v1",
            "task_type": "classification",
            "model_family": "mobilenet",
            "base_model": "mobilenet_v3_small",
            "backbone": {
                "stem_variant": "standard",
                "attention_module": "none",
                "last_channel_multiplier": 1.0,
            },
            "head": {
                "pooling_type": "avg",
                "classifier_dropout": 0.2,
                "classifier_type": "linear",
            },
            "architecture": {
                "stem": {
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 2,
                    "activation_type": "hardswish",
                },
                "backbone": [
                    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [16, 3, 16, 16, True, "RE", 2, 1]}
                ],
                "head": [
                    {"from": -1, "repeat": 1, "module": "pointwise_tail", "args": [6, "hardswish"]},
                    {"from": -1, "repeat": 1, "module": "global_pool", "args": []},
                    {"from": -1, "repeat": 1, "module": "classifier", "args": [1024]},
                ],
            },
        }

        config = ExperimentConfig.model_validate(payload)

        self.assertEqual(config.model_recipe.backbone_config.stem_variant, "standard")
        self.assertEqual(config.model_recipe.backbone[0].module, "stem_conv")
        self.assertEqual(config.model_recipe.head[-1].module, "classifier")
        self.assertEqual(config.model_recipe.components.neck.name, "avg_pool")


if __name__ == "__main__":
    unittest.main()
