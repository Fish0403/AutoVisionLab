"""Tests for MobileNetV3 Small recipe-based model building."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import ModelRecipe
from app.trainers.builder import build_model_from_recipe


class MobileNetRecipeBuilderTest(unittest.TestCase):
    """Verify the MobileNetV3 Small recipe builder."""

    def test_builder_applies_width_multiple_and_dropout(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "width_multiple": 0.75,
                "head_config": {
                    "pooling_type": "avg",
                    "classifier_dropout": 0.5,
                    "classifier_type": "linear",
                },
            }
        )

        model = build_model_from_recipe(recipe, num_classes=7)

        self.assertEqual(model.classifier[2].p, 0.5)
        self.assertEqual(model.classifier[3].out_features, 7)
        self.assertEqual(model.classifier[0].in_features, 432)
        self.assertEqual(len(model.features), 13)

    def test_builder_rejects_unsupported_attention_module(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "backbone_config": {
                    "stem_variant": "standard",
                    "attention_module": "se",
                    "last_channel_multiplier": 1.0,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "Unsupported attention_module"):
            build_model_from_recipe(recipe)

    def test_builder_supports_gem_pooling(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "components": {
                    "backbone": {"name": "mobilenet_v3_small_native", "params": {}},
                    "neck": {"name": "gem_pool", "params": {}},
                    "head": {"name": "native_classifier", "params": {}},
                },
                "head_config": {
                    "pooling_type": "gem",
                    "classifier_dropout": 0.2,
                    "classifier_type": "linear",
                },
            }
        )

        model = build_model_from_recipe(recipe)

        self.assertEqual(model.avgpool.__class__.__name__, "GeMPooling2d")

    def test_builder_prefers_component_level_neck_over_legacy_pooling_field(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "components": {
                    "backbone": {"name": "mobilenet_v3_small_native", "params": {}},
                    "neck": {"name": "gem_pool", "params": {}},
                    "head": {"name": "native_classifier", "params": {}},
                },
                "head_config": {
                    "pooling_type": "avg",
                    "classifier_dropout": 0.2,
                    "classifier_type": "linear",
                },
            }
        )

        model = build_model_from_recipe(recipe)

        self.assertEqual(model.avgpool.__class__.__name__, "GeMPooling2d")

    def test_builder_supports_linear_head_component(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "components": {
                    "backbone": {"name": "mobilenet_v3_small_native", "params": {}},
                    "neck": {"name": "avg_pool", "params": {}},
                    "head": {"name": "linear", "params": {}},
                },
            }
        )

        model = build_model_from_recipe(recipe, num_classes=4)
        output = model(torch.randn(2, 3, 96, 96))

        self.assertEqual(model.classifier.__class__.__name__, "Linear")
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_builder_supports_dropout_linear_head_component(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
                "components": {
                    "backbone": {"name": "mobilenet_v3_small_native", "params": {}},
                    "neck": {"name": "avg_pool", "params": {}},
                    "head": {"name": "dropout_linear", "params": {}},
                },
            }
        )

        model = build_model_from_recipe(recipe, num_classes=6)
        output = model(torch.randn(2, 3, 96, 96))

        self.assertEqual(model.classifier[0].p, 0.2)
        self.assertEqual(model.classifier[1].out_features, 6)
        self.assertEqual(tuple(output.shape), (2, 6))

    def test_builder_hydrates_builtin_architecture_template(self) -> None:
        recipe = ModelRecipe.model_validate(
            {
                "task_type": "classification",
                "model_family": "mobilenet",
                "base_model": "mobilenet_v3_small",
            }
        )

        model = build_model_from_recipe(recipe, num_classes=5)
        output = model(torch.randn(2, 3, 96, 96))

        self.assertEqual(len(model.features), 13)
        self.assertEqual(model.classifier[0].out_features, 1024)
        self.assertEqual(tuple(output.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
