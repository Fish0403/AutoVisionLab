"""Unit tests for frontend-facing model defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.model_defaults_service import get_model_defaults


class ModelDefaultsServiceTest(unittest.TestCase):
    """Verify backend model defaults returned to the workspace."""

    def test_get_model_defaults_returns_search_ready_template(self) -> None:
        model_defaults = get_model_defaults("mobilenet_v2")

        self.assertIsNotNone(model_defaults)
        assert model_defaults is not None
        self.assertEqual(model_defaults.summary.model_name, "mobilenet_v2")
        self.assertEqual(model_defaults.parameter_space.version, "mobilenet_v2@v1")
        self.assertEqual(model_defaults.default_model_recipe.base_model, "mobilenet_v2")
        self.assertEqual(model_defaults.default_train_hyp.optimizer, "adamw")
        self.assertEqual(model_defaults.default_train_hyp.scheduler, "cosine")
        self.assertEqual(model_defaults.default_train_hyp.batch_size, 64)
        self.assertEqual(model_defaults.default_train_hyp.image_size, 224)
        self.assertEqual(
            model_defaults.default_search_policy.allowed_basic_hparam_fields,
            ["optimizer", "learning_rate", "batch_size", "weight_decay", "scheduler", "label_smoothing", "image_size"],
        )
        self.assertTrue(model_defaults.default_search_policy.allow_loss_search)
        self.assertTrue(model_defaults.default_search_policy.allow_augmentation_search)
        self.assertTrue(model_defaults.default_search_policy.allow_model_module_search)
        self.assertEqual(model_defaults.default_ranking_policy.tie_breaker_metric, "latency_ms")
        self.assertEqual(model_defaults.default_ranking_policy.min_tie_breaker_metric_improvement, 0.5)

    def test_get_model_defaults_preserves_aux_logits_defaults(self) -> None:
        model_defaults = get_model_defaults("googlenet")

        self.assertIsNotNone(model_defaults)
        assert model_defaults is not None
        self.assertEqual(model_defaults.default_model_recipe.modules.get("aux_logits"), False)
        self.assertTrue(model_defaults.default_search_policy.allow_model_module_search)

    def test_get_model_defaults_returns_none_for_unknown_model(self) -> None:
        self.assertIsNone(get_model_defaults("missing_model"))


if __name__ == "__main__":
    unittest.main()
