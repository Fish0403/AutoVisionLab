"""Unit tests for AI-assisted model manifest drafting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.model_catalog.registry import get_model_catalog_entry, reload_model_catalog
from app.schemas.model_manifest_draft import ModelManifestCommitRequest, ModelManifestDraftRequest
from app.services.model_manifest_draft_service import commit_model_manifest, draft_model_manifest


def _build_unique_mobilenet_v2_yaml(*, model_name: str = "mobilenet_v2_draft") -> str:
    """Return one unique MobileNetV2-based manifest YAML for testing."""
    source_path = REPO_ROOT / "backend" / "app" / "model_manifests" / "classification" / "mobilenet_v2.yaml"
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    payload["model_name"] = model_name
    payload["label"] = "MobileNetV2 Draft"
    payload["default_model_recipe"]["base_model"] = model_name
    payload["parameter_space"]["model_name"] = model_name
    payload["parameter_space"]["version"] = f"{model_name}@v1"
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120)


class ModelManifestDraftServiceTest(unittest.TestCase):
    """Verify manifest drafting and commit helpers."""

    def tearDown(self) -> None:
        reload_model_catalog()

    def test_draft_model_manifest_returns_valid_preview_for_unique_mobilenet_v2(self) -> None:
        with patch("app.services.model_manifest_draft_service.AIHubMixClient") as client_cls:
            client_cls.return_value.create_json_completion_with_metadata.return_value = (
                {
                    "resolved_model_name": "mobilenet_v2_draft",
                    "draft_spec": {
                        "model_name": "mobilenet_v2_draft",
                        "label": "MobileNetV2 Draft",
                        "model_family": "mobilenet",
                        "builder": {
                            "type": "torchvision_classifier",
                            "torchvision_name": "mobilenet_v2",
                            "backbone_component_name": "mobilenet_v2_native",
                            "backbone_extractor": "feature_sequence",
                            "native_head_attr": "classifier",
                            "feature_dim": 1280,
                            "default_dropout": 0.2,
                            "uses_dropout_arg": True,
                            "supported_neck_names": ["avg_pool", "gem_pool"],
                            "supported_head_names": ["native_classifier", "linear", "dropout_linear"],
                            "default_neck_name": "avg_pool",
                            "default_head_name": "native_classifier",
                        },
                        "recipe_profile": "torchvision_classifier_v1",
                        "parameter_space_profile": "classification_standard_v1",
                    },
                    "warnings": ["Resolved from a fuzzy user query."],
                },
                {},
            )

            response = draft_model_manifest(ModelManifestDraftRequest(query="mobilnet v2"))

        self.assertEqual(response.query, "mobilnet v2")
        self.assertEqual(response.resolved_model_name, "mobilenet_v2_draft")
        self.assertIn("backend/app/model_manifests/classification/mobilenet_v2_draft.yaml", response.target_path)
        self.assertTrue(response.validation.is_valid)
        self.assertTrue(response.validation.yaml_parse_ok)
        self.assertTrue(response.validation.schema_ok)
        self.assertTrue(response.validation.dry_run_build_ok)
        self.assertTrue(response.validation.dry_run_forward_ok)
        self.assertEqual(response.provider_warnings, ["Resolved from a fuzzy user query."])
        self.assertIn("parameter_space_profile: classification_standard_v1", response.ai_preview_text)
        self.assertNotIn("parameter_space:", response.ai_preview_text)
        self.assertIn("parameter_space:", response.yaml_text)
        self.assertIn("default_model_recipe:", response.yaml_text)

    def test_commit_model_manifest_writes_file_and_reloads_catalog(self) -> None:
        draft_yaml_text = _build_unique_mobilenet_v2_yaml(model_name="mobilenet_v2_commit_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_root = Path(temp_dir)
            with patch("app.model_catalog.loader.MODEL_MANIFEST_ROOT", manifest_root):
                response = commit_model_manifest(
                    ModelManifestCommitRequest(
                        yaml_text=draft_yaml_text,
                        expected_model_name="mobilenet_v2_commit_test",
                    )
                )
                reloaded_entry = get_model_catalog_entry("mobilenet_v2_commit_test")

                self.assertEqual(response.model_name, "mobilenet_v2_commit_test")
                self.assertEqual(response.reloaded_model_count, 1)
                self.assertIsNotNone(reloaded_entry)
                self.assertEqual(reloaded_entry.model_name, "mobilenet_v2_commit_test")
                self.assertTrue((manifest_root / "classification" / "mobilenet_v2_commit_test.yaml").exists())


if __name__ == "__main__":
    unittest.main()
