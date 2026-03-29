"""Smoke tests for the shared API response envelope."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_PATH = REPO_ROOT / "test_api_response_envelope.db"
TEST_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "test_api_response_envelope"
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

os.environ["AVL_DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH}"
os.environ["AVL_ARTIFACT_ROOT"] = str(TEST_ARTIFACT_ROOT)

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.api.routes.experiments import (
    create_experiment_endpoint,
    get_experiment,
    save_decision_endpoint,
    save_result_endpoint,
)
from app.api.routes.models import read_parameter_space
from app.api.routes.runs import create_run_endpoint
from app.db.session import SessionLocal
from app.main import healthcheck, initialize_database
from app.schemas.ai import ResultSchema
from app.schemas.api import ApiResponse
from app.schemas.experiment import ExperimentCreateRequest, ExperimentDecisionRequest
from app.schemas.parameter_space import EditableParameterSpace, ExperimentConfig
from app.schemas.run import RunCreateRequest
from app.services.persistence import clear_all_records


def _build_experiment_config(parameter_space_version: str) -> dict[str, object]:
    """Build a minimal valid experiment config payload."""
    return {
        "task_type": "classification",
        "dataset": "cifar10",
        "model_family": "mobilenet",
        "model_name": "mobilenet_v3_small",
        "parameter_space_version": parameter_space_version,
        "participates_in_ranking": True,
        "search_policy": {
            "allow_basic_hparam_search": True,
            "allowed_basic_hparam_fields": [
                "optimizer",
                "learning_rate",
                "batch_size",
                "weight_decay",
                "scheduler",
                "label_smoothing",
            ],
            "allow_strategy_search": False,
            "allow_loss_search": False,
            "allow_augmentation_search": False,
            "require_manual_approval_for_high_impact_changes": True,
        },
        "params": {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 32,
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
            "loss_params": {
                "focal_gamma": 2.0,
            },
            "label_smoothing": 0.1,
            "aux_logits": False,
        },
    }


class ApiResponseEnvelopeTest(unittest.TestCase):
    """Verify major APIs return the shared response envelope."""

    @classmethod
    def setUpClass(cls) -> None:
        initialize_database()

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_DATABASE_PATH.exists():
            TEST_DATABASE_PATH.unlink()
        if TEST_ARTIFACT_ROOT.exists():
            shutil.rmtree(TEST_ARTIFACT_ROOT)

    def setUp(self) -> None:
        with SessionLocal() as db:
            clear_all_records(db)

    def test_health_endpoint_uses_api_response_envelope(self) -> None:
        response = healthcheck()

        self.assertIsInstance(response, ApiResponse)
        payload = response.model_dump()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["code"], "success")
        self.assertEqual(payload["message"], "Service is healthy.")
        self.assertEqual(payload["data"], {"status": "ok"})
        self.assertEqual(payload["errors"], [])
        self.assertIn("meta", payload)

    def test_model_parameter_space_uses_api_response_envelope(self) -> None:
        response = read_parameter_space("mobilenet_v3_small")

        self.assertIsInstance(response, ApiResponse)
        payload = response.model_dump()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "Parameter space loaded.")
        self.assertEqual(payload["data"]["model_name"], "mobilenet_v3_small")
        self.assertIn("editable_params", payload["data"])

    def test_experiment_endpoints_use_api_response_envelope(self) -> None:
        parameter_space_response = read_parameter_space("mobilenet_v3_small")
        parameter_space = parameter_space_response.data.model_dump()
        experiment_config = _build_experiment_config(parameter_space["version"])

        with SessionLocal() as db:
            run_response = create_run_endpoint(
                request=RunCreateRequest(
                    name=f"api-envelope-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    base_config=ExperimentConfig.model_validate(experiment_config),
                    notes="Envelope smoke test.",
                ),
                db=db,
            )
            self.assertIsInstance(run_response, ApiResponse)
            self.assertTrue(run_response.ok)
            run_id = run_response.data.id

            create_response = create_experiment_endpoint(
                request=ExperimentCreateRequest(
                    run_id=run_id,
                    config=ExperimentConfig.model_validate(experiment_config),
                    parameter_space=EditableParameterSpace.model_validate(parameter_space),
                    proposal=None,
                ),
                db=db,
            )
            self.assertIsInstance(create_response, ApiResponse)
            self.assertTrue(create_response.ok)
            self.assertEqual(create_response.code, "created")
            self.assertEqual(create_response.message, "Experiment created.")
            experiment_id = create_response.data.id

            detail_response = get_experiment(experiment_id=experiment_id, db=db)
            self.assertIsInstance(detail_response, ApiResponse)
            self.assertTrue(detail_response.ok)
            self.assertEqual(detail_response.message, "Experiment detail loaded.")
            self.assertEqual(detail_response.data.id, experiment_id)

            result_response = save_result_endpoint(
                experiment_id=experiment_id,
                request=ResultSchema.model_validate({
                    "status": "success",
                    "metrics": {
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "top1_acc": 0.9,
                        "best_epoch": 1,
                    },
                    "resource": {
                        "gpu_memory_mb": 0,
                        "training_seconds": 1,
                    },
                    "params": experiment_config["params"],
                    "artifacts": {
                        "log_path": str(TEST_ARTIFACT_ROOT / "runs" / f"{run_id}.log"),
                        "checkpoint_path": str(TEST_ARTIFACT_ROOT / "checkpoints" / f"{experiment_id}.pt"),
                    },
                }),
                db=db,
            )
            self.assertIsInstance(result_response, ApiResponse)
            self.assertTrue(result_response.ok)
            self.assertEqual(result_response.code, "updated")
            self.assertEqual(result_response.message, "Experiment result saved.")
            self.assertEqual(result_response.data.result.status, "success")

            decision_response = save_decision_endpoint(
                experiment_id=experiment_id,
                request=ExperimentDecisionRequest(
                    decision="keep",
                    decision_reason="Smoke test decision.",
                ),
                db=db,
            )
            self.assertIsInstance(decision_response, ApiResponse)
            self.assertTrue(decision_response.ok)
            self.assertEqual(decision_response.code, "updated")
            self.assertEqual(decision_response.message, "Experiment decision saved.")
            self.assertEqual(decision_response.data.decision, "keep")

    def test_create_experiment_uses_server_parameter_space_snapshot(self) -> None:
        stale_parameter_space = {
            "model_name": "mobilenet_v3_small",
            "version": "mobilenet_v3_small@v1",
            "editable_params": {
                "optimizer": {
                    "type": "enum",
                    "choices": ["sgd", "adam", "adamw"],
                }
            },
        }
        experiment_config = _build_experiment_config("mobilenet_v3_small@v1")

        with SessionLocal() as db:
            run_response = create_run_endpoint(
                request=RunCreateRequest(
                    name=f"api-envelope-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    base_config=ExperimentConfig.model_validate(experiment_config),
                    notes="Server parameter-space snapshot test.",
                ),
                db=db,
            )

            create_response = create_experiment_endpoint(
                request=ExperimentCreateRequest(
                    run_id=run_response.data.id,
                    config=ExperimentConfig.model_validate(experiment_config),
                    parameter_space=EditableParameterSpace.model_validate(stale_parameter_space),
                    proposal=None,
                ),
                db=db,
            )

        self.assertIn("neck_name", create_response.data.parameter_space.editable_params)
        self.assertIn("head_name", create_response.data.parameter_space.editable_params)


if __name__ == "__main__":
    unittest.main()
