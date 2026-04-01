"""Tests for task history persistence across backend restarts."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_PATH = REPO_ROOT / "test_task_history_persistence.db"
TEST_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "test_task_history_persistence"
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

os.environ["AVL_DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH}"
os.environ["AVL_ARTIFACT_ROOT"] = str(TEST_ARTIFACT_ROOT)

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.api.routes.models import read_parameter_space
from app.api.routes.runs import (
    get_auto_train_endpoint,
    get_model_compare_endpoint,
    get_task_history,
    start_auto_train_endpoint,
    start_model_compare_endpoint,
)
from app.db.session import SessionLocal
from app.main import initialize_database
from app.schemas.parameter_space import ExperimentConfig
from app.schemas.run import AutoTrainStartRequest, ModelCompareStartRequest
from app.services.auto_train_service import AUTO_TRAIN_TASKS
from app.services.model_compare_service import MODEL_COMPARE_TASKS
from app.services.persistence import clear_all_records
from app.services.task_store import cleanup_stale_task_payloads


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
            "loss_params": {"focal_gamma": 2.0},
            "label_smoothing": 0.1,
            "aux_logits": False,
        },
    }


class TaskHistoryPersistenceTest(unittest.TestCase):
    """Verify workspace task history survives process restarts."""

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
        AUTO_TRAIN_TASKS.clear()
        MODEL_COMPARE_TASKS.clear()
        with SessionLocal() as db:
            clear_all_records(db)

    def test_auto_train_task_history_survives_memory_reset(self) -> None:
        parameter_space_response = read_parameter_space("mobilenet_v3_small")
        parameter_space = parameter_space_response.data
        config = ExperimentConfig.model_validate(_build_experiment_config(parameter_space.version))

        with patch("app.services.auto_train_service.AUTO_TRAIN_EXECUTOR.submit", return_value=None):
            start_response = start_auto_train_endpoint(
                AutoTrainStartRequest(
                    title="Persisted search task",
                    run_id=None,
                    run_name="persisted-search-task",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    policy_preset=None,
                    source_task_type="model_compare",
                    source_task_id="cmp_seed",
                    source_task_title="Seed compare",
                    source_model_name="mobilenet_v2",
                    config=config,
                    parameter_space=parameter_space,
                )
            )

        task_id = start_response.data.task_id
        AUTO_TRAIN_TASKS.clear()

        history_response = get_task_history()
        history_task_ids = [task.task_id for task in history_response.data]
        self.assertIn(task_id, history_task_ids)

        task_response = get_auto_train_endpoint(task_id)
        self.assertEqual(task_response.data.task_id, task_id)
        self.assertEqual(task_response.data.title, "Persisted search task")
        self.assertEqual(task_response.data.status, "queued")
        self.assertEqual(task_response.data.source_task_type, "model_compare")
        self.assertEqual(task_response.data.source_task_id, "cmp_seed")
        self.assertEqual(task_response.data.source_model_name, "mobilenet_v2")

        persisted_history_item = next(task for task in history_response.data if task.task_id == task_id)
        self.assertEqual(persisted_history_item.source_task_type, "model_compare")
        self.assertEqual(persisted_history_item.source_task_id, "cmp_seed")
        self.assertEqual(persisted_history_item.source_model_name, "mobilenet_v2")

    def test_restart_cleanup_marks_stale_task_stopped(self) -> None:
        parameter_space_response = read_parameter_space("mobilenet_v3_small")
        parameter_space = parameter_space_response.data
        config = ExperimentConfig.model_validate(_build_experiment_config(parameter_space.version))

        with patch("app.services.auto_train_service.AUTO_TRAIN_EXECUTOR.submit", return_value=None):
            start_response = start_auto_train_endpoint(
                AutoTrainStartRequest(
                    title="Interrupted search task",
                    run_id=None,
                    run_name="interrupted-search-task",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    policy_preset=None,
                    config=config,
                    parameter_space=parameter_space,
                )
            )

        task_id = start_response.data.task_id
        AUTO_TRAIN_TASKS.clear()
        cleanup_stale_task_payloads()

        task_response = get_auto_train_endpoint(task_id)
        self.assertEqual(task_response.data.status, "stopped")
        self.assertEqual(task_response.data.stop_reason, "Task interrupted by backend restart.")

    def test_model_compare_task_history_survives_memory_reset(self) -> None:
        parameter_space_response = read_parameter_space("mobilenet_v3_small")
        parameter_space = parameter_space_response.data
        config = ExperimentConfig.model_validate(_build_experiment_config(parameter_space.version))

        with patch("app.services.model_compare_service.MODEL_COMPARE_EXECUTOR.submit", return_value=None):
            start_response = start_model_compare_endpoint(
                ModelCompareStartRequest(
                    title="Persisted compare task",
                    dataset="cifar10",
                    candidate_models=["mobilenet_v2", "mobilenet_v3_small"],
                    config=config,
                )
            )

        task_id = start_response.data.task_id
        MODEL_COMPARE_TASKS.clear()

        history_response = get_task_history()
        history_task_ids = [task.task_id for task in history_response.data]
        self.assertIn(task_id, history_task_ids)

        task_response = get_model_compare_endpoint(task_id)
        self.assertEqual(task_response.data.task_id, task_id)
        self.assertEqual(task_response.data.title, "Persisted compare task")
        self.assertEqual(task_response.data.status, "queued")
