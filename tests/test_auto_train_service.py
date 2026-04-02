"""Unit tests for auto-train service helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import requests

from app.schemas.parameter_space import ExperimentConfig
from app.llm.aihubmix_client import AIHubMixRequestError
from app.services.auto_train_service import (
    AUTO_TRAIN_TASKS,
    _ensure_no_active_task,
    _build_followup_config,
    _is_retryable_proposal_error,
    _load_auto_train_seed_experiment,
    _load_latest_search_policy_for_run,
    _normalize_auto_train_history_summary,
    delete_auto_train_task,
    stop_auto_train_task,
)


class AutoTrainServiceTest(unittest.TestCase):
    """Verify auto-train helper behavior around persisted config objects."""

    def setUp(self) -> None:
        AUTO_TRAIN_TASKS.clear()

    def tearDown(self) -> None:
        AUTO_TRAIN_TASKS.clear()

    def test_load_latest_search_policy_reads_experiment_config_object(self) -> None:
        experiment_config = ExperimentConfig.model_validate(
            {
                "task_type": "classification",
                "dataset": "cifar10",
                "model_family": "mobilenet",
                "model_name": "mobilenet_v3_small",
                "parameter_space_version": "test-v1",
                "search_policy": {
                    "allow_basic_hparam_search": True,
                    "allowed_basic_hparam_fields": ["learning_rate"],
                    "allow_strategy_search": False,
                    "allow_loss_search": True,
                    "allow_augmentation_search": False,
                    "allow_model_module_search": True,
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
        )

        with (
            patch(
                "app.services.auto_train_service.get_run_detail",
                return_value=SimpleNamespace(experiments=[SimpleNamespace(id="exp_1")]),
            ),
            patch(
                "app.services.auto_train_service.get_experiment_detail",
                return_value=SimpleNamespace(config=experiment_config),
            ),
        ):
            search_policy = _load_latest_search_policy_for_run(db=SimpleNamespace(), run_id="run_1")

        self.assertTrue(search_policy.allow_basic_hparam_search)
        self.assertEqual(search_policy.allowed_basic_hparam_fields, ["learning_rate"])
        self.assertTrue(search_policy.allow_loss_search)
        self.assertFalse(search_policy.allow_augmentation_search)
        self.assertTrue(search_policy.allow_model_module_search)

    def test_build_followup_config_preserves_search_and_ranking_policies(self) -> None:
        followup_config = _build_followup_config(
            latest_experiment={
                "config": {
                    "task_type": "classification",
                    "dataset": "cifar10",
                    "model_family": "mobilenet",
                    "model_name": "mobilenet_v3_small",
                    "parameter_space_version": "test-v1",
                    "use_demo_mode": True,
                    "participates_in_ranking": True,
                    "search_policy": {
                        "allow_basic_hparam_search": True,
                        "allowed_basic_hparam_fields": ["learning_rate"],
                        "allow_strategy_search": False,
                        "allow_loss_search": True,
                        "allow_augmentation_search": False,
                        "allow_model_module_search": True,
                        "require_manual_approval_for_high_impact_changes": True,
                    },
                    "ranking_policy": {
                        "primary_metric": "top1_acc",
                        "primary_metric_mode": "max",
                        "min_primary_metric_improvement": 0.005,
                        "primary_metric_parity_epsilon": 0.0005,
                        "tie_breaker_metric": "val_loss",
                        "tie_breaker_mode": "min",
                        "min_tie_breaker_metric_improvement": 0.01,
                        "max_image_size": 64,
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
            },
            proposal_payload={"changes": {"learning_rate": 0.001}},
        )

        self.assertIn("search_policy", followup_config)
        self.assertIn("ranking_policy", followup_config)
        self.assertIn("train_hyp", followup_config)
        self.assertIn("model_recipe", followup_config)
        self.assertTrue(followup_config["use_demo_mode"])
        self.assertEqual(followup_config["ranking_policy"]["max_image_size"], 64)
        self.assertEqual(followup_config["params"]["learning_rate"], 0.001)
        self.assertEqual(followup_config["train_hyp"]["lr0"], 0.001)
        self.assertTrue(followup_config["search_policy"]["allow_model_module_search"])

    def test_build_followup_config_maps_aux_logits_to_model_recipe(self) -> None:
        followup_config = _build_followup_config(
            latest_experiment={
                "config": {
                    "task_type": "classification",
                    "dataset": "cifar10",
                    "model_family": "googlenet",
                    "model_name": "googlenet",
                    "parameter_space_version": "test-v1",
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
            },
            proposal_payload={"changes": {"aux_logits": True}},
        )

        self.assertTrue(followup_config["model_recipe"]["modules"]["aux_logits"])
        self.assertTrue(followup_config["params"]["aux_logits"])

    def test_build_followup_config_prefers_structured_change_views(self) -> None:
        followup_config = _build_followup_config(
            latest_experiment={
                "config": {
                    "task_type": "classification",
                    "dataset": "cifar10",
                    "model_family": "googlenet",
                    "model_name": "googlenet",
                    "parameter_space_version": "test-v1",
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
            },
            proposal_payload={
                "changes": {
                    "learning_rate": 0.009,
                    "mixup_alpha": 0.1,
                    "aux_logits": False,
                },
                "train_hyp_changes": {
                    "lr0": 0.001,
                    "augmentation": {"mixup": 0.4},
                },
                "recipe_changes": {
                    "modules": {"aux_logits": True},
                },
            },
        )

        self.assertEqual(followup_config["train_hyp"]["lr0"], 0.001)
        self.assertEqual(followup_config["train_hyp"]["augmentation"]["mixup"], 0.4)
        self.assertTrue(followup_config["model_recipe"]["modules"]["aux_logits"])
        self.assertEqual(followup_config["params"]["learning_rate"], 0.001)
        self.assertEqual(followup_config["params"]["augmentation_params"]["mixup_alpha"], 0.4)
        self.assertTrue(followup_config["params"]["aux_logits"])

    def test_load_auto_train_seed_experiment_reuses_successful_experiment(self) -> None:
        with patch(
            "app.services.auto_train_service._resolve_followup_source_experiment",
            return_value={"id": "exp_seed", "status": "success"},
        ):
            experiment_detail = _load_auto_train_seed_experiment(SimpleNamespace(), "run_1")

        self.assertEqual(experiment_detail["id"], "exp_seed")

    def test_load_auto_train_seed_experiment_keeps_non_success_seed(self) -> None:
        with patch(
            "app.services.auto_train_service._resolve_followup_source_experiment",
            return_value={"id": "exp_failed", "status": "failed"},
        ):
            experiment_detail = _load_auto_train_seed_experiment(SimpleNamespace(), "run_1")

        self.assertEqual(experiment_detail["id"], "exp_failed")
        self.assertEqual(experiment_detail["status"], "failed")

    def test_is_retryable_proposal_error_accepts_read_timeout_request_exception(self) -> None:
        self.assertTrue(_is_retryable_proposal_error(requests.ReadTimeout("read timed out")))

    def test_is_retryable_proposal_error_accepts_wrapped_timeout_message(self) -> None:
        wrapped_timeout_error = AIHubMixRequestError(
            "chat completions request failed after retries: "
            "HTTPSConnectionPool(host='aihubmix.com', port=443): Read timed out. (read timeout=60)"
        )

        self.assertTrue(_is_retryable_proposal_error(wrapped_timeout_error))

    def test_normalize_auto_train_history_summary_prefers_task_error(self) -> None:
        summary = _normalize_auto_train_history_summary(
            {
                "status": "failed",
                "error": "Auto train stopped after repeated invalid proposals.",
                "summary": {
                    "rounds": [
                        {
                            "result": {
                                "summary": "Older round summary",
                            }
                        }
                    ]
                },
            }
        )

        self.assertEqual(summary, "Auto train stopped after repeated invalid proposals.")

    def test_normalize_auto_train_history_summary_keeps_stop_reason_before_error(self) -> None:
        summary = _normalize_auto_train_history_summary(
            {
                "status": "stopped",
                "stop_reason": "Stopped by user request.",
                "error": "Should not win",
                "summary": {},
            }
        )

        self.assertEqual(summary, "Stopped by user request.")

    def test_stop_auto_train_task_finishes_queued_task_immediately(self) -> None:
        AUTO_TRAIN_TASKS["task_1"] = {
            "task_id": "task_1",
            "title": "Queued task",
            "status": "queued",
            "dataset": "cifar10",
            "model_name": "mobilenet_v3_small",
            "policy_preset": None,
            "run_id": None,
            "source_task_type": None,
            "source_task_id": None,
            "source_task_title": None,
            "source_model_name": None,
            "current_round": 0,
            "elapsed_seconds": 0.0,
            "current_experiment_id": None,
            "created_at": None,
            "updated_at": None,
            "logs": [],
            "summary": None,
            "error": None,
            "stop_requested": False,
            "stop_reason": None,
            "latest_prompt_tokens_estimate": None,
            "estimated_prompt_tokens_total": 0,
            "latest_prompt_history_items": None,
            "latest_provider_prompt_tokens": None,
            "latest_provider_completion_tokens": None,
            "latest_provider_total_tokens": None,
            "provider_prompt_tokens_total": 0,
            "provider_completion_tokens_total": 0,
            "provider_total_tokens_total": 0,
        }

        with patch("app.services.auto_train_service.upsert_task_payload"):
            stopped_task = stop_auto_train_task("task_1")

        self.assertIsNotNone(stopped_task)
        self.assertEqual(stopped_task.status, "stopped")
        self.assertEqual(stopped_task.stop_reason, "Stopped by user request.")

    def test_ensure_no_active_task_mentions_stopping_task(self) -> None:
        AUTO_TRAIN_TASKS["task_1"] = {
            "status": "stopping",
        }

        with self.assertRaisesRegex(ValueError, "still stopping"):
            _ensure_no_active_task()

    def test_delete_auto_train_task_removes_owned_runs(self) -> None:
        AUTO_TRAIN_TASKS["task_1"] = {
            "task_id": "task_1",
            "status": "stopped",
            "logs": [],
            "owned_run_ids": ["run_a"],
        }

        with (
            patch(
                "app.services.auto_train_service.get_task_payload",
                return_value={"task_id": "task_1", "status": "stopped", "logs": [], "owned_run_ids": ["run_a"]},
            ),
            patch("app.services.auto_train_service.clear_run_records", return_value={"deleted_runs": 1, "deleted_experiments": 2, "deleted_results": 3, "deleted_artifact_files": 4}),
            patch("app.services.auto_train_service.delete_task_payload", return_value=True),
        ):
            deleted_counts = delete_auto_train_task("task_1")

        self.assertEqual(
            deleted_counts,
            {
                "deleted_tasks": 1,
                "deleted_runs": 1,
                "deleted_experiments": 2,
                "deleted_results": 3,
                "deleted_artifact_files": 4,
            },
        )
        self.assertNotIn("task_1", AUTO_TRAIN_TASKS)


if __name__ == "__main__":
    unittest.main()
