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

from app.schemas.parameter_space import ExperimentConfig
from app.services.auto_train_service import (
    _build_followup_config,
    _load_auto_train_seed_experiment,
    _load_latest_search_policy_for_run,
    _try_attach_final_proposal,
)


class AutoTrainServiceTest(unittest.TestCase):
    """Verify auto-train helper behavior around persisted config objects."""

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

    def test_load_auto_train_seed_experiment_rejects_non_success_status(self) -> None:
        with patch(
            "app.services.auto_train_service._resolve_followup_source_experiment",
            return_value={"id": "exp_failed", "status": "failed"},
        ):
            with self.assertRaisesRegex(ValueError, "requires one successful experiment"):
                _load_auto_train_seed_experiment(SimpleNamespace(), "run_1")

    def test_try_attach_final_proposal_updates_summary(self) -> None:
        summary = {"mode": "auto", "run_id": "run_1", "baseline": {"experiment_id": "exp_1"}, "rounds": []}
        with (
            patch("app.services.auto_train_service.SessionLocal") as session_local,
            patch(
                "app.services.auto_train_service.generate_aihubmix_proposal",
                return_value=SimpleNamespace(model_dump=lambda: {"hypothesis": "next", "changes": {"learning_rate": 0.001}}),
            ),
            patch("app.services.auto_train_service._update_task") as update_task,
            patch("app.services.auto_train_service._append_task_log"),
        ):
            session_local.return_value = SimpleNamespace(close=lambda: None)
            _try_attach_final_proposal(
                "task_1",
                run_id="run_1",
                round_index=4,
                summary=summary,
            )

        update_task.assert_called_once()
        updated_summary = update_task.call_args.kwargs["summary"]
        self.assertEqual(updated_summary["final_proposal"]["hypothesis"], "next")

    def test_try_attach_final_proposal_records_error_when_generation_fails(self) -> None:
        summary = {"mode": "auto", "run_id": "run_1", "baseline": {"experiment_id": "exp_1"}, "rounds": []}
        with (
            patch("app.services.auto_train_service.SessionLocal") as session_local,
            patch(
                "app.services.auto_train_service.generate_aihubmix_proposal",
                side_effect=ValueError("proposal failed"),
            ),
            patch("app.services.auto_train_service._update_task") as update_task,
            patch("app.services.auto_train_service._append_task_log"),
        ):
            session_local.return_value = SimpleNamespace(close=lambda: None)
            _try_attach_final_proposal(
                "task_1",
                run_id="run_1",
                round_index=1,
                summary=summary,
            )

        update_task.assert_called_once()
        updated_summary = update_task.call_args.kwargs["summary"]
        self.assertIsNone(updated_summary["final_proposal"])
        self.assertEqual(updated_summary["final_suggestion_error"], "proposal failed")


if __name__ == "__main__":
    unittest.main()
