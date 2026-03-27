"""Tests for run promotion policy and proposal change budgets."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_PATH = REPO_ROOT / "test_run_promotion_policy.db"
TEST_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "test_run_promotion_policy"
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

os.environ["AVL_DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH}"
os.environ["AVL_ARTIFACT_ROOT"] = str(TEST_ARTIFACT_ROOT)

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.main import initialize_database
from app.db.session import SessionLocal
from app.schemas.ai import ResultSchema
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.parameter_space import ExperimentConfig, SearchPolicy
from app.schemas.run import RunCreateRequest
from app.services.parameter_space import get_parameter_space
from app.services.persistence import (
    create_experiment,
    create_run,
    get_run_detail,
    save_experiment_result,
    clear_all_records,
)
from app.services.run_policy import (
    determine_change_budget,
    get_available_dimensions,
    get_dimension_attempt_count,
    determine_forbidden_dimensions,
    determine_temporarily_blocked_fields,
    get_default_run_policy,
    proposal_switches_dimension,
    require_non_basic_change_for_elapsed_budget,
    should_stop_after_dimension_coverage,
)


def _build_experiment_config(parameter_space_version: str) -> ExperimentConfig:
    """Build a minimal ranking-enabled config."""
    return ExperimentConfig.model_validate(
        {
            "task_type": "classification",
            "dataset": "cifar10",
            "model_family": "mobilenet",
            "model_name": "mobilenet_v2",
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
    )


def _build_result(
    *,
    top1_acc: float,
    val_loss: float,
    experiment_config: ExperimentConfig,
    run_id: str,
    experiment_id: str,
) -> ResultSchema:
    """Build a synthetic successful result payload."""
    return ResultSchema.model_validate(
        {
            "status": "success",
            "metrics": {
                "train_loss": val_loss + 0.1,
                "val_loss": val_loss,
                "top1_acc": top1_acc,
                "best_epoch": 1,
            },
            "resource": {
                "gpu_memory_mb": 0,
                "training_seconds": 1,
            },
            "params": experiment_config.params.model_dump(),
            "artifacts": {
                "log_path": str(TEST_ARTIFACT_ROOT / "runs" / f"{run_id}.log"),
                "checkpoint_path": str(TEST_ARTIFACT_ROOT / "checkpoints" / f"{experiment_id}.pt"),
            },
        }
    )


class RunPromotionPolicyTest(unittest.TestCase):
    """Verify promotion/discard rules and change budgets."""

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

    def _create_run_with_two_results(
        self,
        *,
        baseline_top1_acc: float,
        baseline_val_loss: float,
        candidate_top1_acc: float,
        candidate_val_loss: float,
    ) -> tuple[dict[str, str], str]:
        """Create one run and two successful experiments for promotion checks."""
        parameter_space = get_parameter_space("mobilenet_v2")
        assert parameter_space is not None
        experiment_config = _build_experiment_config(parameter_space.version)

        with SessionLocal() as db:
            run = create_run(
                db,
                RunCreateRequest(
                    name=f"promotion-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v2",
                    base_config=experiment_config,
                    notes=None,
                ),
            )
            baseline = create_experiment(
                db,
                ExperimentCreateRequest(
                    run_id=run.id,
                    config=experiment_config,
                    parameter_space=parameter_space,
                    proposal=None,
                ),
            )
            assert baseline is not None
            save_experiment_result(
                db,
                baseline.id,
                _build_result(
                    top1_acc=baseline_top1_acc,
                    val_loss=baseline_val_loss,
                    experiment_config=experiment_config,
                    run_id=run.id,
                    experiment_id=baseline.id,
                ),
            )

            candidate = create_experiment(
                db,
                ExperimentCreateRequest(
                    run_id=run.id,
                    config=experiment_config,
                    parameter_space=parameter_space,
                    proposal=None,
                ),
            )
            assert candidate is not None
            candidate_detail = save_experiment_result(
                db,
                candidate.id,
                _build_result(
                    top1_acc=candidate_top1_acc,
                    val_loss=candidate_val_loss,
                    experiment_config=experiment_config,
                    run_id=run.id,
                    experiment_id=candidate.id,
                ),
            )
            assert candidate_detail is not None
            run_detail = get_run_detail(db, run.id)
            assert run_detail is not None
            return (
                {
                    "run_id": run.id,
                    "baseline_id": baseline.id,
                    "candidate_id": candidate.id,
                },
                run_detail.best_experiment_id or "",
            )

    def test_small_metric_gain_does_not_promote(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8010,
            candidate_val_loss=0.4950,
        )

        self.assertEqual(best_experiment_id, ids["baseline_id"])

        with SessionLocal() as db:
            candidate_detail = get_run_detail(db, ids["run_id"])
            assert candidate_detail is not None
            latest_experiment = candidate_detail.experiments[-1]
            self.assertEqual(latest_experiment.id, ids["candidate_id"])
            self.assertEqual(latest_experiment.decision, "discard")

    def test_significant_top1_acc_gain_promotes(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8100,
            candidate_val_loss=0.5050,
        )

        self.assertEqual(best_experiment_id, ids["candidate_id"])

    def test_val_loss_gain_at_top1_parity_promotes(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8002,
            candidate_val_loss=0.4850,
        )

        self.assertEqual(best_experiment_id, ids["candidate_id"])

    def test_change_budget_stays_single_variable_without_stagnation(self) -> None:
        stagnation_rounds, max_changed_fields = determine_change_budget(
            [
                {"status": "success", "decision": "keep"},
                {"status": "success", "decision": "discard"},
            ]
        )

        self.assertEqual(stagnation_rounds, 1)
        self.assertEqual(max_changed_fields, 1)

    def test_change_budget_relaxes_after_two_stagnant_rounds(self) -> None:
        stagnation_rounds, max_changed_fields = determine_change_budget(
            [
                {"status": "success", "decision": "keep"},
                {"status": "success", "decision": "discard"},
                {"status": "failed", "decision": "crash"},
            ]
        )

        self.assertEqual(stagnation_rounds, 2)
        self.assertEqual(max_changed_fields, 2)

    def test_default_run_policy_exposes_current_hard_constraints(self) -> None:
        run_policy = get_default_run_policy()

        self.assertEqual(run_policy.min_top1_acc_promotion_delta, 0.01)
        self.assertEqual(run_policy.min_val_loss_promotion_delta, 0.01)
        self.assertEqual(run_policy.default_max_changed_fields, 1)
        self.assertEqual(run_policy.max_changed_fields_after_stagnation, 2)

    def test_non_basic_change_phase_uses_budget_ratio(self) -> None:
        run_policy = get_default_run_policy()

        self.assertFalse(require_non_basic_change_for_elapsed_budget(300, 10, policy=run_policy))
        self.assertTrue(require_non_basic_change_for_elapsed_budget(301, 10, policy=run_policy))

    def test_repeated_failed_field_enters_temporary_cooldown(self) -> None:
        run_policy = get_default_run_policy()

        blocked_fields = determine_temporarily_blocked_fields(
            [
                {"status": "success", "decision": "keep"},
                {
                    "status": "success",
                    "decision": "discard",
                    "proposal": {"changes": {"label_smoothing": 0.05}},
                },
                {
                    "status": "failed",
                    "decision": "crash",
                    "proposal": {"changes": {"label_smoothing": 0.10}},
                },
            ],
            policy=run_policy,
        )

        self.assertEqual(blocked_fields, {"label_smoothing"})

    def test_dimension_switch_required_after_three_stagnant_rounds_in_same_dimension(self) -> None:
        run_policy = get_default_run_policy()

        forbidden_dimensions = determine_forbidden_dimensions(
            [
                {"status": "success", "decision": "keep"},
                {
                    "status": "success",
                    "decision": "discard",
                    "proposal": {"changes": {"label_smoothing": 0.05}},
                },
                {
                    "status": "success",
                    "decision": "discard",
                    "proposal": {"changes": {"weight_decay": 0.0005}},
                },
                {
                    "status": "failed",
                    "decision": "crash",
                    "proposal": {"changes": {"scheduler": "step"}},
                },
            ],
            policy=run_policy,
        )

        self.assertEqual(forbidden_dimensions, {"basic"})
        self.assertFalse(
            proposal_switches_dimension(
                {"learning_rate"},
                forbidden_dimensions=forbidden_dimensions,
            )
        )
        self.assertTrue(
            proposal_switches_dimension(
                {"mixup_alpha"},
                forbidden_dimensions=forbidden_dimensions,
            )
        )

    def test_available_dimensions_follow_search_policy(self) -> None:
        search_policy = SearchPolicy.model_validate(
            {
                "allow_basic_hparam_search": True,
                "allowed_basic_hparam_fields": ["learning_rate", "batch_size"],
                "allow_strategy_search": False,
                "allow_loss_search": True,
                "allow_augmentation_search": True,
                "require_manual_approval_for_high_impact_changes": True,
            }
        )

        self.assertEqual(
            get_available_dimensions(search_policy),
            {"basic", "loss", "augmentation"},
        )

    def test_dimension_coverage_stop_requires_exploration_budget_and_stagnation(self) -> None:
        search_policy = SearchPolicy.model_validate(
            {
                "allow_basic_hparam_search": True,
                "allowed_basic_hparam_fields": ["learning_rate", "batch_size"],
                "allow_strategy_search": False,
                "allow_loss_search": False,
                "allow_augmentation_search": True,
                "require_manual_approval_for_high_impact_changes": True,
            }
        )
        experiment_history = [
            {
                "id": "exp_keep",
                "status": "success",
                "decision": "keep",
                "params": {
                    "learning_rate": 0.003,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": None,
            },
            {
                "id": "exp_basic_1",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"learning_rate": 0.001}},
            },
            {
                "id": "exp_aug_1",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.2,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.2}},
            },
            {
                "id": "exp_basic_2",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.002,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"learning_rate": 0.002}},
            },
            {
                "id": "exp_aug_2",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.3,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.3}},
            },
            {
                "id": "exp_basic_3",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.004,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"learning_rate": 0.004}},
            },
            {
                "id": "exp_aug_3",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "batch_size": 32,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.4,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.4}},
            },
        ]

        self.assertEqual(
            get_dimension_attempt_count(experiment_history),
            {"basic": 3, "augmentation": 3, "loss": 0, "strategy": 0},
        )
        should_stop, _ = should_stop_after_dimension_coverage(experiment_history, search_policy)
        self.assertTrue(should_stop)

    def test_dimension_coverage_stop_waits_for_minimum_attempt_budget(self) -> None:
        search_policy = SearchPolicy.model_validate(
            {
                "allow_basic_hparam_search": True,
                "allowed_basic_hparam_fields": ["learning_rate"],
                "allow_strategy_search": False,
                "allow_loss_search": False,
                "allow_augmentation_search": True,
                "require_manual_approval_for_high_impact_changes": True,
            }
        )
        experiment_history = [
            {
                "id": "exp_keep",
                "status": "success",
                "decision": "keep",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": None,
            },
            {
                "id": "exp_basic_1",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.001,
                    "augmentation_policy": "basic",
                    "augmentation_params": {
                        "mixup_alpha": 0.0,
                        "cutmix_alpha": 0.0,
                        "random_erasing_prob": 0.0,
                    },
                    "loss_params": {"focal_gamma": 2.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"learning_rate": 0.001}},
            },
            {
                "id": "exp_aug_1",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.2, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.2}},
            },
            {
                "id": "exp_aug_2",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.3, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.3}},
            },
            {
                "id": "exp_aug_3",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.4, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.4}},
            },
            {
                "id": "exp_aug_4",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.5, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.5}},
            },
            {
                "id": "exp_aug_5",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.6, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.6}},
            },
            {
                "id": "exp_aug_6",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.003,
                    "augmentation_policy": "basic",
                    "augmentation_params": {"mixup_alpha": 0.7, "cutmix_alpha": 0.0, "random_erasing_prob": 0.0},
                },
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.7}},
            },
        ]

        should_stop, stop_reason = should_stop_after_dimension_coverage(experiment_history, search_policy)
        self.assertFalse(should_stop)
        self.assertIn("minimum successful attempt budget", stop_reason)


if __name__ == "__main__":
    unittest.main()
