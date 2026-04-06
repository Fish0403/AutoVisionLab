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
from app.schemas.parameter_space import (
    ExperimentConfig,
    ExperimentParams,
    build_default_model_recipe,
)
from app.schemas.ranking_policy import RankingPolicy
from app.schemas.run import RunCreateRequest
from app.services.parameter_space import get_parameter_space
from app.services.persistence import (
    create_experiment,
    create_run,
    get_run_detail,
    get_run_summary,
    get_run_trend,
    save_experiment_result,
    clear_all_records,
    clear_run_records,
)
from app.services.run_policy import (
    determine_change_budget,
    determine_forbidden_dimensions,
    determine_temporarily_blocked_fields,
    get_default_ranking_policy,
    get_default_run_policy,
    get_preferred_fields,
    proposal_switches_dimension,
    require_non_basic_change_after_warmup_rounds,
)
from tests.helpers.experiment_config_builders import build_default_dataset_recipe, build_train_hyp_from_params


def _build_experiment_config(
    parameter_space_version: str,
    *,
    ranking_policy: RankingPolicy | None = None,
) -> ExperimentConfig:
    """Build a minimal ranking-enabled config."""
    effective_ranking_policy = ranking_policy or RankingPolicy()
    params = ExperimentParams.model_validate(
        {
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
        }
    )
    return ExperimentConfig.model_validate(
        {
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
            "ranking_policy": effective_ranking_policy.model_dump(),
            "params": params.model_dump(),
            "model_recipe": build_default_model_recipe(
                model_name="mobilenet_v3_small",
                task_type="classification",
                model_family="mobilenet",
            ).model_dump(by_alias=True),
            "train_hyp": build_train_hyp_from_params(
                task_type="classification",
                params=params,
            ).model_dump(),
            "dataset_recipe": build_default_dataset_recipe(
                dataset_name="cifar10",
                task_type="classification",
            ).model_dump(),
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
        ranking_policy: RankingPolicy | None = None,
        candidate_image_size: int = 32,
    ) -> tuple[dict[str, str], str]:
        """Create one run and two successful experiments for promotion checks."""
        parameter_space = get_parameter_space("mobilenet_v3_small")
        assert parameter_space is not None
        experiment_config = _build_experiment_config(
            parameter_space.version,
            ranking_policy=ranking_policy,
        )
        candidate_config = ExperimentConfig.model_validate(
            {
                **experiment_config.model_dump(),
                "train_hyp": {
                    **experiment_config.train_hyp.model_dump(),
                    "image_size": candidate_image_size,
                },
                "params": {
                    **experiment_config.params.model_dump(),
                    "image_size": candidate_image_size,
                },
            }
        )

        with SessionLocal() as db:
            run = create_run(
                db,
                RunCreateRequest(
                    name=f"promotion-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
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
                    config=candidate_config,
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
                    experiment_config=candidate_config,
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
            candidate_top1_acc=0.8009,
            candidate_val_loss=0.4950,
        )

        self.assertEqual(best_experiment_id, ids["baseline_id"])

    def test_dimension_attempt_count_reads_train_hyp_history(self) -> None:
        experiment_history = [
            {
                "id": "exp_keep",
                "status": "success",
                "decision": "keep",
                "train_hyp": {
                    "optimizer": "adamw",
                    "lr0": 0.003,
                    "batch_size": 32,
                    "image_size": 32,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "label_smoothing": 0.1,
                    "augmentation": {"policy": "basic", "mixup": 0.0, "cutmix": 0.0, "random_erasing": 0.0},
                    "loss": {"name": "cross_entropy_with_label_smoothing"},
                    "fl_gamma": 0.0,
                },
                "model_recipe": {"modules": {}},
            },
            {
                "id": "exp_aug",
                "status": "success",
                "decision": "discard",
                "train_hyp": {
                    "optimizer": "adamw",
                    "lr0": 0.003,
                    "batch_size": 32,
                    "image_size": 32,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "label_smoothing": 0.1,
                    "augmentation": {"policy": "basic", "mixup": 0.3, "cutmix": 0.0, "random_erasing": 0.0},
                    "loss": {"name": "cross_entropy_with_label_smoothing"},
                    "fl_gamma": 0.0,
                },
                "model_recipe": {"modules": {}},
                "proposal": {"based_on_experiment_ids": ["exp_keep"], "changes": {"mixup_alpha": 0.3}},
            },
        ]

    def test_significant_top1_acc_gain_promotes(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8100,
            candidate_val_loss=0.5050,
        )

        self.assertEqual(best_experiment_id, ids["candidate_id"])

    def test_run_trend_returns_all_metric_series_in_one_snapshot(self) -> None:
        parameter_space = get_parameter_space("mobilenet_v3_small")
        assert parameter_space is not None
        experiment_config = _build_experiment_config(parameter_space.version)

        with SessionLocal() as db:
            run = create_run(
                db,
                RunCreateRequest(
                    name=f"trend-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    base_config=experiment_config,
                    notes=None,
                ),
            )
            for top1_acc, val_loss, train_loss, best_epoch in [
                (0.8000, 0.5000, 0.6000, 1),
                (0.8110, 0.4700, 0.5600, 2),
            ]:
                experiment = create_experiment(
                    db,
                    ExperimentCreateRequest(
                        run_id=run.id,
                        config=experiment_config,
                        parameter_space=parameter_space,
                        proposal=None,
                    ),
                )
                assert experiment is not None
                result_payload = _build_result(
                    top1_acc=top1_acc,
                    val_loss=val_loss,
                    experiment_config=experiment_config,
                    run_id=run.id,
                    experiment_id=experiment.id,
                ).model_dump()
                result_payload["metrics"]["train_loss"] = train_loss
                result_payload["metrics"]["best_epoch"] = best_epoch
                result_payload["resource"]["latency_ms"] = 7.5 - best_epoch
                save_experiment_result(db, experiment.id, ResultSchema.model_validate(result_payload))

            trend = get_run_trend(db, run.id)
            assert trend is not None

        self.assertEqual(trend.run_id, run.id)
        self.assertIn("top1_acc", trend.available_metrics)
        self.assertIn("latency_ms", trend.available_metrics)
        series_by_metric = {item.metric_name: item for item in trend.series}
        self.assertEqual(len(series_by_metric["top1_acc"].points), 2)
        self.assertEqual(len(series_by_metric["val_loss"].points), 2)
        self.assertEqual(len(series_by_metric["train_loss"].points), 2)
        self.assertEqual(len(series_by_metric["best_epoch"].points), 2)
        self.assertEqual(len(series_by_metric["latency_ms"].points), 2)
        self.assertEqual(series_by_metric["top1_acc"].points[1].experiment_index, 2)
        self.assertEqual(series_by_metric["latency_ms"].points[1].metric_value, 5.5)

    def test_val_loss_gain_at_top1_parity_promotes(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8002,
            candidate_val_loss=0.4850,
        )

        self.assertEqual(best_experiment_id, ids["candidate_id"])

    def test_latency_gain_at_top1_parity_promotes(self) -> None:
        parameter_space = get_parameter_space("mobilenet_v3_small")
        assert parameter_space is not None
        ranking_policy = RankingPolicy(
            primary_metric="top1_acc",
            primary_metric_mode="max",
            min_primary_metric_improvement=0.01,
            primary_metric_parity_epsilon=0.0005,
            tie_breaker_metric="latency_ms",
            tie_breaker_mode="min",
            min_tie_breaker_metric_improvement=1.0,
        )
        experiment_config = _build_experiment_config(
            parameter_space.version,
            ranking_policy=ranking_policy,
        )

        with SessionLocal() as db:
            run = create_run(
                db,
                RunCreateRequest(
                    name=f"latency-promotion-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
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
            baseline_result = _build_result(
                top1_acc=0.8000,
                val_loss=0.5000,
                experiment_config=experiment_config,
                run_id=run.id,
                experiment_id=baseline.id,
            ).model_dump()
            baseline_result["resource"]["latency_ms"] = 8.4
            save_experiment_result(db, baseline.id, ResultSchema.model_validate(baseline_result))

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
            candidate_result = _build_result(
                top1_acc=0.8003,
                val_loss=0.5010,
                experiment_config=experiment_config,
                run_id=run.id,
                experiment_id=candidate.id,
            ).model_dump()
            candidate_result["resource"]["latency_ms"] = 6.9
            save_experiment_result(db, candidate.id, ResultSchema.model_validate(candidate_result))

            run_detail = get_run_detail(db, run.id)
            assert run_detail is not None

        self.assertEqual(run_detail.best_experiment_id, candidate.id)

    def test_run_summary_exposes_efficiency_and_tradeoff_anchors(self) -> None:
        parameter_space = get_parameter_space("mobilenet_v3_small")
        assert parameter_space is not None
        ranking_policy = RankingPolicy(
            primary_metric="top1_acc",
            primary_metric_mode="max",
            min_primary_metric_improvement=0.005,
            primary_metric_parity_epsilon=0.0005,
            tie_breaker_metric="latency_ms",
            tie_breaker_mode="min",
            min_tie_breaker_metric_improvement=0.5,
        )
        experiment_config = _build_experiment_config(
            parameter_space.version,
            ranking_policy=ranking_policy,
        )

        with SessionLocal() as db:
            run = create_run(
                db,
                RunCreateRequest(
                    name=f"anchor-test-{uuid4().hex[:8]}",
                    dataset="cifar10",
                    model_name="mobilenet_v3_small",
                    base_config=experiment_config,
                    notes=None,
                ),
            )

            def create_success_experiment(
                *,
                top1_acc: float,
                val_loss: float,
                latency_ms: float,
                parameter_count_million: float,
            ) -> str:
                experiment = create_experiment(
                    db,
                    ExperimentCreateRequest(
                        run_id=run.id,
                        config=experiment_config,
                        parameter_space=parameter_space,
                        proposal=None,
                    ),
                )
                assert experiment is not None
                result_payload = _build_result(
                    top1_acc=top1_acc,
                    val_loss=val_loss,
                    experiment_config=experiment_config,
                    run_id=run.id,
                    experiment_id=experiment.id,
                ).model_dump()
                result_payload["resource"]["latency_ms"] = latency_ms
                result_payload["resource"]["parameter_count_million"] = parameter_count_million
                save_experiment_result(db, experiment.id, ResultSchema.model_validate(result_payload))
                return experiment.id

            baseline_id = create_success_experiment(
                top1_acc=0.8000,
                val_loss=0.5000,
                latency_ms=8.0,
                parameter_count_million=2.0,
            )
            fast_id = create_success_experiment(
                top1_acc=0.7998,
                val_loss=0.5050,
                latency_ms=5.2,
                parameter_count_million=1.6,
            )
            accurate_id = create_success_experiment(
                top1_acc=0.8100,
                val_loss=0.4950,
                latency_ms=9.3,
                parameter_count_million=2.8,
            )
            tradeoff_id = create_success_experiment(
                top1_acc=0.8097,
                val_loss=0.4940,
                latency_ms=6.1,
                parameter_count_million=2.1,
            )

            run_detail = get_run_detail(db, run.id)
            run_summary = get_run_summary(db, run.id)
            assert run_detail is not None
            assert run_summary is not None

        self.assertEqual(run_detail.best_experiment_id, tradeoff_id)
        self.assertEqual(run_summary.best_experiment_id, tradeoff_id)

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

    def test_default_policies_expose_current_constraints(self) -> None:
        run_policy = get_default_run_policy()
        ranking_policy = get_default_ranking_policy()

        self.assertEqual(run_policy.default_max_changed_fields, 1)
        self.assertEqual(run_policy.max_changed_fields_after_stagnation, 2)
        self.assertEqual(ranking_policy.primary_metric, "top1_acc")
        self.assertEqual(ranking_policy.tie_breaker_metric, "val_loss")
        self.assertEqual(ranking_policy.min_primary_metric_improvement, 0.001)
        self.assertEqual(ranking_policy.min_tie_breaker_metric_improvement, 0.01)

    def test_max_image_size_gate_blocks_promotion(self) -> None:
        ids, best_experiment_id = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8200,
            candidate_val_loss=0.4700,
            ranking_policy=RankingPolicy(max_image_size=64),
            candidate_image_size=96,
        )

        self.assertEqual(best_experiment_id, ids["baseline_id"])

    def test_non_basic_change_phase_uses_warmup_rounds(self) -> None:
        run_policy = get_default_run_policy()

        self.assertFalse(require_non_basic_change_after_warmup_rounds(3, policy=run_policy))
        self.assertTrue(require_non_basic_change_after_warmup_rounds(4, policy=run_policy))

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

    def test_clear_run_records_removes_run_artifacts(self) -> None:
        ids, _ = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8100,
            candidate_val_loss=0.4900,
        )
        run_log_path = TEST_ARTIFACT_ROOT / "runs" / f"{ids['run_id']}.log"
        baseline_checkpoint_path = TEST_ARTIFACT_ROOT / "checkpoints" / f"{ids['baseline_id']}.pt"
        candidate_checkpoint_path = TEST_ARTIFACT_ROOT / "checkpoints" / f"{ids['candidate_id']}.pt"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text("run-log", encoding="utf-8")
        baseline_checkpoint_path.write_text("baseline", encoding="utf-8")
        candidate_checkpoint_path.write_text("candidate", encoding="utf-8")

        with SessionLocal() as db:
            deleted_counts = clear_run_records(db, ids["run_id"])

        self.assertIsNotNone(deleted_counts)
        self.assertEqual(deleted_counts["deleted_runs"], 1)
        self.assertGreaterEqual(deleted_counts["deleted_artifact_files"], 3)
        self.assertFalse(run_log_path.exists())
        self.assertFalse(baseline_checkpoint_path.exists())
        self.assertFalse(candidate_checkpoint_path.exists())

    def test_clear_all_records_removes_all_artifacts(self) -> None:
        ids, _ = self._create_run_with_two_results(
            baseline_top1_acc=0.8000,
            baseline_val_loss=0.5000,
            candidate_top1_acc=0.8100,
            candidate_val_loss=0.4900,
        )
        run_log_path = TEST_ARTIFACT_ROOT / "runs" / f"{ids['run_id']}.log"
        baseline_checkpoint_path = TEST_ARTIFACT_ROOT / "checkpoints" / f"{ids['baseline_id']}.pt"
        candidate_checkpoint_path = TEST_ARTIFACT_ROOT / "checkpoints" / f"{ids['candidate_id']}.pt"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text("run-log", encoding="utf-8")
        baseline_checkpoint_path.write_text("baseline", encoding="utf-8")
        candidate_checkpoint_path.write_text("candidate", encoding="utf-8")

        with SessionLocal() as db:
            deleted_counts = clear_all_records(db)

        self.assertGreaterEqual(deleted_counts["deleted_artifact_files"], 3)
        self.assertFalse(run_log_path.exists())
        self.assertFalse(baseline_checkpoint_path.exists())
        self.assertFalse(candidate_checkpoint_path.exists())

    def test_preferred_fields_fall_back_when_soft_preferences_would_empty_space(self) -> None:
        preferred_fields, preference_notes = get_preferred_fields(
            {"learning_rate"},
            blocked_fields={"learning_rate"},
            discouraged_dimensions={"basic"},
            prefer_non_basic=True,
        )

        self.assertEqual(preferred_fields, {"learning_rate"})
        self.assertEqual(preference_notes, [])


if __name__ == "__main__":
    unittest.main()
