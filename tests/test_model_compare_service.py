"""Unit tests for cross-model compare service helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import (
    ExperimentConfig,
    ExperimentParams,
    build_default_model_recipe,
)
from app.services.model_compare_service import (
    MODEL_COMPARE_TASKS,
    _build_compare_config,
    _build_compare_summary_prompt,
    _try_attach_compare_ai_summary,
    delete_model_compare_task,
    list_model_compare_tasks,
    start_model_compare_task,
)
from app.schemas.run import ModelCompareCandidateResult, ModelCompareStartRequest, ModelCompareSummary
from tests.helpers.experiment_config_builders import build_default_dataset_recipe, build_train_hyp_from_params


def _build_base_config(dataset_name: str | None = None) -> ExperimentConfig:
    """Build one minimal config payload for compare tests."""
    dataset_name = dataset_name or f"dataset_{uuid4().hex}"
    params = ExperimentParams.model_validate(
        {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 64,
            "image_size": 96,
            "epochs": 10,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "augmentation_params": {
                "mixup_alpha": 0.1,
                "cutmix_alpha": 0.0,
                "random_erasing_prob": 0.1,
            },
            "loss_name": "cross_entropy_with_label_smoothing",
            "loss_params": {"focal_gamma": 2.0},
            "label_smoothing": 0.1,
            "aux_logits": True,
        }
    )
    return ExperimentConfig.model_validate(
        {
            "task_type": "classification",
            "dataset": dataset_name,
            "model_family": "mobilenet",
            "model_name": "mobilenet_v3_small",
            "parameter_space_version": "mobilenet_v3_small@v1",
            "use_demo_mode": True,
            "search_policy": {
                "allow_basic_hparam_search": True,
                "allowed_basic_hparam_fields": ["learning_rate"],
                "allow_strategy_search": True,
                "allow_loss_search": True,
                "allow_augmentation_search": True,
                "allow_model_module_search": True,
                "require_manual_approval_for_high_impact_changes": True,
            },
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
                dataset_name=dataset_name,
                task_type="classification",
            ).model_dump(),
        }
    )


class ModelCompareServiceTest(unittest.TestCase):
    """Verify compare-task helper behavior."""

    def setUp(self) -> None:
        MODEL_COMPARE_TASKS.clear()

    def test_build_compare_config_for_googlenet_forces_aux_logits_off(self) -> None:
        compare_config, compare_notes = _build_compare_config(
            base_config=_build_base_config(),
            model_name="googlenet",
        )

        self.assertEqual(compare_config.model_name, "googlenet")
        self.assertFalse(compare_config.params.aux_logits)
        self.assertFalse(compare_config.search_policy.allow_basic_hparam_search)
        self.assertFalse(compare_config.search_policy.allow_strategy_search)
        self.assertEqual(compare_notes, [])

    def test_build_compare_config_for_mobilenet_v3_small_disables_component_search(self) -> None:
        compare_config, compare_notes = _build_compare_config(
            base_config=_build_base_config(),
            model_name="mobilenet_v3_small",
        )

        self.assertEqual(compare_config.model_name, "mobilenet_v3_small")
        self.assertFalse(compare_config.search_policy.allow_model_module_search)
        self.assertEqual(compare_config.model_recipe.components.neck.name, "avg_pool")
        self.assertIn("Disabled component search", compare_notes[0])

    def test_start_model_compare_task_returns_queued_snapshot(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        request = ModelCompareStartRequest(
            dataset=dataset_name,
            config=_build_base_config(dataset_name),
            candidate_models=["mobilenet_v2", "googlenet"],
        )

        with (
            patch("app.services.model_compare_service.get_active_auto_train_task", return_value=None),
            patch("app.services.model_compare_service.MODEL_COMPARE_EXECUTOR.submit"),
            patch("app.services.model_compare_service.upsert_task_payload"),
        ):
            task = start_model_compare_task(request)

        self.assertEqual(task.status, "queued")
        self.assertEqual(task.total_models, 2)
        self.assertEqual(task.summary.mode, "model_compare")
        self.assertEqual(task.summary.shared_baseline_config["dataset"], dataset_name)

    def test_start_model_compare_task_uses_lightweight_dataset_probe(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        request = ModelCompareStartRequest(
            dataset=dataset_name,
            config=_build_base_config(dataset_name),
            candidate_models=["mobilenet_v2"],
        )

        with (
            patch("app.services.model_compare_service.get_active_auto_train_task", return_value=None),
            patch("app.services.model_compare_service.get_lightweight_local_dataset_summary") as get_probe,
            patch("app.services.model_compare_service.build_lightweight_dataset_summary_text", return_value="Source directory found."),
            patch("app.services.model_compare_service.get_local_dataset_summary") as get_full_summary,
            patch("app.services.model_compare_service.MODEL_COMPARE_EXECUTOR.submit"),
            patch("app.services.model_compare_service.upsert_task_payload"),
        ):
            get_probe.return_value = object()
            task = start_model_compare_task(request)

        get_probe.assert_called_once_with(dataset_name)
        get_full_summary.assert_not_called()
        self.assertEqual(task.dataset_summary, "Source directory found.")

    def test_build_compare_summary_prompt_contains_candidate_metrics(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        summary = ModelCompareSummary(
            shared_baseline_config={"dataset": dataset_name},
            candidate_results=[
                ModelCompareCandidateResult(
                    model_name="mobilenet_v2",
                    status="success",
                    top1_acc=0.85,
                    latency_ms=2.1,
                )
            ],
        )

        system_prompt, user_prompt = _build_compare_summary_prompt(summary)

        self.assertIn("summary_text", system_prompt)
        self.assertIn("\"top1_acc\": 0.85", user_prompt)
        self.assertIn("\"latency_ms\": 2.1", user_prompt)

    def test_try_attach_compare_ai_summary_updates_summary(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        summary = ModelCompareSummary(
            shared_baseline_config={"dataset": dataset_name},
            candidate_results=[
                ModelCompareCandidateResult(
                    model_name="mobilenet_v2",
                    status="success",
                    top1_acc=0.85,
                    latency_ms=2.1,
                )
            ],
        )

        with patch(
            "app.services.model_compare_service._generate_compare_ai_summary",
            return_value="MobileNetV2 leads with 85.0% Top1 at 2.1 ms latency.",
        ):
            updated_summary = _try_attach_compare_ai_summary("cmp_test", summary)

        self.assertEqual(updated_summary.ai_summary, "MobileNetV2 leads with 85.0% Top1 at 2.1 ms latency.")
        self.assertIsNone(updated_summary.ai_summary_error)

    def test_try_attach_compare_ai_summary_records_error_when_provider_fails(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        summary = ModelCompareSummary(
            shared_baseline_config={"dataset": dataset_name},
            candidate_results=[
                ModelCompareCandidateResult(
                    model_name="mobilenet_v2",
                    status="success",
                    top1_acc=0.85,
                    latency_ms=2.1,
                )
            ],
        )

        with (
            patch(
                "app.services.model_compare_service._generate_compare_ai_summary",
                side_effect=RuntimeError("provider quota exceeded"),
            ),
            patch("app.services.model_compare_service._append_task_log"),
        ):
            updated_summary = _try_attach_compare_ai_summary("cmp_test", summary)

        self.assertIsNone(updated_summary.ai_summary)
        self.assertEqual(updated_summary.ai_summary_error, "provider quota exceeded")

    def test_list_model_compare_tasks_prefers_ai_summary(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        with patch(
            "app.services.model_compare_service.list_task_payloads",
            return_value=[
                {
                    "task_id": "cmp_demo_001",
                    "title": f"Compare {dataset_name}",
                    "status": "success",
                    "dataset": dataset_name,
                    "candidate_models": ["mobilenet_v2", "googlenet"],
                    "current_model_index": 2,
                    "total_models": 2,
                    "created_at": "2026-03-31T00:00:00+00:00",
                    "updated_at": "2026-03-31T00:01:00+00:00",
                    "summary": {
                        "mode": "model_compare",
                        "shared_baseline_config": {"dataset": dataset_name},
                        "candidate_results": [],
                        "ai_summary": "GoogLeNet leads with 91.8% Top1 at 3.23 ms latency.",
                    },
                }
            ],
        ):
            history_items = list_model_compare_tasks()

        self.assertEqual(len(history_items), 1)
        self.assertEqual(history_items[0].summary, "GoogLeNet leads with 91.8% Top1 at 3.23 ms latency.")

    def test_delete_model_compare_task_cascades_linked_search_tasks(self) -> None:
        MODEL_COMPARE_TASKS["cmp_1"] = {
            "task_id": "cmp_1",
            "status": "stopped",
            "logs": [],
            "owned_run_ids": ["run_cmp_1", "run_cmp_2"],
            "summary": {"shared_baseline_config": {}, "candidate_results": []},
        }

        with (
            patch(
                "app.services.model_compare_service.get_task_payload",
                return_value={
                    "task_id": "cmp_1",
                    "status": "stopped",
                    "logs": [],
                    "owned_run_ids": ["run_cmp_1", "run_cmp_2"],
                    "summary": {"shared_baseline_config": {}, "candidate_results": []},
                },
            ),
            patch(
                "app.services.model_compare_service.list_task_payloads",
                return_value=[
                    {
                        "task_id": "auto_1",
                        "status": "stopped",
                        "source_task_type": "model_compare",
                        "source_task_id": "cmp_1",
                    }
                ],
            ),
            patch(
                "app.services.model_compare_service.delete_auto_train_task",
                return_value={
                    "deleted_tasks": 1,
                    "deleted_runs": 0,
                    "deleted_experiments": 0,
                    "deleted_results": 0,
                    "deleted_artifact_files": 0,
                },
            ),
            patch(
                "app.services.model_compare_service.clear_run_records",
                side_effect=[
                    {"deleted_runs": 1, "deleted_experiments": 2, "deleted_results": 3, "deleted_artifact_files": 4},
                    {"deleted_runs": 1, "deleted_experiments": 1, "deleted_results": 1, "deleted_artifact_files": 1},
                ],
            ),
            patch("app.services.model_compare_service.delete_task_payload", return_value=True),
        ):
            deleted_counts = delete_model_compare_task("cmp_1")

        self.assertEqual(
            deleted_counts,
            {
                "deleted_tasks": 1,
                "deleted_search_tasks": 1,
                "deleted_runs": 2,
                "deleted_experiments": 3,
                "deleted_results": 4,
                "deleted_artifact_files": 5,
            },
        )
        self.assertNotIn("cmp_1", MODEL_COMPARE_TASKS)


if __name__ == "__main__":
    unittest.main()
