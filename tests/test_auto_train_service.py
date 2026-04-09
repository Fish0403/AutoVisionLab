"""Unit tests for auto-train service helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import requests

from app.schemas.parameter_space import (
    ExperimentConfig,
    ExperimentParams,
    build_default_model_recipe,
)
from app.schemas.run import AutoTrainStartRequest
from app.llm.aihubmix_client import AIHubMixRequestError
from app.services.auto_train_service import (
    AUTO_TRAIN_TASKS,
    _build_auto_train_summary_prompt,
    _build_search_scope_summary,
    _ensure_no_active_task,
    _build_followup_config,
    _generate_auto_train_proposal,
    _is_retryable_proposal_error,
    _load_auto_train_seed_experiment,
    _normalize_auto_train_history_summary,
    _run_auto_train_task,
    _try_attach_auto_train_ai_summary,
    _validate_followup_proposal,
    delete_auto_train_task,
    start_auto_train_task,
    stop_auto_train_task,
)
from app.services.parameter_space import get_parameter_space
from tests.helpers.experiment_config_builders import build_default_dataset_recipe, build_train_hyp_from_params


def _build_auto_train_config_payload(
    *,
    dataset_name: str | None = None,
    model_family: str = "mobilenet",
    model_name: str = "mobilenet_v3_small",
    parameter_space_version: str = "mobilenet_v3_small@v1",
    use_demo_mode: bool = True,
) -> dict[str, object]:
    """Build one minimal structured auto-train config payload for service tests."""
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
    return {
        "task_type": "classification",
        "dataset": dataset_name,
        "model_family": model_family,
        "model_name": model_name,
        "parameter_space_version": parameter_space_version,
        "use_demo_mode": use_demo_mode,
        "search_policy": {
            "allow_basic_hparam_search": True,
            "allowed_basic_hparam_fields": ["learning_rate", "image_size"],
            "allow_strategy_search": False,
            "allow_loss_search": True,
            "allow_augmentation_search": True,
            "allow_model_module_search": True,
            "require_manual_approval_for_high_impact_changes": True,
        },
        "params": params.model_dump(),
        "model_recipe": build_default_model_recipe(
            model_name=model_name,
            task_type="classification",
            model_family=model_family,
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


def _build_auto_train_config(dataset_name: str | None = None) -> ExperimentConfig:
    """Build one minimal auto-train config payload for service tests."""
    return ExperimentConfig.model_validate(_build_auto_train_config_payload(dataset_name=dataset_name))


class AutoTrainServiceTest(unittest.TestCase):
    """Verify auto-train helper behavior around persisted config objects."""

    def setUp(self) -> None:
        AUTO_TRAIN_TASKS.clear()

    def tearDown(self) -> None:
        AUTO_TRAIN_TASKS.clear()

    def test_build_followup_config_preserves_search_and_ranking_policies(self) -> None:
        latest_config = _build_auto_train_config_payload(dataset_name="cifar10")
        latest_config["parameter_space_version"] = "test-v1"
        latest_config["participates_in_ranking"] = True
        latest_config["search_policy"] = {
            "allow_basic_hparam_search": True,
            "allowed_basic_hparam_fields": ["learning_rate"],
            "allow_strategy_search": False,
            "allow_loss_search": True,
            "allow_augmentation_search": False,
            "allow_model_module_search": True,
            "require_manual_approval_for_high_impact_changes": True,
        }
        latest_config["ranking_policy"] = {
            "primary_metric": "top1_acc",
            "primary_metric_mode": "max",
            "min_primary_metric_improvement": 0.005,
            "primary_metric_parity_epsilon": 0.0005,
            "tie_breaker_metric": "val_loss",
            "tie_breaker_mode": "min",
            "min_tie_breaker_metric_improvement": 0.01,
            "max_image_size": 64,
        }
        latest_config["train_hyp"]["batch_size"] = 32
        latest_config["train_hyp"]["image_size"] = 32
        latest_config["train_hyp"]["epochs"] = 1
        followup_config = _build_followup_config(
            latest_experiment={"config": ExperimentConfig.model_validate(latest_config).model_dump(mode="python")},
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

    def test_start_auto_train_task_uses_lightweight_dataset_probe(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        request = AutoTrainStartRequest(
            run_name=f"Run {dataset_name}",
            dataset=dataset_name,
            model_name="mobilenet_v3_small",
            config=_build_auto_train_config(dataset_name),
            parameter_space=get_parameter_space("mobilenet_v3_small"),
        )

        with (
            patch("app.services.auto_train_service.get_lightweight_local_dataset_summary") as get_probe,
            patch("app.services.auto_train_service.build_lightweight_dataset_summary_text", return_value="Source directory found."),
            patch("app.services.auto_train_service.get_local_dataset_summary") as get_full_summary,
            patch("app.services.auto_train_service.AUTO_TRAIN_EXECUTOR.submit"),
            patch("app.services.auto_train_service.upsert_task_payload"),
        ):
            get_probe.return_value = SimpleNamespace(has_source_dir=True, message="Source directory found.")
            task = start_auto_train_task(request)

        get_probe.assert_called_once_with(dataset_name)
        get_full_summary.assert_not_called()
        self.assertEqual(task.dataset_summary, "Source directory found.")

    def test_build_followup_config_maps_aux_logits_to_model_recipe(self) -> None:
        latest_config = _build_auto_train_config_payload(
            dataset_name="cifar10",
            model_family="googlenet",
            model_name="googlenet",
            parameter_space_version="test-v1",
        )
        latest_config["train_hyp"]["batch_size"] = 32
        latest_config["train_hyp"]["image_size"] = 32
        latest_config["train_hyp"]["epochs"] = 1
        latest_config["model_recipe"]["modules"] = {"aux_logits": False}
        followup_config = _build_followup_config(
            latest_experiment={"config": ExperimentConfig.model_validate(latest_config).model_dump(mode="python")},
            proposal_payload={"changes": {"aux_logits": True}},
        )

        self.assertTrue(followup_config["model_recipe"]["modules"]["aux_logits"])
        self.assertTrue(followup_config["params"]["aux_logits"])

    def test_build_followup_config_prefers_flat_changes_for_ai_proposals(self) -> None:
        latest_config = _build_auto_train_config_payload(
            dataset_name="cifar10",
            model_family="googlenet",
            model_name="googlenet",
            parameter_space_version="test-v1",
        )
        latest_config["train_hyp"]["batch_size"] = 32
        latest_config["train_hyp"]["image_size"] = 32
        latest_config["train_hyp"]["epochs"] = 1
        latest_config["model_recipe"]["modules"] = {"aux_logits": False}
        followup_config = _build_followup_config(
            latest_experiment={"config": ExperimentConfig.model_validate(latest_config).model_dump(mode="python")},
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

        self.assertEqual(followup_config["train_hyp"]["lr0"], 0.009)
        self.assertEqual(followup_config["train_hyp"]["augmentation"]["mixup"], 0.1)
        self.assertFalse(followup_config["model_recipe"]["modules"]["aux_logits"])
        self.assertEqual(followup_config["params"]["learning_rate"], 0.009)
        self.assertEqual(followup_config["params"]["augmentation_params"]["mixup_alpha"], 0.1)
        self.assertFalse(followup_config["params"]["aux_logits"])

    def test_build_followup_config_normalizes_head_name_recipe_compatibility(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config = ExperimentConfig.model_validate(
            _build_auto_train_config_payload(
                dataset_name=dataset_name,
                model_family="resnet",
                model_name="resnet18",
                parameter_space_version="resnet18@v1",
            )
        )

        followup_config = _build_followup_config(
            latest_experiment={"config": latest_config.model_dump(mode="python")},
            proposal_payload={
                "changes": {"head_name": "dropout_linear", "mixup_alpha": 0.2},
                "train_hyp_changes": {"augmentation": {"mixup": 0.2}},
                "recipe_changes": {
                    "head_config": {"classifier_type": "dropout_linear"},
                },
            },
        )

        self.assertEqual(followup_config["model_recipe"]["components"]["head"]["name"], "dropout_linear")
        self.assertEqual(followup_config["model_recipe"]["head_config"]["classifier_type"], "linear")
        self.assertEqual(followup_config["model_recipe"]["head_config"]["classifier_dropout"], 0.2)
        self.assertEqual(followup_config["train_hyp"]["augmentation"]["mixup"], 0.2)

    def test_build_followup_config_fills_missing_component_slots_for_partial_head_changes(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config_payload = _build_auto_train_config_payload(
            dataset_name=dataset_name,
            model_family="resnet",
            model_name="resnet18",
            parameter_space_version="resnet18@v1",
        )
        latest_config_payload["model_recipe"]["components"] = None
        latest_config = ExperimentConfig.model_validate(latest_config_payload)

        followup_config = _build_followup_config(
            latest_experiment={"config": latest_config.model_dump(mode="python")},
            proposal_payload={
                "changes": {"head_name": "dropout_linear"},
                "recipe_changes": {
                    "components": {
                        "head": {"name": "dropout_linear"},
                    },
                },
            },
        )

        self.assertEqual(followup_config["model_recipe"]["components"]["backbone"]["name"], "resnet18_native")
        self.assertEqual(followup_config["model_recipe"]["components"]["neck"]["name"], "avg_pool")
        self.assertEqual(followup_config["model_recipe"]["components"]["head"]["name"], "dropout_linear")

    def test_build_followup_config_normalizes_top_level_component_slot_payloads(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config = ExperimentConfig.model_validate(
            _build_auto_train_config_payload(
                dataset_name=dataset_name,
                model_family="resnet",
                model_name="resnet18",
                parameter_space_version="resnet18@v1",
            )
        )

        followup_config = _build_followup_config(
            latest_experiment={"config": latest_config.model_dump(mode="python")},
            proposal_payload={
                "changes": {},
                "recipe_changes": {
                    "neck": {"name": "gem_pool"},
                },
            },
        )

        self.assertEqual(followup_config["model_recipe"]["components"]["backbone"]["name"], "resnet18_native")
        self.assertEqual(followup_config["model_recipe"]["components"]["neck"]["name"], "gem_pool")
        self.assertEqual(followup_config["model_recipe"]["components"]["head"]["name"], "native_classifier")

    def test_build_followup_config_normalizes_legacy_component_name_fields(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config = ExperimentConfig.model_validate(
            _build_auto_train_config_payload(
                dataset_name=dataset_name,
                model_family="resnet",
                model_name="resnet18",
                parameter_space_version="resnet18@v1",
            )
        )

        followup_config = _build_followup_config(
            latest_experiment={"config": latest_config.model_dump(mode="python")},
            proposal_payload={
                "changes": {},
                "recipe_changes": {
                    "neck_name": "gem_pool",
                    "head_name": "dropout_linear",
                },
            },
        )

        self.assertEqual(followup_config["model_recipe"]["components"]["neck"]["name"], "gem_pool")
        self.assertEqual(followup_config["model_recipe"]["components"]["head"]["name"], "dropout_linear")
        self.assertEqual(followup_config["model_recipe"]["head_config"]["pooling_type"], "gem")
        self.assertEqual(followup_config["model_recipe"]["head_config"]["classifier_type"], "linear")
        self.assertEqual(followup_config["model_recipe"]["head_config"]["classifier_dropout"], 0.2)

    def test_build_followup_config_prefers_flat_changes_over_ai_structured_patches(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config = ExperimentConfig.model_validate(
            _build_auto_train_config_payload(
                dataset_name=dataset_name,
                model_family="resnet",
                model_name="resnet18",
                parameter_space_version="resnet18@v1",
            )
        )

        followup_config = _build_followup_config(
            latest_experiment={"config": latest_config.model_dump(mode="python")},
            proposal_payload={
                "changes": {
                    "mixup_alpha": 0.2,
                    "head_name": "dropout_linear",
                },
                "train_hyp_changes": {"augmentation": {"mixup": 0.7}},
                "recipe_changes": {
                    "components": {
                        "head": {"name": "linear"},
                    },
                },
            },
        )

        self.assertEqual(followup_config["train_hyp"]["augmentation"]["mixup"], 0.2)
        self.assertEqual(followup_config["model_recipe"]["components"]["head"]["name"], "dropout_linear")

    def test_validate_followup_proposal_rejects_builder_incompatible_recipe_changes(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        latest_config = ExperimentConfig.model_validate(
            _build_auto_train_config_payload(
                dataset_name=dataset_name,
                model_family="resnet",
                model_name="resnet18",
                parameter_space_version="resnet18@v1",
            )
        )

        with (
            patch(
                "app.services.auto_train_service.SessionLocal",
                return_value=SimpleNamespace(close=lambda: None),
            ),
            patch(
                "app.services.auto_train_service._resolve_followup_source_experiment",
                return_value={"config": latest_config.model_dump(mode="python")},
            ),
        ):
            rejection_reason = _validate_followup_proposal(
                "run_1",
                {
                    "changes": {},
                    "recipe_changes": {
                        "head_config": {"classifier_type": "dropout_linear"},
                    },
                },
            )

        self.assertIsNotNone(rejection_reason)
        self.assertIn("Unsupported classifier_type for resnet18: dropout_linear", rejection_reason)

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

    def test_run_auto_train_task_stops_after_failed_baseline(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        request = AutoTrainStartRequest(
            run_name="Auto Train Test",
            dataset=dataset_name,
            model_name="mobilenet_v3_small",
            config=_build_auto_train_config(dataset_name),
            parameter_space=get_parameter_space("mobilenet_v3_small"),
        )
        task_id = "auto_failed_baseline"
        AUTO_TRAIN_TASKS[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "current_round": 0,
            "logs": [],
        }

        with (
            patch("app.services.auto_train_service.upsert_task_payload"),
            patch("app.services.auto_train_service.SessionLocal", return_value=SimpleNamespace(close=lambda: None)),
            patch("app.services.auto_train_service.create_run", return_value=SimpleNamespace(id="run_1")),
            patch(
                "app.services.auto_train_service._run_experiment_with_retries",
                return_value={
                    "id": "exp_baseline",
                    "status": "failed",
                    "decision_reason": "train.image_size must be declared in data.metadata.image_size_options",
                    "result": {},
                },
            ),
            patch("app.services.auto_train_service._try_attach_auto_train_ai_summary") as mock_attach_summary,
            patch("app.services.auto_train_service._generate_auto_train_proposal") as mock_generate_proposal,
        ):
            _run_auto_train_task(task_id, request)

        task_payload = AUTO_TRAIN_TASKS[task_id]
        self.assertEqual(task_payload["status"], "failed")
        self.assertIn("Baseline exp_baseline failed:", task_payload["error"])
        self.assertIn("train.image_size must be declared", task_payload["error"])
        self.assertIn("stopped before generating any AI proposals", task_payload["error"])
        self.assertEqual(task_payload["current_round"], 0)
        self.assertEqual(task_payload["summary"]["baseline"]["experiment_id"], "exp_baseline")
        self.assertIsNone(task_payload["summary"].get("ai_summary"))
        self.assertNotIn("Round 1: generating AI proposal", task_payload["logs"])
        mock_attach_summary.assert_not_called()
        mock_generate_proposal.assert_not_called()

    def test_is_retryable_proposal_error_accepts_read_timeout_request_exception(self) -> None:
        self.assertTrue(_is_retryable_proposal_error(requests.ReadTimeout("read timed out")))

    def test_is_retryable_proposal_error_accepts_wrapped_timeout_message(self) -> None:
        wrapped_timeout_error = AIHubMixRequestError(
            "chat completions request failed after retries: "
            "HTTPSConnectionPool(host='aihubmix.com', port=443): Read timed out. (read timeout=60)"
        )

        self.assertTrue(_is_retryable_proposal_error(wrapped_timeout_error))

    def test_generate_auto_train_proposal_retries_with_validator_feedback(self) -> None:
        task_id = "auto_proposal_retry"
        AUTO_TRAIN_TASKS[task_id] = {
            "task_id": task_id,
            "status": "running",
            "logs": [],
        }
        first_proposal = SimpleNamespace(model_dump=lambda: {"changes": {"image_size": 512}})
        second_proposal = SimpleNamespace(model_dump=lambda: {"changes": {"image_size": 128}})
        mock_generate_proposal = Mock(side_effect=[first_proposal, second_proposal])
        validator = Mock(
            side_effect=[
                "Proposal cannot build a valid follow-up config: image_size must stay within dataset bounds",
                None,
            ]
        )

        with (
            patch("app.services.auto_train_service.SessionLocal", return_value=SimpleNamespace(close=lambda: None)),
            patch("app.services.auto_train_service.generate_aihubmix_proposal", mock_generate_proposal),
            patch("app.services.auto_train_service._build_proposal_retry_delay_seconds", return_value=0),
            patch("app.services.auto_train_service._sleep_with_stop_check"),
            patch("app.services.auto_train_service.upsert_task_payload"),
        ):
            proposal = _generate_auto_train_proposal(
                task_id,
                "run_1",
                proposal_validator=validator,
            )

        self.assertIs(proposal, second_proposal)
        self.assertEqual(mock_generate_proposal.call_count, 2)
        self.assertIsNone(mock_generate_proposal.call_args_list[0].kwargs["retry_feedback"])
        self.assertEqual(
            mock_generate_proposal.call_args_list[1].kwargs["retry_feedback"],
            "Proposal cannot build a valid follow-up config: image_size must stay within dataset bounds",
        )
        self.assertIn("Proposal attempt failed (attempt 1)", AUTO_TRAIN_TASKS[task_id]["logs"][0])
        self.assertIn("image_size must stay within dataset bounds", AUTO_TRAIN_TASKS[task_id]["logs"][0])
        self.assertIsNone(AUTO_TRAIN_TASKS[task_id]["proposal_warning"])

    def test_build_search_scope_summary_marks_training_budget_when_epochs_are_searchable(self) -> None:
        scope_summary = _build_search_scope_summary(
            {
                "allow_basic_hparam_search": True,
                "allowed_basic_hparam_fields": ["learning_rate", "epochs"],
                "allow_strategy_search": False,
                "allow_loss_search": True,
                "allow_augmentation_search": True,
                "allow_model_module_search": True,
            }
        )

        self.assertEqual(scope_summary, "Basic / Training Budget / Loss / Data Augmentation / Architecture")

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

    def test_build_auto_train_summary_prompt_contains_stop_reason(self) -> None:
        dataset_name = f"dataset_{uuid4().hex}"
        system_prompt, user_prompt = _build_auto_train_summary_prompt(
            run_payload={
                "id": "run_1",
                "dataset": dataset_name,
                "model_name": "mobilenet_v2",
                "best_experiment_id": "exp_best",
                "experiment_count": 2,
            },
            experiment_history=[
                {"id": "exp_1", "status": "success", "metrics": {"top1_acc": 0.81}},
                {"id": "exp_2", "status": "failed", "metrics": {}},
            ],
            search_summary={
                "mode": "auto",
                "run_id": "run_1",
                "baseline": {"experiment_id": "exp_1", "summary": "top1_acc=0.81"},
                "rounds": [],
            },
            stop_reason="Stopped by user request.",
        )

        self.assertIn("summary_text", system_prompt)
        self.assertIn("summary_text 必须使用简洁、客观、自然的中文", system_prompt)
        self.assertIn("这是一段结果摘要，不是下一步 proposal", system_prompt)
        self.assertIn("best 更新次数或阶段数", system_prompt)
        self.assertIn("暂未形成明确结论", system_prompt)
        self.assertIn("搜索任务概览", user_prompt)
        self.assertIn("停止原因", user_prompt)
        self.assertIn("Baseline 上下文", user_prompt)
        self.assertIn("\"best_experiment_id\": \"exp_best\"", user_prompt)
        self.assertIn("\"top1_acc\": 0.81", user_prompt)
        self.assertIn("\"Stopped by user request.\"", user_prompt)
        self.assertNotIn("输出结构", user_prompt)
        self.assertNotIn("策略上下文", user_prompt)
        self.assertNotIn("上一条完整 Proposal 的拒绝反馈", user_prompt)

    def test_try_attach_auto_train_ai_summary_updates_summary(self) -> None:
        search_summary = {
            "mode": "auto",
            "run_id": "run_1",
            "baseline": {"experiment_id": "exp_1", "summary": "top1_acc=0.81"},
            "rounds": [
                {"round_index": 1, "result": {"experiment_id": "exp_2"}},
                {"round_index": 2, "result": {"experiment_id": "exp_3"}},
                {"round_index": 3, "result": {"experiment_id": "exp_4"}},
            ],
            "current_proposal": None,
        }

        with patch(
            "app.services.auto_train_service._generate_auto_train_ai_summary",
            return_value=(
                "本次搜索由用户手动停止，共完成两轮有效 follow-up。"
                "exp_1 仍是当前领先实验，top1_acc 为 0.81。"
                "基线方向是这次搜索里最稳定的有效方案。"
                "其余 follow-up 暂未形成明确增益。"
            ),
        ):
            updated_summary = _try_attach_auto_train_ai_summary(
                "auto_test",
                "run_1",
                search_summary,
                stop_reason="Stopped by user request.",
            )

        self.assertEqual(
            updated_summary["ai_summary"],
            (
                "本次搜索由用户手动停止，共完成两轮有效 follow-up。"
                "exp_1 仍是当前领先实验，top1_acc 为 0.81。"
                "基线方向是这次搜索里最稳定的有效方案。"
                "其余 follow-up 暂未形成明确增益。"
            ),
        )
        self.assertIsNone(updated_summary.get("ai_summary_error"))

    def test_try_attach_auto_train_ai_summary_skips_short_searches(self) -> None:
        search_summary = {
            "mode": "auto",
            "run_id": "run_1",
            "baseline": {"experiment_id": "exp_1", "summary": "top1_acc=0.81"},
            "rounds": [
                {"round_index": 1, "result": {"experiment_id": "exp_2"}},
                {"round_index": 2, "result": {"experiment_id": "exp_3"}},
            ],
            "current_proposal": None,
        }

        with patch("app.services.auto_train_service._generate_auto_train_ai_summary") as mock_generate_summary:
            updated_summary = _try_attach_auto_train_ai_summary(
                "auto_test",
                "run_1",
                search_summary,
                stop_reason="Stopped by user request.",
            )

        mock_generate_summary.assert_not_called()
        self.assertIsNone(updated_summary["ai_summary"])
        self.assertIsNone(updated_summary["ai_summary_error"])

    def test_try_attach_auto_train_ai_summary_records_error_when_provider_fails(self) -> None:
        search_summary = {
            "mode": "auto",
            "run_id": "run_1",
            "baseline": {"experiment_id": "exp_1", "summary": "top1_acc=0.81"},
            "rounds": [
                {"round_index": 1, "result": {"experiment_id": "exp_2"}},
                {"round_index": 2, "result": {"experiment_id": "exp_3"}},
                {"round_index": 3, "result": {"experiment_id": "exp_4"}},
            ],
            "current_proposal": None,
        }

        with (
            patch(
                "app.services.auto_train_service._generate_auto_train_ai_summary",
                side_effect=RuntimeError("provider quota exceeded"),
            ),
            patch("app.services.auto_train_service._append_task_log"),
        ):
            updated_summary = _try_attach_auto_train_ai_summary(
                "auto_test",
                "run_1",
                search_summary,
                stop_reason="Stopped by user request.",
            )

        self.assertIsNone(updated_summary["ai_summary"])
        self.assertEqual(updated_summary["ai_summary_error"], "provider quota exceeded")

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
