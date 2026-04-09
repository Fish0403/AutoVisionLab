"""Unit tests for AI proposal generation retries."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "test_proposal_service"
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

os.environ["AVL_ARTIFACT_ROOT"] = str(TEST_ARTIFACT_ROOT)

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.ai import ProposalSchema
from app.schemas.parameter_space import EditableParameterSpace, SearchPolicy
from app.services.proposal_context_cache import clear_run_proposal_context_cache
from app.services.proposal_service import (
    _build_run_context_cache_entry_from_history,
    _load_or_refresh_run_context_cache_entry,
    generate_aihubmix_proposal,
)


def _build_parameter_space() -> EditableParameterSpace:
    """Build a minimal parameter space used by the retry test."""
    return EditableParameterSpace.model_validate(
        {
            "model_name": "mobilenet_v3_small",
            "version": "test-v1",
            "editable_params": {
                "epochs": {
                    "type": "discrete_values",
                    "choices": [10, 20, 30],
                },
                "label_smoothing": {
                    "type": "number_range",
                    "min": 0.0,
                    "max": 0.2,
                },
                "weight_decay": {
                    "type": "number_range",
                    "min": 0.0,
                    "max": 0.01,
                },
                "image_size": {
                    "type": "number_range",
                    "min": 1,
                    "max": 10000,
                },
            },
        }
    )


class ProposalServiceTest(unittest.TestCase):
    """Verify retry behavior for invalid AI proposals."""

    def setUp(self) -> None:
        clear_run_proposal_context_cache()
        if TEST_ARTIFACT_ROOT.exists():
            shutil.rmtree(TEST_ARTIFACT_ROOT)

    def tearDown(self) -> None:
        clear_run_proposal_context_cache()
        if TEST_ARTIFACT_ROOT.exists():
            shutil.rmtree(TEST_ARTIFACT_ROOT)

    def test_proposal_schema_backfills_recipe_change_views(self) -> None:
        proposal = ProposalSchema.model_validate(
            {
                "task_type": "classification",
                "model_name": "googlenet",
                "based_on_experiment_ids": ["exp_1"],
                "hypothesis": "Enable aux heads and add mixup.",
                "changes": {
                    "mixup_alpha": 0.2,
                    "weight_decay": 0.0005,
                    "aux_logits": True,
                },
                "reason": "Test one recipe change plus one train hyp change.",
            }
        )

        self.assertEqual(
            proposal.train_hyp_changes,
            {
                "weight_decay": 0.0005,
                "augmentation": {"mixup": 0.2},
            },
        )
        self.assertEqual(
            proposal.recipe_changes,
            {
                "modules": {"aux_logits": True},
            },
        )

    def test_proposal_schema_maps_model_recipe_structure_fields(self) -> None:
        proposal = ProposalSchema.model_validate(
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_1"],
                "hypothesis": "Widen the model and switch to GeM pooling.",
                "changes": {
                    "width_multiple": 1.25,
                    "pooling_type": "gem",
                    "classifier_dropout": 0.2,
                },
                "reason": "Test one wider backbone with a stronger head pooling variant.",
            }
        )

        self.assertIsNone(proposal.train_hyp_changes)
        self.assertEqual(
            proposal.recipe_changes,
            {
                "width_multiple": 1.25,
                "head_config": {
                    "pooling_type": "gem",
                    "classifier_dropout": 0.2,
                },
                "components": {
                    "neck": {"name": "gem_pool"},
                },
            },
        )

    def test_proposal_schema_maps_component_level_neck_field(self) -> None:
        proposal = ProposalSchema.model_validate(
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_1"],
                "hypothesis": "尝试把 neck 切到 GeM pooling。",
                "changes": {
                    "neck_name": "gem_pool",
                },
                "reason": "测试组件级搜索字段能否回写成 recipe changes。",
            }
        )

        self.assertIsNone(proposal.train_hyp_changes)
        self.assertEqual(
            proposal.recipe_changes,
            {
                "components": {
                    "neck": {"name": "gem_pool"},
                },
                "head_config": {
                    "pooling_type": "gem",
                },
            },
        )

    def test_proposal_schema_maps_component_level_head_field(self) -> None:
        proposal = ProposalSchema.model_validate(
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_1"],
                "hypothesis": "尝试简化分类头。",
                "changes": {
                    "head_name": "linear",
                },
                "reason": "测试组件级 head 字段能否回写成兼容 recipe changes。",
            }
        )

        self.assertIsNone(proposal.train_hyp_changes)
        self.assertEqual(
            proposal.recipe_changes,
            {
                "components": {
                    "head": {"name": "linear"},
                },
                "head_config": {
                    "classifier_type": "linear",
                    "classifier_dropout": 0.0,
                },
            },
        )

    def test_generate_aihubmix_proposal_retries_with_soft_preference_feedback(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="retry-test",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.side_effect = [
            (
                {
                    "task_type": "classification",
                    "model_name": "mobilenet_v3_small",
                    "based_on_experiment_ids": ["exp_fail_1", "exp_fail_2"],
                    "hypothesis": "先给一个空动作。",
                    "changes": {"epochs": 2},
                    "reason": "先试一个会被系统清理掉的字段。",
                },
                {},
            ),
            (
                {
                    "task_type": "classification",
                    "model_name": "mobilenet_v3_small",
                    "based_on_experiment_ids": ["exp_fail_1", "exp_fail_2"],
                    "hypothesis": "改测权重衰减。",
                    "changes": {"weight_decay": 0.0005},
                    "reason": "优先避开最近连续失败的字段，改看正则强度是否更稳。",
                },
                {},
            ),
        ]
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep"},
            {
                "id": "exp_fail_1",
                "status": "success",
                "decision": "discard",
                "proposal": {"changes": {"label_smoothing": 0.05}},
            },
            {
                "id": "exp_fail_2",
                "status": "failed",
                "decision": "crash",
                "proposal": {"changes": {"label_smoothing": 0.10}},
            },
        ]

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.weight_decay, 0.0005)
        self.assertIsNone(proposal.changes.label_smoothing)
        self.assertEqual(proposal.train_hyp_changes, {"weight_decay": 0.0005})
        self.assertIsNone(proposal.recipe_changes)
        self.assertEqual(mock_client.create_json_completion_with_metadata.call_count, 2)
        second_prompt = mock_client.create_json_completion_with_metadata.call_args_list[1].kwargs["user_prompt"]
        self.assertIn("Proposal does not contain any effective parameter changes", second_prompt)
        self.assertIn("请基于完整历史选择一个更可执行的方向", second_prompt)

    def test_build_run_context_cache_entry_finalizes_stage_when_new_best_appears(self) -> None:
        run = SimpleNamespace(
            id="run_stage_finalize",
            name="stage-finalize",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_base",
            best_experiment_id="exp_best_2",
        )
        experiment_history = [
            {
                "id": "exp_base",
                "status": "success",
                "decision": "keep",
                "is_best_so_far": True,
                "created_at": "2026-04-08T12:00:00",
                "params": {"learning_rate": 0.003, "image_size": 224},
                "train_hyp": {"lr0": 0.003, "image_size": 224},
                "result": {"status": "success", "metrics": {"top1_acc": 0.91}, "resource": {}},
            },
            {
                "id": "exp_try_1",
                "status": "success",
                "decision": "discard",
                "created_at": "2026-04-08T12:01:00",
                "params": {"learning_rate": 0.0025, "image_size": 224},
                "train_hyp": {"lr0": 0.0025, "image_size": 224},
                "result": {"status": "success", "metrics": {"top1_acc": 0.90}, "resource": {}},
            },
            {
                "id": "exp_try_2",
                "status": "failed",
                "decision": "crash",
                "created_at": "2026-04-08T12:02:00",
                "params": {"learning_rate": 0.002, "image_size": 256},
                "train_hyp": {"lr0": 0.002, "image_size": 256},
                "result": {"status": "failed", "metrics": {"top1_acc": 0.89}, "resource": {}},
                "error_summary": "cuda out of memory",
            },
            {
                "id": "exp_best_2",
                "status": "success",
                "decision": "keep",
                "is_best_so_far": True,
                "created_at": "2026-04-08T12:03:00",
                "params": {"learning_rate": 0.0015, "image_size": 224},
                "train_hyp": {"lr0": 0.0015, "image_size": 224},
                "result": {"status": "success", "metrics": {"top1_acc": 0.94}, "resource": {}},
            },
        ]

        entry = _build_run_context_cache_entry_from_history(
            run=run,
            history_signature=("exp_base", "exp_best_2", "2026-04-08T12:03:00"),
            experiment_history=experiment_history,
        )

        self.assertEqual(entry.base_experiment_payload["id"], "exp_base")
        self.assertEqual(entry.source_experiment_payload["id"], "exp_best_2")
        self.assertEqual(len(entry.past_stage_summaries), 1)
        self.assertEqual(entry.past_stage_summaries[0].covered_experiment_ids, ["exp_try_1", "exp_try_2"])
        self.assertIn("cuda out of memory (x1)", entry.past_stage_summaries[0].failure_patterns)
        self.assertEqual(len(entry.recent_history_queue), 0)
        self.assertEqual(len(entry.compacted_bucket_queue), 0)

    def test_generate_aihubmix_proposal_includes_outer_retry_feedback_in_prompt(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="retry-feedback",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep", "created_at": "2026-04-08T12:00:00"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Adjust weight decay.",
                "changes": {"weight_decay": 0.0005},
                "reason": "Keep the next trial executable after a config-build rejection.",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            generate_aihubmix_proposal(
                db,
                "run_1",
                retry_feedback="Proposal cannot build a valid follow-up config: image_size must stay within dataset bounds",
            )

        first_prompt = mock_client.create_json_completion_with_metadata.call_args.kwargs["user_prompt"]
        system_prompt = mock_client.create_json_completion_with_metadata.call_args.kwargs["system_prompt"]
        self.assertIn("上一条完整 Proposal 的拒绝反馈", first_prompt)
        self.assertIn("image_size must stay within dataset bounds", first_prompt)
        self.assertIn("通常优先选择不大于当前 source experiment image_size 的值", first_prompt)
        self.assertIn("不要修改 epochs；它当前是固定值，AI 不允许调整。", first_prompt)
        self.assertIn("当前 source experiment 的 config 就是下一轮 follow-up 的完整起点", first_prompt)
        self.assertNotIn("allowed_change_fields", first_prompt)
        self.assertNotIn("image_size_definition", first_prompt)
        self.assertIn("mixup_alpha", system_prompt)
        self.assertIn("cutmix_alpha", system_prompt)
        self.assertIn("random_erasing_prob", system_prompt)
        self.assertNotIn('"augmentation_policy"', first_prompt)
        self.assertNotIn("augmentation_policy", system_prompt)
        self.assertIn("不要返回 train_hyp_changes、recipe_changes", system_prompt)

    def test_generate_aihubmix_proposal_accepts_larger_image_size_when_allowed(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="image-size-preference",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep", "created_at": "2026-04-08T12:00:00"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Try a larger image size.",
                "changes": {"image_size": 128},
                "reason": "Probe a bigger crop that is still inside the allowed search space.",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch(
                "app.services.proposal_service._load_latest_search_policy",
                return_value=SearchPolicy(
                    allow_basic_hparam_search=True,
                    allowed_basic_hparam_fields=["image_size"],
                    allow_strategy_search=False,
                    allow_loss_search=False,
                    allow_augmentation_search=False,
                    allow_model_module_search=False,
                ),
            ),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.image_size, 128)
        mock_client.create_json_completion_with_metadata.assert_called_once()

    def test_generate_aihubmix_proposal_accepts_concrete_regularization_wording(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="text-hint-precision",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "A slightly lower learning rate with higher weight decay may improve stability.",
                "changes": {"weight_decay": 0.0005},
                "reason": "Keep the plan simple while using the current best result as the baseline.",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.weight_decay, 0.0005)
        mock_client.create_json_completion_with_metadata.assert_called_once()

    def test_generate_aihubmix_proposal_respects_disabled_run_search_policy(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="empty-search-policy",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "改测权重衰减。",
                "changes": {"weight_decay": 0.0005},
                "reason": "当前 run 允许 AI 在模型 parameter space 内自主搜索。",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch(
                "app.services.proposal_service._load_latest_search_policy",
                return_value=SearchPolicy(
                    allow_basic_hparam_search=False,
                    allowed_basic_hparam_fields=[],
                    allow_strategy_search=False,
                    allow_loss_search=False,
                    allow_augmentation_search=False,
                    allow_model_module_search=False,
                ),
            ),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
        ):
            with self.assertRaisesRegex(ValueError, "No AI-editable fields are available for this run."):
                generate_aihubmix_proposal(db, "run_1")

        mock_client.create_json_completion_with_metadata.assert_not_called()

    def test_generate_aihubmix_proposal_allows_epoch_changes_when_search_policy_enables_it(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="epoch-search-enabled",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Train longer to confirm whether the current direction keeps improving.",
                "changes": {"epochs": 20},
                "reason": "Use a larger training budget for the next attempt.",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch(
                "app.services.proposal_service._load_latest_search_policy",
                return_value=SearchPolicy(
                    allow_basic_hparam_search=True,
                    allowed_basic_hparam_fields=["epochs"],
                    allow_strategy_search=False,
                    allow_loss_search=False,
                    allow_augmentation_search=False,
                    allow_model_module_search=False,
                ),
            ),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.epochs, 20)
        prompt = mock_client.create_json_completion_with_metadata.call_args.kwargs["user_prompt"]
        self.assertIn("你可以在有帮助时调整 epochs", prompt)

    def test_generate_aihubmix_proposal_reuses_run_context_cache_between_calls(self) -> None:
        db = Mock()
        run = SimpleNamespace(
            id="run_cache",
            name="cache-hit",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
            updated_at=datetime(2026, 4, 8, 12, 0, 0),
        )
        db.get.return_value = run
        db.scalars.return_value.all.return_value = [
            SimpleNamespace(
                id="exp_keep",
                status="success",
                decision="keep",
                decision_reason=None,
                is_best_so_far=True,
                created_at=datetime(2026, 4, 8, 12, 0, 0),
                result={"status": "success", "metrics": {"top1_acc": 0.91}, "resource": {}},
                proposal=None,
                experiment_config={"params": {"learning_rate": 0.003, "image_size": 224}, "train_hyp": {"image_size": 224}},
            )
        ]
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep", "created_at": "2026-04-08T12:00:00"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Adjust weight decay.",
                "changes": {"weight_decay": 0.0005},
                "reason": "Keep the next trial executable.",
            },
            {},
        )
        history_loader = Mock(return_value=experiment_history)
        append_run_log_mock = Mock()

        with (
            patch("app.services.proposal_service.get_run_history_payload", history_loader),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log", append_run_log_mock),
        ):
            first_proposal = generate_aihubmix_proposal(db, "run_cache")
            second_proposal = generate_aihubmix_proposal(db, "run_cache")

        self.assertEqual(first_proposal.changes.weight_decay, 0.0005)
        self.assertEqual(second_proposal.changes.weight_decay, 0.0005)
        self.assertEqual(history_loader.call_count, 1)
        self.assertEqual(mock_client.create_json_completion_with_metadata.call_count, 2)
        logged_messages = [call.args[1] for call in append_run_log_mock.call_args_list if len(call.args) >= 2]
        self.assertTrue(any(message.startswith("INFO |proposal-cache| cache miss;") for message in logged_messages))
        self.assertTrue(any(message.startswith("INFO |proposal-cache| cache hit;") for message in logged_messages))

    def test_generate_aihubmix_proposal_refreshes_run_context_cache_when_run_updates(self) -> None:
        db = Mock()
        run = SimpleNamespace(
            id="run_cache_refresh",
            name="cache-refresh",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
            updated_at=datetime(2026, 4, 8, 12, 0, 0),
        )
        db.get.return_value = run
        db.scalars.return_value.all.return_value = [
            SimpleNamespace(
                id="exp_keep",
                status="success",
                decision="keep",
                decision_reason=None,
                is_best_so_far=True,
                created_at=datetime(2026, 4, 8, 12, 0, 0),
                result={"status": "success", "metrics": {"top1_acc": 0.91}, "resource": {}},
                proposal=None,
                experiment_config={"params": {"learning_rate": 0.003, "image_size": 224}, "train_hyp": {"image_size": 224}},
            )
        ]
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep", "created_at": "2026-04-08T12:00:00"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Adjust weight decay.",
                "changes": {"weight_decay": 0.0005},
                "reason": "Keep the next trial executable.",
            },
            {},
        )
        history_loader = Mock(return_value=experiment_history)
        append_run_log_mock = Mock()

        with (
            patch("app.services.proposal_service.get_run_history_payload", history_loader),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log", append_run_log_mock),
        ):
            generate_aihubmix_proposal(db, "run_cache_refresh")
            run.updated_at = run.updated_at + timedelta(seconds=1)
            generate_aihubmix_proposal(db, "run_cache_refresh")

        self.assertEqual(history_loader.call_count, 1)
        logged_messages = [call.args[1] for call in append_run_log_mock.call_args_list if len(call.args) >= 2]
        self.assertTrue(any(message.startswith("INFO |proposal-cache| cache refresh;") for message in logged_messages))

    def test_load_or_refresh_run_context_cache_entry_advances_existing_state_with_new_best(self) -> None:
        run = SimpleNamespace(
            id="run_cache_advance",
            name="cache-advance",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_base",
            best_experiment_id="exp_base",
            updated_at=datetime(2026, 4, 8, 12, 0, 0),
        )
        initial_history = [
            {
                "id": "exp_base",
                "status": "success",
                "decision": "keep",
                "is_best_so_far": True,
                "created_at": "2026-04-08T12:00:00",
                "params": {"learning_rate": 0.003, "image_size": 224},
                "train_hyp": {"lr0": 0.003, "image_size": 224},
                "result": {"status": "success", "metrics": {"top1_acc": 0.91}, "resource": {}},
            },
            {
                "id": "exp_try",
                "status": "failed",
                "decision": "crash",
                "created_at": "2026-04-08T12:01:00",
                "params": {"learning_rate": 0.002, "image_size": 256},
                "train_hyp": {"lr0": 0.002, "image_size": 256},
                "result": {"status": "failed", "metrics": {"top1_acc": 0.89}, "resource": {}},
                "error_summary": "cuda out of memory",
            },
        ]
        db = Mock()
        db.scalars.return_value.all.return_value = [
            SimpleNamespace(
                id="exp_try",
                status="failed",
                decision="crash",
                decision_reason=None,
                is_best_so_far=False,
                created_at=datetime(2026, 4, 8, 12, 1, 0),
                result={"status": "failed", "metrics": {"top1_acc": 0.89}, "resource": {}},
                proposal=None,
                experiment_config={"params": {"learning_rate": 0.002, "image_size": 256}, "train_hyp": {"image_size": 256}},
            ),
            SimpleNamespace(
                id="exp_best_2",
                status="success",
                decision="keep",
                decision_reason=None,
                is_best_so_far=True,
                created_at=datetime(2026, 4, 8, 12, 2, 0),
                result={"status": "success", "metrics": {"top1_acc": 0.94}, "resource": {}},
                proposal=None,
                experiment_config={"params": {"learning_rate": 0.0015, "image_size": 224}, "train_hyp": {"image_size": 224}},
            ),
        ]

        with patch("app.services.proposal_service.get_run_history_payload", return_value=initial_history):
            initial_entry, initial_status = _load_or_refresh_run_context_cache_entry(db, run)

        self.assertEqual(initial_status, "miss")
        self.assertEqual(initial_entry.source_experiment_payload["id"], "exp_base")
        self.assertEqual(initial_entry.history_item_count, 2)
        self.assertEqual(len(initial_entry.recent_history_queue), 1)

        run.best_experiment_id = "exp_best_2"
        run.updated_at = run.updated_at + timedelta(seconds=1)
        advanced_entry, advanced_status = _load_or_refresh_run_context_cache_entry(db, run)

        self.assertEqual(advanced_status, "advance")
        self.assertEqual(advanced_entry.source_experiment_payload["id"], "exp_best_2")
        self.assertEqual(advanced_entry.history_item_count, 3)
        self.assertEqual(len(advanced_entry.past_stage_summaries), 1)
        self.assertEqual(advanced_entry.past_stage_summaries[0].covered_experiment_ids, ["exp_try"])
        self.assertIn("cuda out of memory (x1)", advanced_entry.past_stage_summaries[0].failure_patterns)
        self.assertEqual(len(advanced_entry.recent_history_queue), 0)

    def test_generate_aihubmix_proposal_omits_duplicate_source_config_summary(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="source-constraints-summary",
            dataset="cifar10",
            model_name="mobilenet_v3_small",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        experiment_history = [
            {"id": "exp_keep", "status": "success", "decision": "keep"},
        ]
        mock_client = Mock()
        mock_client.create_json_completion_with_metadata.return_value = (
            {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_keep"],
                "hypothesis": "Adjust weight decay from the current baseline.",
                "changes": {"weight_decay": 0.0005},
                "reason": "Use the current source constraints as the branching baseline.",
            },
            {},
        )

        with (
            patch("app.services.proposal_service.get_run_history_payload", return_value=experiment_history),
            patch("app.services.proposal_service._load_latest_search_policy", return_value=SearchPolicy()),
            patch(
                "app.services.proposal_service._load_latest_parameter_space",
                return_value=_build_parameter_space(),
            ),
            patch("app.services.proposal_service.AIHubMixClient", return_value=mock_client),
            patch("app.services.proposal_service.append_run_log"),
        ):
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.weight_decay, 0.0005)
        prompt = mock_client.create_json_completion_with_metadata.call_args.kwargs["user_prompt"]
        self.assertNotIn("\"config\": {", prompt)
        system_prompt = mock_client.create_json_completion_with_metadata.call_args.kwargs["system_prompt"]
        self.assertIn("如果没有单独的 source block，表示当前 source 与 base 相同。", system_prompt)


if __name__ == "__main__":
    unittest.main()
