"""Unit tests for AI proposal generation retries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.ai import ProposalSchema
from app.schemas.parameter_space import EditableParameterSpace, SearchPolicy
from app.services.proposal_service import generate_aihubmix_proposal


def _build_parameter_space() -> EditableParameterSpace:
    """Build a minimal parameter space used by the retry test."""
    return EditableParameterSpace.model_validate(
        {
            "model_name": "mobilenet_v3_small",
            "version": "test-v1",
            "editable_params": {
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
            },
        }
    )


class ProposalServiceTest(unittest.TestCase):
    """Verify retry behavior for invalid AI proposals."""

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
                "risk": "medium",
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
                "risk": "medium",
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
                "risk": "medium",
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
                "risk": "medium",
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
                    "risk": "low",
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
                    "risk": "low",
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
        self.assertIn("请基于完整历史换一个更可执行的方向", second_prompt)

    def test_generate_aihubmix_proposal_ignores_disabled_run_search_policy(self) -> None:
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
                "risk": "low",
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
            proposal = generate_aihubmix_proposal(db, "run_1")

        self.assertEqual(proposal.changes.weight_decay, 0.0005)
        mock_client.create_json_completion_with_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main()
