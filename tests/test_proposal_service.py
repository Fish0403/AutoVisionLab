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

from app.schemas.parameter_space import EditableParameterSpace, SearchPolicy
from app.services.proposal_service import generate_aihubmix_proposal


def _build_parameter_space() -> EditableParameterSpace:
    """Build a minimal parameter space used by the retry test."""
    return EditableParameterSpace.model_validate(
        {
            "model_name": "mobilenet_v2",
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

    def test_generate_aihubmix_proposal_retries_with_explicit_cooldown_feedback(self) -> None:
        db = Mock()
        db.get.return_value = SimpleNamespace(
            id="run_1",
            name="retry-test",
            dataset="cifar10",
            model_name="mobilenet_v2",
            baseline_experiment_id="exp_keep",
            best_experiment_id="exp_keep",
            frontier_experiment_id="exp_keep",
        )
        mock_client = Mock()
        mock_client.create_json_completion.side_effect = [
            {
                "task_type": "classification",
                "model_name": "mobilenet_v2",
                "based_on_experiment_ids": ["exp_fail_1", "exp_fail_2"],
                "hypothesis": "继续提高标签平滑。",
                "changes": {"label_smoothing": 0.12},
                "reason": "想继续增强泛化。",
                "risk": "low",
            },
            {
                "task_type": "classification",
                "model_name": "mobilenet_v2",
                "based_on_experiment_ids": ["exp_fail_1", "exp_fail_2"],
                "hypothesis": "改测权重衰减。",
                "changes": {"weight_decay": 0.0005},
                "reason": "label_smoothing 在冷却期，改看正则强度是否更稳。",
                "risk": "low",
            },
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
        self.assertEqual(mock_client.create_json_completion.call_count, 2)
        second_prompt = mock_client.create_json_completion.call_args_list[1].kwargs["user_prompt"]
        self.assertIn("Proposal reuses fields currently in cooldown: label_smoothing", second_prompt)
        self.assertIn("changes 中绝对不能再次包含", second_prompt)


if __name__ == "__main__":
    unittest.main()
