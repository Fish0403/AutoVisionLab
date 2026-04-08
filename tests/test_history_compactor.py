"""Unit tests for proposal history compaction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.context.history_compactor import compact_current_stage_history, compact_past_stage_history
from app.schemas.prompt_context import HistoryItemSummary


def _build_history_item(
    *,
    experiment_id: str,
    decision: str,
    delta_field: str,
    status: str = "success",
    execution_error_summary: str | None = None,
) -> HistoryItemSummary:
    return HistoryItemSummary.model_validate(
        {
            "experiment_id": experiment_id,
            "based_on_source_experiment_id": "exp_source",
            "delta_from_source": [
                {
                    "field": delta_field,
                    "from_value": 0.001,
                    "to_value": 0.0005,
                    "change_type": "decrease",
                }
            ],
            "decision": decision,
            "result_snapshot": {
                "status": status,
                "metrics": {"top1_acc": 0.91},
                "resource": {"training_seconds": 120},
            },
            "outcome_summary": f"{experiment_id} summary",
            "execution_error_summary": execution_error_summary,
        }
    )


class HistoryCompactorTest(unittest.TestCase):
    """Verify current-stage and past-stage compaction behavior."""

    def test_compact_current_stage_history_tracks_failure_patterns(self) -> None:
        compacted = compact_current_stage_history(
            [
                _build_history_item(experiment_id="exp_keep", decision="keep", delta_field="learning_rate"),
                _build_history_item(
                    experiment_id="exp_fail_a",
                    decision="crash",
                    delta_field="batch_size",
                    status="failed",
                    execution_error_summary="cuda out of memory",
                ),
                _build_history_item(
                    experiment_id="exp_fail_b",
                    decision="crash",
                    delta_field="batch_size",
                    status="failed",
                    execution_error_summary="cuda out of memory",
                ),
            ]
        )

        self.assertEqual(len(compacted.buckets), 1)
        self.assertEqual(compacted.buckets[0].effective_directions, ["learning_rate: 0.001 -> 0.0005"])
        self.assertEqual(compacted.buckets[0].ineffective_directions, ["batch_size: 0.001 -> 0.0005"])
        self.assertEqual(compacted.buckets[0].failure_patterns, ["cuda out of memory (x2)"])
        self.assertEqual(compacted.buckets[0].covered_experiment_ids, ["exp_keep", "exp_fail_a", "exp_fail_b"])

    def test_compact_past_stage_history_keeps_best_boundaries(self) -> None:
        compacted = compact_past_stage_history(
            from_best_experiment_id="exp_best_1",
            to_best_experiment_id="exp_best_2",
            history_items=[
                _build_history_item(experiment_id="exp_mid_1", decision="discard", delta_field="scheduler"),
            ],
        )

        self.assertEqual(compacted.from_best_experiment_id, "exp_best_1")
        self.assertEqual(compacted.to_best_experiment_id, "exp_best_2")
        self.assertEqual(compacted.covered_experiment_ids, ["exp_mid_1"])


if __name__ == "__main__":
    unittest.main()
