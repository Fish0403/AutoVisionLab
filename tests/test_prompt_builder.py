"""Unit tests for proposal prompt bundle assembly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.context.prompt_builder import build_proposal_prompt_bundle


def _build_experiment(
    experiment_id: str,
    *,
    learning_rate: float,
    image_size: int,
    status: str = "success",
    decision: str | None = None,
    is_best_so_far: bool = False,
    top1_acc: float = 0.9,
    latency_ms: float = 12.0,
    error_summary: str | None = None,
) -> dict:
    return {
        "id": experiment_id,
        "status": status,
        "decision": decision,
        "is_best_so_far": is_best_so_far,
        "summary": f"{experiment_id} summary",
        "error_summary": error_summary,
        "config": {
            "params": {
                "learning_rate": learning_rate,
                "image_size": image_size,
                "neck_name": "avg_pool",
            },
            "train_hyp": {
                "lr0": learning_rate,
                "image_size": image_size,
            },
            "model_recipe": {
                "components": {"neck": {"name": "avg_pool"}},
            },
        },
        "result": {
            "status": status,
            "metrics": {
                "top1_acc": top1_acc,
                "val_loss": 0.1,
                "best_epoch": 8,
            },
            "resource": {
                "training_seconds": 120,
                "gpu_memory_mb": 512,
                "latency_ms": latency_ms,
                "parameter_count_million": 3.1,
            },
        },
    }


class PromptBuilderTest(unittest.TestCase):
    """Verify proposal bundle assembly and block ordering."""

    def test_build_proposal_prompt_bundle_omits_empty_initial_history_blocks(self) -> None:
        run_payload = {
            "id": "run_initial_prompt",
            "name": "demo-initial-prompt",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "exp_base",
        }
        experiment_history = [
            _build_experiment("exp_base", learning_rate=0.003, image_size=224, status="success", is_best_so_far=True, top1_acc=0.91),
        ]

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 224},
        )

        self.assertEqual(
            [block.name for block in bundle.blocks],
            [
                "system_prompt",
                "output_schema_prompt",
                "policy_prompt",
                "base_prompt",
            ],
        )
        self.assertNotIn("当前 Source 上下文", bundle.user_prompt)
        self.assertNotIn("当前阶段历史", bundle.user_prompt)
        self.assertIn("当前 run 背景：dataset=NEU, model=mobilenet_v3_small。", bundle.system_prompt)
        self.assertIn("如果没有单独的 source block，表示当前 source 与 base 相同。", bundle.system_prompt)

    def test_build_proposal_prompt_bundle_reads_flat_history_payload_from_runtime(self) -> None:
        run_payload = {
            "id": "run_flat_payload",
            "name": "demo-flat-payload",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "exp_source",
        }
        experiment_history = [
            {
                "id": "exp_base",
                "status": "success",
                "decision": "keep",
                "params": {
                    "learning_rate": 0.003,
                    "image_size": 224,
                    "neck_name": "avg_pool",
                },
                "train_hyp": {
                    "lr0": 0.003,
                    "image_size": 224,
                },
                "model_recipe": {
                    "components": {"neck": {"name": "avg_pool"}},
                },
                "metrics": {"top1_acc": 0.91, "val_loss": 0.1, "best_epoch": 8},
                "resource": {
                    "training_seconds": 120,
                    "gpu_memory_mb": 512,
                    "latency_ms": 12.0,
                    "parameter_count_million": 3.1,
                },
            },
            {
                "id": "exp_mid",
                "status": "success",
                "decision": "discard",
                "params": {
                    "learning_rate": 0.0025,
                    "image_size": 224,
                    "neck_name": "avg_pool",
                },
                "train_hyp": {
                    "lr0": 0.0025,
                    "image_size": 224,
                },
                "model_recipe": {
                    "components": {"neck": {"name": "avg_pool"}},
                },
                "metrics": {"top1_acc": 0.92, "val_loss": 0.09, "best_epoch": 8},
                "resource": {
                    "training_seconds": 118,
                    "gpu_memory_mb": 500,
                    "latency_ms": 11.8,
                    "parameter_count_million": 3.1,
                },
            },
            {
                "id": "exp_best_1",
                "status": "success",
                "decision": "keep",
                "params": {
                    "learning_rate": 0.002,
                    "image_size": 224,
                    "neck_name": "avg_pool",
                },
                "train_hyp": {
                    "lr0": 0.002,
                    "image_size": 224,
                },
                "model_recipe": {
                    "components": {"neck": {"name": "avg_pool"}},
                },
                "metrics": {"top1_acc": 0.93, "val_loss": 0.08, "best_epoch": 8},
                "resource": {
                    "training_seconds": 116,
                    "gpu_memory_mb": 496,
                    "latency_ms": 11.5,
                    "parameter_count_million": 3.1,
                },
            },
            {
                "id": "exp_source",
                "status": "success",
                "decision": "keep",
                "is_best_so_far": True,
                "params": {
                    "learning_rate": 0.001,
                    "image_size": 256,
                    "neck_name": "gem_pool",
                },
                "train_hyp": {
                    "lr0": 0.001,
                    "image_size": 256,
                },
                "model_recipe": {
                    "components": {"neck": {"name": "gem_pool"}},
                },
                "metrics": {"top1_acc": 0.95, "val_loss": 0.07, "best_epoch": 8},
                "resource": {
                    "training_seconds": 114,
                    "gpu_memory_mb": 480,
                    "latency_ms": 11.0,
                    "parameter_count_million": 3.1,
                },
            },
        ]

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 256},
        )

        base_block = next(block for block in bundle.blocks if block.name == "base_prompt")
        source_block = next(block for block in bundle.blocks if block.name == "source_prompt")
        past_stage_block = next(block for block in bundle.blocks if block.name == "past_stage_summaries_prompt")
        self.assertEqual(base_block.payload.params["learning_rate"], 0.003)
        self.assertEqual(source_block.payload.result_snapshot.metrics.top1_acc, 0.95)
        self.assertEqual(source_block.payload.result_snapshot.metrics.latency_ms, 11.0)
        self.assertEqual(source_block.payload.result_snapshot.resource.training_seconds, 114)
        self.assertEqual(
            [(stage["from_best_experiment_id"], stage["to_best_experiment_id"]) for stage in past_stage_block.payload["stages"]],
            [("exp_base", "exp_best_1"), ("exp_best_1", "exp_source")],
        )
        self.assertEqual(past_stage_block.payload["stages"][0]["covered_experiment_ids"], ["exp_mid"])

    def test_build_proposal_prompt_bundle_normalizes_best_experiment_ids(self) -> None:
        run_payload = {
            "id": "run_normalized_ids",
            "name": "demo-normalized-ids",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "3",
        }
        experiment_history = [
            _build_experiment(1, learning_rate=0.003, image_size=224, status="success", is_best_so_far=True),
            _build_experiment(2, learning_rate=0.002, image_size=224, status="success", decision="discard"),
            _build_experiment(3, learning_rate=0.001, image_size=224, status="success", is_best_so_far=True, top1_acc=0.95),
            _build_experiment(4, learning_rate=0.0005, image_size=256, status="failed", decision="crash"),
        ]

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 224},
        )

        source_block = next(block for block in bundle.blocks if block.name == "source_prompt")
        stage_history_block = next(block for block in bundle.blocks if block.name == "stage_history_prompt")
        self.assertEqual(source_block.payload.experiment_id, "3")
        self.assertEqual(
            [item["experiment_id"] for item in stage_history_block.payload["items"]],
            ["4"],
        )

    def test_build_proposal_prompt_bundle_includes_retry_and_metrics(self) -> None:
        run_payload = {
            "id": "run_1",
            "name": "demo",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "exp_best",
        }
        experiment_history = [
            _build_experiment("exp_base", learning_rate=0.003, image_size=224, status="success", is_best_so_far=True),
            _build_experiment("exp_mid", learning_rate=0.002, image_size=224, status="success", decision="discard"),
            _build_experiment("exp_best", learning_rate=0.001, image_size=224, status="success", is_best_so_far=True, top1_acc=0.95),
            _build_experiment(
                "exp_fail",
                learning_rate=0.0005,
                image_size=256,
                status="failed",
                decision="crash",
                error_summary="cuda out of memory",
            ),
        ]

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 224},
            retry_feedback="image_size is not allowed",
        )

        self.assertEqual(
            [block.name for block in bundle.blocks],
            [
                "system_prompt",
                "output_schema_prompt",
                "policy_prompt",
                "base_prompt",
                "source_prompt",
                "stage_history_prompt",
                "past_stage_summaries_prompt",
                "retry_prompt",
            ],
        )
        self.assertGreater(bundle.prompt_chars, 0)
        self.assertGreater(bundle.prompt_tokens_estimate, 0)
        self.assertIn("latency_ms", bundle.user_prompt)
        self.assertIn("cuda out of memory", bundle.user_prompt)
        self.assertIn("image_size is not allowed", bundle.user_prompt)
        self.assertIn("证据关系如下：base 是初始 baseline；source 是当前参考实验", bundle.system_prompt)

    def test_build_proposal_prompt_bundle_compacts_long_current_stage(self) -> None:
        run_payload = {
            "id": "run_2",
            "name": "demo-2",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "exp_best",
        }
        experiment_history = [
            _build_experiment("exp_base", learning_rate=0.003, image_size=224, status="success", is_best_so_far=True),
            _build_experiment("exp_best", learning_rate=0.001, image_size=224, status="success", is_best_so_far=True, top1_acc=0.95),
        ]
        for index in range(21):
            experiment_history.append(
                _build_experiment(
                    f"exp_try_{index}",
                    learning_rate=0.001 - (index + 1) * 0.00005,
                    image_size=224 + index,
                    status="success",
                    decision="discard" if index % 2 else "keep",
                )
            )

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 224},
        )

        self.assertIn("current_stage_compacted_prompt", [block.name for block in bundle.blocks])
        stage_history_block = next(block for block in bundle.blocks if block.name == "stage_history_prompt")
        compacted_block = next(block for block in bundle.blocks if block.name == "current_stage_compacted_prompt")
        self.assertEqual(len(stage_history_block.payload["items"]), 10)
        self.assertEqual(len(compacted_block.payload.buckets), 2)
        self.assertEqual(compacted_block.payload.buckets[0].covered_experiment_ids[0], "exp_try_0")
        self.assertEqual(compacted_block.payload.buckets[1].covered_experiment_ids, ["exp_try_10"])

    def test_build_proposal_prompt_bundle_includes_stage_leading_to_current_source(self) -> None:
        run_payload = {
            "id": "run_3",
            "name": "demo-3",
            "dataset": "NEU",
            "model_name": "mobilenet_v3_small",
            "best_experiment_id": "exp_source",
        }
        experiment_history = [
            _build_experiment("exp_base", learning_rate=0.003, image_size=224, status="success", decision="keep"),
            _build_experiment("exp_mid_a", learning_rate=0.0025, image_size=224, status="success", decision="discard"),
            _build_experiment("exp_best_1", learning_rate=0.002, image_size=224, status="success", decision="keep", top1_acc=0.93),
            _build_experiment("exp_source", learning_rate=0.001, image_size=256, status="success", decision="keep", is_best_so_far=True, top1_acc=0.95),
            _build_experiment("exp_after", learning_rate=0.0005, image_size=288, status="failed", decision="crash"),
        ]

        bundle = build_proposal_prompt_bundle(
            run_payload=run_payload,
            experiment_history=experiment_history,
            policy_payload={"allowed_fields": ["learning_rate", "image_size"]},
            source_constraints={"image_size": 256},
        )

        past_stage_block = next(block for block in bundle.blocks if block.name == "past_stage_summaries_prompt")
        stages = past_stage_block.payload["stages"]
        self.assertEqual(
            [(stage["from_best_experiment_id"], stage["to_best_experiment_id"]) for stage in stages],
            [("exp_base", "exp_best_1"), ("exp_best_1", "exp_source")],
        )
        self.assertEqual(stages[0]["covered_experiment_ids"], ["exp_mid_a"])
        self.assertEqual(stages[1]["covered_experiment_ids"], [])


if __name__ == "__main__":
    unittest.main()
