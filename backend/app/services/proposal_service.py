"""Proposal generation services."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.aihubmix_client import AIHubMixClient
from app.models.experiment import ExperimentModel
from app.models.run import RunModel
from app.schemas.ai import ProposalChanges, ProposalSchema
from app.services.parameter_space import AI_BLOCKED_PROPOSAL_FIELDS, validate_proposal_against_space


def _summarize_experiment_for_prompt(experiment: ExperimentModel, run: RunModel) -> dict[str, Any]:
    """Build a compact experiment summary for proposal prompting."""
    result_payload = experiment.result or {}
    metrics_payload = result_payload.get("metrics") or {}
    resource_payload = result_payload.get("resource") or {}
    proposal_payload = experiment.proposal or {}
    config_payload = experiment.experiment_config or {}
    params_payload = config_payload.get("params") or {}

    return {
        "id": experiment.id,
        "status": experiment.status,
        "decision": experiment.decision,
        "decision_reason": experiment.decision_reason,
        "is_baseline": experiment.id == run.baseline_experiment_id,
        "is_best": experiment.id == run.best_experiment_id,
        "is_frontier": experiment.id == run.frontier_experiment_id,
        "baseline_experiment_id": experiment.baseline_experiment_id,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "metrics": {
            "top1_acc": metrics_payload.get("top1_acc"),
            "val_loss": metrics_payload.get("val_loss"),
            "train_loss": metrics_payload.get("train_loss"),
            "best_epoch": metrics_payload.get("best_epoch"),
        },
        "resource": {
            "training_seconds": resource_payload.get("training_seconds"),
        },
        "params": params_payload,
        "proposal": {
            "based_on_experiment_ids": proposal_payload.get("based_on_experiment_ids"),
            "hypothesis": proposal_payload.get("hypothesis"),
            "changes": (proposal_payload.get("changes") or {}),
            "reason": proposal_payload.get("reason"),
            "risk": proposal_payload.get("risk"),
        }
        if proposal_payload
        else None,
    }


def get_run_history_payload(db: Session, run_id: str) -> list[dict[str, Any]]:
    """Return the full experiment history payload for a run."""
    run = db.get(RunModel, run_id)
    if run is None:
        return []

    experiments = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.asc())
    ).all()
    return [_summarize_experiment_for_prompt(experiment, run) for experiment in experiments]


def sanitize_blocked_proposal_fields(proposal: ProposalSchema) -> ProposalSchema:
    """Clear blocked AI-controlled fields instead of failing the whole proposal."""
    changes_payload = proposal.changes.model_dump()
    sanitized_changes = {
        key: (None if key in AI_BLOCKED_PROPOSAL_FIELDS else value)
        for key, value in changes_payload.items()
    }
    return proposal.model_copy(update={"changes": ProposalChanges.model_validate(sanitized_changes)})


def generate_aihubmix_proposal(db: Session, run_id: str) -> ProposalSchema:
    """Generate a structured proposal using AIHubMix."""
    run = db.get(RunModel, run_id)
    if run is None:
        raise ValueError("Run not found")

    experiment_history = get_run_history_payload(db, run_id)
    if not experiment_history:
        raise ValueError("No experiment is available for this run")

    prompt_run_payload = {
        "id": run.id,
        "name": run.name,
        "dataset": run.dataset,
        "model_name": run.model_name,
        "baseline_experiment_id": run.baseline_experiment_id,
        "best_experiment_id": run.best_experiment_id,
        "frontier_experiment_id": run.frontier_experiment_id,
        "experiment_count": len(experiment_history),
    }

    system_prompt = (
        "你要为图像分类训练生成下一轮结构化 proposal。"
        "只返回 JSON，不要输出任何额外说明。"
        "必须严格遵循这个 schema："
        '{"task_type":"classification","model_name":"string","based_on_experiment_ids":["string"],'
        '"hypothesis":"string","changes":{"optimizer":"string|null","learning_rate":"number|null",'
        '"batch_size":"number|null","image_size":"number|null","epochs":"number|null","weight_decay":"number|null",'
        '"scheduler":"string|null","augmentation_level":"string|null","label_smoothing":"number|null","aux_logits":"boolean|null"},'
        '"reason":"string","risk":"low|medium|high"}'
        "其中 hypothesis 和 reason 必须使用简洁中文。"
        "你会收到同一个 run 的完整实验历史，而不是只收到最新一轮。"
        "你必须综合所有历史轮次，重点参考 baseline、best、frontier 以及每轮指标变化趋势。"
        "如果某些历史实验已经被标记为 discard、crash、timeout 或 failed，要把它们视为负样本，避免重复无效尝试。"
        "based_on_experiment_ids 必须填写你实际参考的实验 id，可包含多个。"
    )
    user_prompt = (
        f"Run summary:\n{json.dumps(prompt_run_payload, ensure_ascii=True)}\n"
        f"Experiment history:\n{json.dumps(experiment_history, ensure_ascii=True)}\n"
        "请为同一个 run 生成下一轮 proposal。"
        "task_type 必须保持 classification。"
        "不要修改 model_name。"
        "不要修改 epochs；epochs 已固定，AI 不允许调整。"
        "只能提出结构化参数改动。"
        "不要只根据最后一轮实验下结论；必须结合整个 run 的历史记录判断下一步。"
    )
    client = AIHubMixClient()
    proposal_payload = client.create_json_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    proposal = ProposalSchema.model_validate(proposal_payload)
    if proposal.model_name != run.model_name:
        raise ValueError("Proposal model_name does not match the run model")
    proposal = sanitize_blocked_proposal_fields(proposal)
    if not validate_proposal_against_space(proposal):
        raise ValueError("Proposal contains blocked parameter changes")
    return proposal


def test_aihubmix_connection() -> dict:
    """Run a minimal AIHubMix connectivity test."""
    client = AIHubMixClient()
    return client.create_json_completion(
        system_prompt='Return JSON only: {"ok": true, "message": "string"}',
        user_prompt="Reply with a tiny JSON object confirming connectivity.",
    )
