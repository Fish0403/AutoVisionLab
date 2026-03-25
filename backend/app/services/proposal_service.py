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
from app.schemas.parameter_space import EditableParameterSpace, SearchPolicy
from app.services.parameter_space import (
    explain_proposal_rejection,
    get_allowed_ai_search_fields,
)
from app.services.run_logging import append_run_log


def _has_effective_changes(proposal: ProposalSchema) -> bool:
    """Return whether a proposal contains at least one concrete parameter change."""
    return any(value is not None for value in proposal.changes.model_dump().values())


def _contains_unsupported_text_hint(
    proposal: ProposalSchema,
    allowed_field_definitions: dict[str, Any],
) -> str | None:
    """Return one unsupported textual hint if the proposal text references unavailable options."""
    combined_text = f"{proposal.hypothesis} {proposal.reason}".lower()
    unsupported_tokens = {
        "medium": "augmentation_policy",
        "strong": "augmentation_policy",
        "autoaugment": "augmentation_policy",
        "augmentation_level": "augmentation_policy",
    }
    augmentation_definition = allowed_field_definitions.get("augmentation_policy") or {}
    allowed_augmentation_choices = set(augmentation_definition.get("choices") or [])
    for token, field_name in unsupported_tokens.items():
        if token in combined_text and token not in allowed_augmentation_choices:
            return f"text mentions unsupported {field_name} option: {token}"
    return None


def _contains_non_basic_change(proposal: ProposalSchema) -> bool:
    """Return whether the proposal changes include at least one non-basic search field."""
    non_basic_fields = {
        "augmentation_policy",
        "mixup_alpha",
        "cutmix_alpha",
        "random_erasing_prob",
        "loss_name",
        "focal_gamma",
        "aux_logits",
    }
    proposal_changes = proposal.changes.model_dump()
    return any(proposal_changes.get(field_name) is not None for field_name in non_basic_fields)


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


def _load_latest_search_policy(db: Session, run_id: str) -> SearchPolicy:
    """Load the latest experiment search policy for a run."""
    latest_experiment = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.desc())
    ).first()
    if latest_experiment is None:
        return SearchPolicy()
    config_payload = latest_experiment.experiment_config or {}
    return SearchPolicy.model_validate(config_payload.get("search_policy") or {})


def _load_latest_parameter_space(db: Session, run_id: str) -> EditableParameterSpace | None:
    """Load the latest experiment parameter space for a run."""
    latest_experiment = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.desc())
    ).first()
    if latest_experiment is None:
        return None
    return EditableParameterSpace.model_validate(latest_experiment.editable_parameter_space or {})


def sanitize_disallowed_proposal_fields(proposal: ProposalSchema, search_policy: SearchPolicy) -> ProposalSchema:
    """Clear disallowed AI-controlled fields instead of failing the whole proposal."""
    changes_payload = proposal.changes.model_dump()
    allowed_fields = get_allowed_ai_search_fields(search_policy)
    sanitized_changes = {
        key: (value if key in allowed_fields else None)
        for key, value in changes_payload.items()
    }
    return proposal.model_copy(update={"changes": ProposalChanges.model_validate(sanitized_changes)})


def generate_aihubmix_proposal(
    db: Session,
    run_id: str,
    *,
    require_non_basic_change: bool = False,
) -> ProposalSchema:
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
    search_policy = _load_latest_search_policy(db, run_id)
    parameter_space = _load_latest_parameter_space(db, run_id)
    allowed_fields = sorted(get_allowed_ai_search_fields(search_policy))
    image_size_choices: list[int] | None = None
    allowed_field_definitions: dict[str, Any] = {}
    if parameter_space is not None:
        image_size_definition = parameter_space.editable_params.get("image_size")
        image_size_choices = getattr(image_size_definition, "choices", None)
        allowed_field_definitions = {
            field_name: definition.model_dump()
            for field_name, definition in parameter_space.editable_params.items()
            if field_name in allowed_fields
        }

    system_prompt = (
        "你要为图像分类训练生成下一轮结构化 proposal。"
        "只返回 JSON，不要输出任何额外说明。"
        "必须严格遵循这个 schema："
        '{"task_type":"classification","model_name":"string","based_on_experiment_ids":["string"],'
        '"hypothesis":"string","changes":{"optimizer":"string|null","learning_rate":"number|null",'
        '"batch_size":"number|null","image_size":"number|null","epochs":"number|null","weight_decay":"number|null",'
        '"scheduler":"string|null","augmentation_policy":"string|null","mixup_alpha":"number|null",'
        '"cutmix_alpha":"number|null","random_erasing_prob":"number|null","loss_name":"string|null",'
        '"focal_gamma":"number|null","label_smoothing":"number|null","aux_logits":"boolean|null"},'
        '"reason":"string","risk":"low|medium|high"}'
        "其中 hypothesis 和 reason 必须使用简洁中文。"
        "你会收到同一个 run 的完整实验历史，而不是只收到最新一轮。"
        "你必须综合所有历史轮次，重点参考 baseline、best、frontier 以及每轮指标变化趋势。"
        "如果某些历史实验已经被标记为 discard、crash、timeout 或 failed，要把它们视为负样本，避免重复无效尝试。"
        "based_on_experiment_ids 必须填写你实际参考的实验 id，可包含多个。"
        "changes 中至少要有一个字段是非 null；不要返回空 proposal。"
        "hypothesis 和 reason 只能讨论当前参数空间里真实存在的字段和取值，不要臆造 medium、strong、autoaugment 等未开放选项。"
    )
    base_user_prompt = (
        f"Run summary:\n{json.dumps(prompt_run_payload, ensure_ascii=True)}\n"
        f"Experiment history:\n{json.dumps(experiment_history, ensure_ascii=True)}\n"
        f"Allowed AI change fields:\n{json.dumps(allowed_fields, ensure_ascii=True)}\n"
        f"Allowed field definitions:\n{json.dumps(allowed_field_definitions, ensure_ascii=True)}\n"
        f"Allowed image_size choices for this run:\n{json.dumps(image_size_choices, ensure_ascii=True)}\n"
        "请为同一个 run 生成下一轮 proposal。"
        "task_type 必须保持 classification。"
        "不要修改 model_name。"
        "不要修改 epochs；epochs 已固定，AI 不允许调整。"
        "只能修改 Allowed AI change fields 中列出的字段。"
        "每个字段的可选值或范围必须严格遵循 Allowed field definitions。"
        "只能提出结构化参数改动。"
        "不要只根据最后一轮实验下结论；必须结合整个 run 的历史记录判断下一步。"
        "如果最近几轮没有明显提升，必须切换搜索维度，不要重复最近 3 轮几乎相同的建议。"
    )
    if require_non_basic_change:
        base_user_prompt += (
            "当前处于 auto-train 的后半阶段。"
            "本轮 proposal 不能只修改基础超参数；"
            "changes 中必须至少包含一个非基础字段："
            "augmentation_policy、mixup_alpha、cutmix_alpha、random_erasing_prob、loss_name、focal_gamma、aux_logits。"
        )
    client = AIHubMixClient()
    last_error: str | None = None
    for attempt_index in range(2):
        retry_note = ""
        if attempt_index > 0:
            retry_note = (
                "\n上一版 proposal 无效。请务必返回至少一个真实可执行的参数改动，"
                "并且不要在文本里提到未开放的增强等级或策略。"
            )
        proposal_payload = client.create_json_completion(
            system_prompt=system_prompt,
            user_prompt=base_user_prompt + retry_note,
        )
        proposal = ProposalSchema.model_validate(proposal_payload)
        if proposal.model_name != run.model_name:
            last_error = "Proposal model_name does not match the run model"
            continue
        proposal = sanitize_disallowed_proposal_fields(proposal, search_policy)
        if not _has_effective_changes(proposal):
            last_error = "Proposal does not contain any effective parameter changes"
            continue
        if require_non_basic_change and not _contains_non_basic_change(proposal):
            last_error = "Proposal must include at least one non-basic change in the current auto-train phase"
            continue
        unsupported_text_reason = _contains_unsupported_text_hint(proposal, allowed_field_definitions)
        if unsupported_text_reason is not None:
            last_error = unsupported_text_reason
            continue
        rejection_reason = explain_proposal_rejection(proposal, search_policy, parameter_space=parameter_space)
        if rejection_reason is not None:
            last_error = f"Proposal contains blocked parameter changes: {rejection_reason}"
            continue
        append_run_log(
            run_id,
            (
                f"[proposal] based_on={','.join(proposal.based_on_experiment_ids)} | "
                f"hypothesis={proposal.hypothesis} | "
                f"changes={json.dumps(proposal.changes.model_dump(exclude_none=True), ensure_ascii=False)} | "
                f"reason={proposal.reason}"
            ),
        )
        return proposal
    raise ValueError(last_error or "Proposal generation failed validation")


def test_aihubmix_connection() -> dict:
    """Run a minimal AIHubMix connectivity test."""
    client = AIHubMixClient()
    return client.create_json_completion(
        system_prompt='Return JSON only: {"ok": true, "message": "string"}',
        user_prompt="Reply with a tiny JSON object confirming connectivity.",
    )
