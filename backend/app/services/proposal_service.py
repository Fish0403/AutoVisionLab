"""Proposal generation services."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

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
    is_epoch_search_enabled,
)
from app.services.run_logging import append_run_llm_event, append_run_log


def _get_effective_change_map(proposal: ProposalSchema) -> dict[str, Any]:
    """Return only the concrete changed fields in the proposal."""
    return {
        field_name: value
        for field_name, value in proposal.changes.model_dump().items()
        if value is not None
    }


def _count_effective_changes(proposal: ProposalSchema) -> int:
    """Return the number of concrete parameter changes in the proposal."""
    return len(_get_effective_change_map(proposal))


def _estimate_text_tokens(text: str) -> int:
    """Return a rough token estimate for mixed JSON, English, and Chinese text."""
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4 + non_ascii_chars / 1.8))


def _contains_unsupported_text_hint(
    proposal: ProposalSchema,
    allowed_field_definitions: dict[str, Any],
) -> str | None:
    """Return one unsupported textual hint if the proposal text references unavailable options."""
    combined_text = f"{proposal.hypothesis} {proposal.reason}".lower()
    unsupported_patterns = {
        r"\baugmentation_level\b": ("augmentation_policy", "augmentation_level"),
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?medium\b": (
            "augmentation_policy",
            "medium",
        ),
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?strong\b": (
            "augmentation_policy",
            "strong",
        ),
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?autoaugment\b": (
            "augmentation_policy",
            "autoaugment",
        ),
        r"\"augmentation_policy\"\s*:\s*\"(medium|strong|autoaugment)\"": (
            "augmentation_policy",
            None,
        ),
    }
    augmentation_definition = allowed_field_definitions.get("augmentation_policy") or {}
    allowed_augmentation_choices = set(augmentation_definition.get("choices") or [])
    for pattern, (field_name, token_override) in unsupported_patterns.items():
        match = re.search(pattern, combined_text)
        if match is None:
            continue
        token = token_override or match.group(1)
        if token not in allowed_augmentation_choices:
            return f"text mentions unsupported {field_name} option: {token}"
    return None


def _build_retry_note(
    *,
    last_error: str | None,
) -> str:
    """Build targeted retry guidance after one invalid proposal."""
    retry_lines = [
        "",
        "The previous proposal was invalid. Fix the issues below before returning a new complete JSON object.",
    ]
    if last_error:
        retry_lines.append(f"Previous rejection reason: {last_error}.")
    retry_lines.append("Choose a more executable direction based on the full history and do not repeat the invalid plan.")
    return "\n".join(retry_lines)


def _summarize_experiment_for_prompt(experiment: ExperimentModel, run: RunModel) -> dict[str, Any]:
    """Build a compact experiment summary for proposal prompting."""
    result_payload = experiment.result or {}
    metrics_payload = result_payload.get("metrics") or {}
    resource_payload = result_payload.get("resource") or {}
    proposal_payload = experiment.proposal or {}
    config_payload = experiment.experiment_config or {}
    params_payload = config_payload.get("params") or {}
    train_hyp_payload = config_payload.get("train_hyp") or {}
    model_recipe_payload = config_payload.get("model_recipe") or {}

    return {
        "id": experiment.id,
        "status": experiment.status,
        "decision": experiment.decision,
        "decision_reason": experiment.decision_reason,
        "is_best": experiment.id == run.best_experiment_id,
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
        "train_hyp": train_hyp_payload,
        "model_recipe": model_recipe_payload,
        "proposal": {
            "based_on_experiment_ids": proposal_payload.get("based_on_experiment_ids"),
            "hypothesis": proposal_payload.get("hypothesis"),
            "changes": (proposal_payload.get("changes") or {}),
            "train_hyp_changes": proposal_payload.get("train_hyp_changes"),
            "recipe_changes": proposal_payload.get("recipe_changes"),
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


def _load_latest_search_policy(
    db: Session,
    run_id: str,
) -> SearchPolicy:
    """Load the latest experiment search policy for a run."""
    latest_experiment = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.desc())
    ).first()
    if latest_experiment is None:
        return SearchPolicy()
    config_payload = latest_experiment.experiment_config or {}
    raw_search_policy = config_payload.get("search_policy")
    if raw_search_policy is None:
        return SearchPolicy()
    return SearchPolicy.model_validate(raw_search_policy)


def _load_latest_parameter_space(db: Session, run_id: str) -> EditableParameterSpace | None:
    """Load the latest experiment parameter space for a run."""
    latest_experiment = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.desc())
    ).first()
    if latest_experiment is None:
        return None
    return EditableParameterSpace.model_validate(latest_experiment.editable_parameter_space or {})


def _load_followup_source_constraints(db: Session, run: RunModel) -> dict[str, Any]:
    """Return the current source experiment metadata used to branch the next proposal."""
    source_experiment: ExperimentModel | None = None
    if run.best_experiment_id:
        source_experiment = db.get(ExperimentModel, run.best_experiment_id)
    if source_experiment is None:
        source_experiment = db.scalars(
            select(ExperimentModel).where(ExperimentModel.run_id == run.id).order_by(ExperimentModel.created_at.desc())
        ).first()
    if source_experiment is None:
        return {
            "experiment_id": None,
            "image_size": None,
        }
    config_payload = source_experiment.experiment_config or {}
    train_hyp_payload = config_payload.get("train_hyp") or {}
    source_image_size = train_hyp_payload.get("image_size")
    return {
        "experiment_id": source_experiment.id,
        "image_size": source_image_size,
    }


def sanitize_disallowed_proposal_fields(
    proposal: ProposalSchema,
    search_policy: SearchPolicy,
    *,
    parameter_space: EditableParameterSpace | None = None,
) -> ProposalSchema:
    """Clear disallowed AI-controlled fields instead of failing the whole proposal."""
    changes_payload = proposal.changes.model_dump()
    allowed_fields = get_allowed_ai_search_fields(search_policy, parameter_space=parameter_space)
    sanitized_changes = {
        key: (value if key in allowed_fields else None)
        for key, value in changes_payload.items()
    }
    return proposal.model_copy(update={"changes": ProposalChanges.model_validate(sanitized_changes)})


def _build_epoch_policy_instruction(search_policy: SearchPolicy) -> str:
    """Return one prompt instruction that matches the current epoch-search policy."""
    if is_epoch_search_enabled(search_policy):
        return (
            "You may change epochs when it is helpful, but treat epochs as training budget rather than a pure strategy field. "
            "If you change epochs, make sure the hypothesis and reason describe that budget tradeoff clearly. "
        )
    return "Do not change epochs; epochs is fixed and the AI is not allowed to adjust it. "


def generate_aihubmix_proposal(
    db: Session,
    run_id: str,
    *,
    require_non_basic_change: bool = False,
    max_changed_fields: int | None = None,
    retry_feedback: str | None = None,
    on_prompt_metadata: Callable[[dict[str, Any]], None] | None = None,
    on_provider_metadata: Callable[[dict[str, Any]], None] | None = None,
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
        "best_experiment_id": run.best_experiment_id,
        "experiment_count": len(experiment_history),
    }
    source_constraints = _load_followup_source_constraints(db, run)
    parameter_space = _load_latest_parameter_space(db, run_id)
    search_policy = _load_latest_search_policy(db, run_id)
    allowed_fields = sorted(get_allowed_ai_search_fields(search_policy, parameter_space=parameter_space))
    if not allowed_fields:
        raise ValueError("No AI-editable fields are available for this run.")
    image_size_definition: dict[str, Any] | None = None
    allowed_field_definitions: dict[str, Any] = {}
    if parameter_space is not None:
        image_size_param_definition = parameter_space.editable_params.get("image_size")
        if image_size_param_definition is not None:
            image_size_definition = image_size_param_definition.model_dump()
        allowed_field_definitions = {
            field_name: definition.model_dump()
            for field_name, definition in parameter_space.editable_params.items()
            if field_name in allowed_fields
        }

    system_prompt = (
        "Generate the next structured proposal for an image classification training run. "
        "Return JSON only with no extra text. "
        "You must strictly follow this schema:"
        '{"task_type":"classification","model_name":"string","based_on_experiment_ids":["string"],'
        '"hypothesis":"string","changes":{"optimizer":"string|null","learning_rate":"number|null",'
        '"batch_size":"number|null","image_size":"number|null","epochs":"number|null","weight_decay":"number|null",'
        '"scheduler":"string|null","augmentation_policy":"string|null","mixup_alpha":"number|null",'
        '"cutmix_alpha":"number|null","random_erasing_prob":"number|null","loss_name":"string|null",'
        '"focal_gamma":"number|null","label_smoothing":"number|null","aux_logits":"boolean|null",'
        '"neck_name":"string|null","head_name":"string|null"},'
        '"train_hyp_changes":"object|null","recipe_changes":"object|null",'
        '"reason":"string","risk":"low|medium|high"}'
        "hypothesis and reason must be concise English. "
        "changes is the required compatibility-layer change map; if you can map changes clearly into recipe-oriented views, also return train_hyp_changes or recipe_changes. "
        "You will receive the full experiment history for the same run, not only the latest round. "
        "You must use the full history, focusing on the current best result and metric trends across rounds. "
        "If past experiments were marked discard, crash, timeout, or failed, treat them as negative examples and avoid repeating ineffective directions. "
        "based_on_experiment_ids must list the experiment ids you actually used as evidence and may contain multiple ids. "
        "At least one field in changes must be non-null; never return an empty proposal. "
        "You may change one or multiple fields, but every field and value must come strictly from the current parameter space. "
        "hypothesis and reason may only discuss fields and values that truly exist in the current parameter space. "
        "When discussing augmentation, name the concrete fields and legal values directly, such as augmentation_policy=none/basic, "
        "mixup_alpha, cutmix_alpha, and random_erasing_prob. Do not invent extra augmentation preset names."
    )
    retry_feedback_prompt = (
        f"Previous full-proposal rejection:\n{json.dumps(retry_feedback, ensure_ascii=True)}\n"
        if retry_feedback
        else ""
    )
    base_user_prompt = (
        f"Run summary:\n{json.dumps(prompt_run_payload, ensure_ascii=True)}\n"
        f"Experiment history:\n{json.dumps(experiment_history, ensure_ascii=True)}\n"
        f"Allowed AI change fields:\n{json.dumps(allowed_fields, ensure_ascii=True)}\n"
        f"Allowed field definitions:\n{json.dumps(allowed_field_definitions, ensure_ascii=True)}\n"
        f"image_size parameter definition:\n{json.dumps(image_size_definition, ensure_ascii=True)}\n"
        f"Current source experiment constraints:\n{json.dumps(source_constraints, ensure_ascii=True)}\n"
        f"{retry_feedback_prompt}"
        "Generate the next proposal for the same run. "
        "task_type must remain classification. "
        "Do not change model_name. "
        f"{_build_epoch_policy_instruction(search_policy)}"
        "Only modify fields listed in Allowed AI change fields. "
        "Every value must strictly follow Allowed field definitions. "
        "Only propose structured parameter changes. "
        "If the current parameter space enables component-level search, prefer neck_name and head_name over old fine-grained recipe fields. "
        "Do not decide only from the last round; use the full run history and keep optimizing around the current best by default. "
        "If you change image_size, it must stay a positive integer within the allowed parameter space. "
        "Prefer a smaller value than the current source experiment image_size when that keeps the next step more efficient, but this is a search preference rather than a hard rule. "
        "You may choose the next search direction freely, but do not mechanically repeat nearly identical suggestions from the most recent rounds."
    )
    client = AIHubMixClient()
    last_error: str | None = None
    for attempt_index in range(4):
        retry_note = ""
        if attempt_index > 0:
            retry_note = _build_retry_note(
                last_error=last_error,
            )
        effective_user_prompt = base_user_prompt + retry_note
        prompt_chars = len(system_prompt) + len(effective_user_prompt)
        prompt_tokens_estimate = _estimate_text_tokens(system_prompt + effective_user_prompt)
        prompt_metadata = {
            "attempt": attempt_index + 1,
            "history_items": len(experiment_history),
            "prompt_chars": prompt_chars,
            "prompt_tokens_estimate": prompt_tokens_estimate,
        }
        if on_prompt_metadata is not None:
            on_prompt_metadata(prompt_metadata)
        append_run_log(
            run_id,
            (
                f"[proposal-meta] attempt={attempt_index + 1} | "
                f"history_items={len(experiment_history)} | "
                f"prompt_chars={prompt_chars} | "
                f"prompt_tokens_estimate={prompt_tokens_estimate}"
            ),
        )
        append_run_llm_event(
            run_id,
            "proposal_request",
            {
                "attempt": attempt_index + 1,
                "history_items": len(experiment_history),
                "prompt_chars": prompt_chars,
                "prompt_tokens_estimate": prompt_tokens_estimate,
                "system_prompt": system_prompt,
                "user_prompt": effective_user_prompt,
            },
        )
        try:
            proposal_payload, provider_metadata = client.create_json_completion_with_metadata(
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
            )
        except Exception as error:
            append_run_log(
                run_id,
                (
                    f"[proposal-meta] attempt={attempt_index + 1} failed | "
                    f"prompt_tokens_estimate={prompt_tokens_estimate} | "
                    f"error={error}"
                ),
            )
            append_run_llm_event(
                run_id,
                "proposal_error",
                {
                    "attempt": attempt_index + 1,
                    "prompt_tokens_estimate": prompt_tokens_estimate,
                    "error": str(error),
                },
            )
            raise
        if on_provider_metadata is not None:
            on_provider_metadata(provider_metadata)
        append_run_llm_event(
            run_id,
            "proposal_response",
            {
                "attempt": attempt_index + 1,
                "usage": provider_metadata.get("usage"),
                "response_model": provider_metadata.get("response_model"),
                "response_chars": provider_metadata.get("response_chars"),
                "raw_content": provider_metadata.get("raw_content"),
                "parsed_payload": proposal_payload,
            },
        )
        proposal = ProposalSchema.model_validate(proposal_payload)
        if proposal.model_name != run.model_name:
            last_error = "Proposal model_name does not match the run model"
            continue
        proposal = sanitize_disallowed_proposal_fields(
            proposal,
            search_policy,
            parameter_space=parameter_space,
        )
        effective_change_count = _count_effective_changes(proposal)
        if effective_change_count == 0:
            last_error = "Proposal does not contain any effective parameter changes"
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
                f"changed_fields={json.dumps(sorted(_get_effective_change_map(proposal).keys()), ensure_ascii=False)} | "
                f"prompt_tokens_estimate={prompt_tokens_estimate} | "
                f"provider_usage={json.dumps(provider_metadata.get('usage'), ensure_ascii=False)} | "
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
