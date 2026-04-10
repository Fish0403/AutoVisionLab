"""Proposal generation services."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.history_context import build_proposal_history_context
from app.context.prompt_builder import build_proposal_prompt_bundle
from app.context.proposal_policy import build_policy_prompt_payload
from app.llm.aihubmix_client import AIHubMixClient, AIHubMixRequestError
from app.models.experiment import ExperimentModel
from app.models.run import RunModel
from app.prompts.context_blocks import build_epoch_policy_instruction, build_retry_note
from app.schemas.ai import ProposalChanges, ProposalSchema
from app.schemas.prompt_context import PromptBlock
from app.schemas.parameter_space import EditableParameterSpace, SearchPolicy
from app.services.parameter_space import (
    explain_proposal_rejection,
    get_allowed_ai_search_fields,
    is_epoch_search_enabled,
)
from app.services.log_events import format_run_log_message
from app.services.proposal_context_cache import (
    RunProposalContextCacheEntry,
    get_run_proposal_context_cache_entry,
    set_run_proposal_context_cache_entry,
)
from app.services.run_logging import (
    append_run_llm_event,
    append_run_log,
    append_run_prompt_context_event,
    append_run_prompt_markdown_event,
)


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
        r"\baugmentation_level\b": "augmentation_level",
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?medium\b": "medium",
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?strong\b": "strong",
        r"\b(?:augmentation[_\s-]*policy|policy)\s*(?:=|:)?\s*[\"']?autoaugment\b": "autoaugment",
    }
    for pattern, token_override in unsupported_patterns.items():
        match = re.search(pattern, combined_text)
        if match is None:
            continue
        token = token_override or match.group(1)
        return f"text mentions unsupported augmentation strategy: {token}"
    return None


def _serialize_prompt_blocks(blocks: list[PromptBlock]) -> list[dict[str, Any]]:
    """Return prompt blocks in a log-friendly format."""
    serialized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        payload = block.payload.model_dump(mode="python") if hasattr(block.payload, "model_dump") else block.payload
        serialized_blocks.append(
            {
                "name": block.name,
                "role": block.role,
                "render_priority": block.render_priority,
                "payload": payload,
            }
        )
    return serialized_blocks


def _build_run_history_signature(run: RunModel) -> tuple[str | None, str | None, str]:
    """Return one cache signature for run-scoped proposal history."""
    updated_at = run.updated_at.isoformat() if getattr(run, "updated_at", None) else ""
    return (
        run.baseline_experiment_id,
        run.best_experiment_id,
        updated_at,
    )


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
        "is_best_so_far": bool(experiment.is_best_so_far),
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "metrics": {
            "top1_acc": metrics_payload.get("top1_acc"),
            "val_loss": metrics_payload.get("val_loss"),
            "train_loss": metrics_payload.get("train_loss"),
            "best_epoch": metrics_payload.get("best_epoch"),
        },
        "resource": {
            "training_seconds": resource_payload.get("training_seconds"),
            "gpu_memory_mb": resource_payload.get("gpu_memory_mb"),
            "latency_ms": resource_payload.get("latency_ms"),
            "parameter_count_million": resource_payload.get("parameter_count_million"),
        },
        "result": {
            "status": result_payload.get("status") or experiment.status,
            "metrics": {
                "top1_acc": metrics_payload.get("top1_acc"),
                "val_loss": metrics_payload.get("val_loss"),
                "train_loss": metrics_payload.get("train_loss"),
                "best_epoch": metrics_payload.get("best_epoch"),
            },
            "resource": {
                "training_seconds": resource_payload.get("training_seconds"),
                "gpu_memory_mb": resource_payload.get("gpu_memory_mb"),
                "latency_ms": resource_payload.get("latency_ms"),
                "parameter_count_million": resource_payload.get("parameter_count_million"),
            },
        },
        "config": {
            "params": params_payload,
            "train_hyp": train_hyp_payload,
            "model_recipe": model_recipe_payload,
        },
        "params": params_payload,
        "train_hyp": train_hyp_payload,
        "model_recipe": model_recipe_payload,
        "proposal": {
            "based_on_experiment_ids": proposal_payload.get("based_on_experiment_ids"),
            "hypothesis": proposal_payload.get("hypothesis"),
            "changes": (proposal_payload.get("changes") or {}),
            "reason": proposal_payload.get("reason"),
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


def _parse_payload_created_at(experiment_payload: dict[str, Any]) -> datetime | None:
    created_at = experiment_payload.get("created_at")
    if isinstance(created_at, datetime):
        return created_at
    if isinstance(created_at, str):
        return datetime.fromisoformat(created_at)
    return None


def _build_prompt_run_payload(run: RunModel, *, experiment_count: int) -> dict[str, Any]:
    """Build the run payload passed into prompt assembly."""
    return {
        "id": run.id,
        "name": run.name,
        "dataset": run.dataset,
        "model_name": run.model_name,
        "best_experiment_id": run.best_experiment_id,
        "experiment_count": experiment_count,
    }


def _build_run_context_cache_entry_from_history(
    *,
    run: RunModel,
    history_signature: tuple[str | None, str | None, str],
    experiment_history: list[dict[str, Any]],
) -> RunProposalContextCacheEntry:
    """Build one run context state by replaying the persisted experiment history."""
    if not experiment_history:
        raise ValueError("No experiment is available for this run")
    prompt_run_payload = _build_prompt_run_payload(run, experiment_count=len(experiment_history))
    history_context = build_proposal_history_context(
        run_payload=prompt_run_payload,
        experiment_history=experiment_history,
    )
    source_experiment_id = str(history_context.source_experiment_payload.get("id"))
    source_index = next(
        index
        for index, experiment_payload in enumerate(experiment_history)
        if str(experiment_payload.get("id")) == source_experiment_id
    )
    latest_experiment_payload = experiment_history[-1]
    entry = RunProposalContextCacheEntry(
        run_id=run.id,
        history_signature=history_signature,
        run_payload=prompt_run_payload,
        history_item_count=source_index + 1,
        base_experiment_payload=history_context.base_experiment_payload,
        source_experiment_payload=history_context.source_experiment_payload,
        last_processed_experiment_id=str(latest_experiment_payload.get("id")),
        last_processed_experiment_created_at=_parse_payload_created_at(latest_experiment_payload),
        past_stage_summaries=list(history_context.past_stage_summaries),
    )
    for experiment_payload in experiment_history[source_index + 1 :]:
        entry.append_experiment_payload(experiment_payload)
    return entry


def _load_incremental_run_history_payload(
    db: Session,
    run: RunModel,
    *,
    last_processed_experiment_id: str,
    last_processed_experiment_created_at: datetime | None,
) -> list[dict[str, Any]] | None:
    """Return only experiments created after the cached run state marker."""
    if last_processed_experiment_created_at is None:
        return None
    experiments = db.scalars(
        select(ExperimentModel)
        .where(
            ExperimentModel.run_id == run.id,
            ExperimentModel.created_at >= last_processed_experiment_created_at,
        )
        .order_by(ExperimentModel.created_at.asc())
    ).all()
    experiment_payloads = [_summarize_experiment_for_prompt(experiment, run) for experiment in experiments]
    marker_index = next(
        (
            index
            for index, experiment_payload in enumerate(experiment_payloads)
            if str(experiment_payload.get("id")) == last_processed_experiment_id
        ),
        None,
    )
    if marker_index is None:
        return None
    return experiment_payloads[marker_index + 1 :]


def _load_or_refresh_run_context_cache_entry(
    db: Session,
    run: RunModel,
) -> tuple[RunProposalContextCacheEntry, str]:
    """Return one run state from cache, or rebuild/advance it when needed."""
    history_signature = _build_run_history_signature(run)
    cached_entry = get_run_proposal_context_cache_entry(run.id)
    if cached_entry is None:
        experiment_history = get_run_history_payload(db, run.id)
        rebuilt_entry = _build_run_context_cache_entry_from_history(
            run=run,
            history_signature=history_signature,
            experiment_history=experiment_history,
        )
        set_run_proposal_context_cache_entry(rebuilt_entry)
        return rebuilt_entry, "miss"
    cached_entry.refresh_run_payload(best_experiment_id=run.best_experiment_id)
    if cached_entry.history_signature == history_signature:
        return cached_entry, "hit"
    incremental_history = _load_incremental_run_history_payload(
        db,
        run,
        last_processed_experiment_id=cached_entry.last_processed_experiment_id,
        last_processed_experiment_created_at=cached_entry.last_processed_experiment_created_at,
    )
    if incremental_history is None:
        experiment_history = get_run_history_payload(db, run.id)
        rebuilt_entry = _build_run_context_cache_entry_from_history(
            run=run,
            history_signature=history_signature,
            experiment_history=experiment_history,
        )
        set_run_proposal_context_cache_entry(rebuilt_entry)
        return rebuilt_entry, "rebuild"
    if incremental_history:
        for experiment_payload in incremental_history:
            cached_entry.append_experiment_payload(experiment_payload)
        if run.best_experiment_id is not None and str(cached_entry.source_experiment_payload.get("id")) != str(run.best_experiment_id):
            experiment_history = get_run_history_payload(db, run.id)
            rebuilt_entry = _build_run_context_cache_entry_from_history(
                run=run,
                history_signature=history_signature,
                experiment_history=experiment_history,
            )
            set_run_proposal_context_cache_entry(rebuilt_entry)
            return rebuilt_entry, "rebuild"
        cached_entry.refresh_run_payload(best_experiment_id=run.best_experiment_id)
        cached_entry.history_signature = history_signature
        set_run_proposal_context_cache_entry(cached_entry)
        return cached_entry, "advance"
    cached_entry.history_signature = history_signature
    set_run_proposal_context_cache_entry(cached_entry)
    return cached_entry, "refresh"


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

    context_cache_entry, cache_status = _load_or_refresh_run_context_cache_entry(db, run)
    cache_summary = context_cache_entry.summarize_cache_state()
    append_run_log(
        run_id,
        format_run_log_message(
            level="INFO",
            section="proposal-cache",
            message=f"cache {cache_status}",
            history_items=cache_summary["history_items"],
            recent_history_items=cache_summary["recent_history_items"],
            compacted_bucket_count=cache_summary["compacted_bucket_count"],
            compacted_history_items=cache_summary["compacted_history_items"],
            past_stage_summary_count=cache_summary["past_stage_summary_count"],
            source_experiment_id=cache_summary["source_experiment_id"],
        ),
    )
    source_constraints = context_cache_entry.build_source_constraints()
    parameter_space = _load_latest_parameter_space(db, run_id)
    search_policy = _load_latest_search_policy(db, run_id)
    allowed_fields = sorted(get_allowed_ai_search_fields(search_policy, parameter_space=parameter_space))
    if not allowed_fields:
        raise ValueError("No AI-editable fields are available for this run.")
    allowed_field_definitions: dict[str, Any] = {}
    if parameter_space is not None:
        allowed_field_definitions = {
            field_name: definition.model_dump()
            for field_name, definition in parameter_space.editable_params.items()
            if field_name in allowed_fields
        }
    policy_payload = build_policy_prompt_payload(
        allowed_field_definitions=allowed_field_definitions,
        epoch_policy_instruction=build_epoch_policy_instruction(is_epoch_search_enabled(search_policy)),
        require_non_basic_change=require_non_basic_change,
        max_changed_fields=max_changed_fields,
    )
    client = AIHubMixClient()
    last_error: str | None = None
    for attempt_index in range(4):
        effective_retry_feedback = retry_feedback
        if attempt_index > 0:
            retry_note = build_retry_note(last_error)
            effective_retry_feedback = (
                f"{retry_feedback}\n\n{retry_note}" if retry_feedback else retry_note
            )
        prompt_bundle = build_proposal_prompt_bundle(
            run_payload=context_cache_entry.run_payload,
            policy_payload=policy_payload,
            source_constraints=source_constraints,
            history_context=context_cache_entry.to_history_context(),
            retry_feedback=effective_retry_feedback,
        )
        prompt_metadata = {
            "attempt": attempt_index + 1,
            "history_items": context_cache_entry.history_item_count,
            "prompt_chars": prompt_bundle.prompt_chars,
            "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
        }
        if on_prompt_metadata is not None:
            on_prompt_metadata(prompt_metadata)
        append_run_log(
            run_id,
            format_run_log_message(
                level="INFO",
                section="proposal-meta",
                message="prompt prepared",
                attempt=attempt_index + 1,
                history_items=context_cache_entry.history_item_count,
                prompt_chars=prompt_bundle.prompt_chars,
                prompt_tokens_estimate=prompt_bundle.prompt_tokens_estimate,
            ),
        )
        append_run_llm_event(
            run_id,
            "proposal_request",
            {
                "attempt": attempt_index + 1,
                "history_items": context_cache_entry.history_item_count,
                "prompt_chars": prompt_bundle.prompt_chars,
                "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
                "system_prompt": prompt_bundle.system_prompt,
                "user_prompt": prompt_bundle.user_prompt,
            },
        )
        append_run_prompt_markdown_event(
            run_id,
            "proposal_request",
            {
                "attempt": attempt_index + 1,
                "history_items": context_cache_entry.history_item_count,
                "prompt_chars": prompt_bundle.prompt_chars,
                "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
                "system_prompt": prompt_bundle.system_prompt,
                "user_prompt": prompt_bundle.user_prompt,
            },
        )
        append_run_prompt_context_event(
            run_id,
            "proposal_prompt_context",
            {
                "attempt": attempt_index + 1,
                "history_items": context_cache_entry.history_item_count,
                "prompt_chars": prompt_bundle.prompt_chars,
                "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
                "system_prompt_chars": len(prompt_bundle.system_prompt),
                "system_prompt_tokens_estimate": _estimate_text_tokens(prompt_bundle.system_prompt),
                "blocks": _serialize_prompt_blocks(prompt_bundle.blocks),
            },
        )
        try:
            proposal_payload, provider_metadata = client.create_json_completion_with_metadata(
                system_prompt=prompt_bundle.system_prompt,
                user_prompt=prompt_bundle.user_prompt,
            )
        except Exception as error:
            raw_content = error.raw_content if isinstance(error, AIHubMixRequestError) else None
            response_model = error.response_model if isinstance(error, AIHubMixRequestError) else None
            response_chars = error.response_chars if isinstance(error, AIHubMixRequestError) else None
            usage = error.usage if isinstance(error, AIHubMixRequestError) else None
            append_run_log(
                run_id,
                format_run_log_message(
                    level="ERROR",
                    section="proposal-meta",
                    message="proposal request failed",
                    attempt=attempt_index + 1,
                    prompt_tokens_estimate=prompt_bundle.prompt_tokens_estimate,
                    error=str(error),
                ),
            )
            append_run_llm_event(
                run_id,
                "proposal_error",
                {
                    "attempt": attempt_index + 1,
                    "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
                    "usage": usage,
                    "response_model": response_model,
                    "response_chars": response_chars,
                    "raw_content": raw_content,
                    "error": str(error),
                },
            )
            append_run_prompt_markdown_event(
                run_id,
                "proposal_error",
                {
                    "attempt": attempt_index + 1,
                    "prompt_tokens_estimate": prompt_bundle.prompt_tokens_estimate,
                    "usage": usage,
                    "response_model": response_model,
                    "response_chars": response_chars,
                    "raw_content": raw_content,
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
        append_run_prompt_markdown_event(
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
            format_run_log_message(
                level="INFO",
                section="proposal",
                message="proposal accepted",
                based_on=proposal.based_on_experiment_ids,
                changed_fields=sorted(_get_effective_change_map(proposal).keys()),
                prompt_tokens_estimate=prompt_bundle.prompt_tokens_estimate,
                provider_usage=provider_metadata.get("usage"),
                hypothesis=proposal.hypothesis,
                changes=proposal.changes.model_dump(exclude_none=True),
                reason=proposal.reason,
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
