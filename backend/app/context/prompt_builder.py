"""Build proposal prompt bundles from structured run history."""

from __future__ import annotations

from typing import Any

from app.context.context_mapper import build_base_prompt_payload, build_retry_prompt_payload, build_source_prompt_payload
from app.context.history_context import build_proposal_history_context
from app.context.types import DEFAULT_STAGE_COMPACT_THRESHOLD, DEFAULT_STAGE_HISTORY_KEEP
from app.prompts.context_blocks import (
    PROPOSAL_BLOCK_ORDER,
    build_output_schema_block,
    build_payload_block_text,
    build_policy_block_text,
    build_proposal_system_prompt,
    get_prompt_block_title,
)
from app.schemas.prompt_context import PromptBlock, PromptBundle, ProposalHistoryContext


def _estimate_text_tokens(text: str) -> int:
    """Return a rough token estimate for mixed JSON, English, and Chinese text."""
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    estimated_tokens = ascii_chars / 4 + non_ascii_chars / 1.8
    return max(1, int(estimated_tokens + 0.999))


def _render_user_prompt(blocks: list[PromptBlock]) -> str:
    block_texts: list[str] = []
    for block in sorted(blocks, key=lambda item: item.render_priority):
        if block.name == "system_prompt":
            continue
        if block.name == "output_schema_prompt":
            block_texts.append(build_output_schema_block())
            continue
        if block.name == "policy_prompt":
            block_texts.append(build_policy_block_text(block.payload))
            continue
        if block.name == "retry_prompt":
            payload = block.payload.model_dump(mode="python") if hasattr(block.payload, "model_dump") else block.payload
            block_texts.append(build_payload_block_text(get_prompt_block_title(block.name), payload))
            continue
        title = get_prompt_block_title(block.name)
        payload = block.payload.model_dump(mode="python") if hasattr(block.payload, "model_dump") else block.payload
        block_texts.append(build_payload_block_text(title, payload))
    return "\n\n".join(block_texts)


def build_proposal_prompt_bundle(
    *,
    run_payload: dict[str, Any],
    experiment_history: list[dict[str, Any]] | None = None,
    policy_payload: dict[str, Any],
    source_constraints: dict[str, Any],
    history_context: ProposalHistoryContext | None = None,
    retry_feedback: str | None = None,
    stage_history_keep: int = DEFAULT_STAGE_HISTORY_KEEP,
    stage_compact_threshold: int = DEFAULT_STAGE_COMPACT_THRESHOLD,
) -> PromptBundle:
    """Build one proposal prompt bundle from ordered run history."""
    if history_context is None:
        if not experiment_history:
            raise ValueError("experiment_history must not be empty when history_context is missing")
        effective_history_context = build_proposal_history_context(
            run_payload=run_payload,
            experiment_history=experiment_history,
            stage_history_keep=stage_history_keep,
            stage_compact_threshold=stage_compact_threshold,
        )
    else:
        effective_history_context = history_context
    source_experiment_id = str(effective_history_context.source_experiment_payload.get("id"))
    base_experiment_id = str(effective_history_context.base_experiment_payload.get("id"))
    blocks: list[PromptBlock] = [
        PromptBlock(
            name="system_prompt",
            role="system",
            payload=build_proposal_system_prompt(
                dataset_name=run_payload.get("dataset"),
                model_name=run_payload.get("model_name"),
            ),
            render_priority=0,
        ),
        PromptBlock(name="output_schema_prompt", role="user", payload={"mode": "proposal"}, render_priority=1),
        PromptBlock(name="policy_prompt", role="user", payload=policy_payload, render_priority=2),
        PromptBlock(
            name="base_prompt",
            role="user",
            payload=build_base_prompt_payload(effective_history_context.base_experiment_payload),
            render_priority=3,
        ),
    ]
    if source_experiment_id != base_experiment_id:
        blocks.append(
            PromptBlock(
                name="source_prompt",
                role="user",
                payload=build_source_prompt_payload(
                    base_experiment_payload=effective_history_context.base_experiment_payload,
                    source_experiment_payload=effective_history_context.source_experiment_payload,
                    is_best=source_experiment_id == str(run_payload.get("best_experiment_id")),
                ),
                render_priority=4,
            )
        )
    if effective_history_context.recent_stage_history:
        blocks.append(
            PromptBlock(
                name="stage_history_prompt",
                role="user",
                payload={"items": [item.model_dump(mode="python") for item in effective_history_context.recent_stage_history]},
                render_priority=6,
            )
        )
    if effective_history_context.current_stage_compacted_summary is not None:
        blocks.append(
            PromptBlock(
                name="current_stage_compacted_prompt",
                role="user",
                payload=effective_history_context.current_stage_compacted_summary,
                render_priority=7,
            )
        )
    if effective_history_context.past_stage_summaries:
        blocks.append(
            PromptBlock(
                name="past_stage_summaries_prompt",
                role="user",
                payload={"stages": [stage.model_dump(mode="python") for stage in effective_history_context.past_stage_summaries]},
                render_priority=8,
            )
        )
    if retry_feedback:
        blocks.append(
            PromptBlock(
                name="retry_prompt",
                role="user",
                payload=build_retry_prompt_payload(retry_feedback),
                render_priority=9,
            )
        )
    block_order = {name: index for index, name in enumerate(PROPOSAL_BLOCK_ORDER)}
    blocks = sorted(blocks, key=lambda block: block_order[block.name])
    system_prompt = next(block.payload for block in blocks if block.name == "system_prompt")
    user_prompt = _render_user_prompt(blocks)
    prompt_chars = len(system_prompt) + len(user_prompt)
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        blocks=blocks,
        prompt_chars=prompt_chars,
        prompt_tokens_estimate=_estimate_text_tokens(system_prompt + user_prompt),
    )
