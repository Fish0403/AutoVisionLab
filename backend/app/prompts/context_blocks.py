"""Prompt templates for proposal-context blocks."""

from __future__ import annotations

import json
from typing import Any


PROPOSAL_BLOCK_ORDER = (
    "system_prompt",
    "output_schema_prompt",
    "policy_prompt",
    "base_prompt",
    "source_prompt",
    "stage_history_prompt",
    "current_stage_compacted_prompt",
    "past_stage_summaries_prompt",
    "retry_prompt",
)

PROPOSAL_BLOCK_TITLES = {
    "output_schema_prompt": "输出结构",
    "policy_prompt": "策略上下文",
    "base_prompt": "Baseline 上下文",
    "source_prompt": "当前 Source 上下文",
    "stage_history_prompt": "当前阶段历史",
    "current_stage_compacted_prompt": "当前阶段压缩历史",
    "past_stage_summaries_prompt": "历史阶段摘要",
    "retry_prompt": "上一条完整 Proposal 的拒绝反馈",
}


def build_proposal_system_prompt(*, dataset_name: str | None = None, model_name: str | None = None) -> str:
    """Return the stable system rules for proposal generation."""
    background_parts: list[str] = []
    if dataset_name:
        background_parts.append(f"dataset={dataset_name}")
    if model_name:
        background_parts.append(f"model={model_name}")
    background_text = ""
    if background_parts:
        background_text = f"当前 run 背景：{', '.join(background_parts)}。"
    return (
        "请为一个图像分类训练 run 生成下一条结构化 proposal。"
        f"{background_text}"
        "只返回 JSON，不要附加任何额外文本。"
        "hypothesis 和 reason 请使用你最自然、最清晰的语言表达。"
        "不要生成自由形式代码。"
        "不要返回 train_hyp_changes、recipe_changes 或任何额外的顶层字段。"
        "不要提议不在 allowed AI change fields 内的字段。"
        "changes 里至少要有一个非 null 字段。"
        "based_on_experiment_ids 必须列出你实际作为证据使用的实验 id。"
        "证据关系如下：base 是初始 baseline；source 是当前参考实验，相对 base 提供 delta 和结果；"
        "current 是 source 之后最近几轮未压缩尝试，每条都相对 source 给出 delta；"
        "history 是更早尝试的压缩摘要。"
        "如果没有单独的 source block，表示当前 source 与 base 相同。"
        "请把提供的 baseline、source、current 和 history 都当作证据。"
        "要把 discard、crash、timeout 和 failed 实验视为负面证据。"
        "当讨论 augmentation 时，只能使用合法、具体且可叠加的字段和值，例如 "
        "mixup_alpha、cutmix_alpha 和 random_erasing_prob。"
    )


def build_epoch_policy_instruction(epoch_search_enabled: bool) -> str:
    """Return the epoch-search instruction fragment for proposal policy."""
    if epoch_search_enabled:
        return (
            "你可以在有帮助时调整 epochs，但必须把它视为 training budget，而不是纯策略字段。"
            "如果你修改了 epochs，hypothesis 和 reason 里要明确说明这次 budget 取舍。"
        )
    return "不要修改 epochs；它当前是固定值，AI 不允许调整。"


def build_retry_note(last_error: str | None) -> str:
    """Return targeted retry guidance after one invalid proposal."""
    retry_lines = [
        "",
        "上一条 proposal 被判定为无效。请先修复下面的问题，再返回一份新的完整 JSON。",
    ]
    if last_error:
        retry_lines.append(f"上一次拒绝原因：{last_error}。")
    retry_lines.append("请基于完整历史选择一个更可执行的方向，不要重复这条无效方案。")
    return "\n".join(retry_lines)


def build_output_schema_block() -> str:
    """Return the output-schema guidance block."""
    return (
        '输出结构：\n'
        '{"task_type":"classification","model_name":"string","based_on_experiment_ids":["string"],'
        '"hypothesis":"string","changes":{"optimizer":"string|null","learning_rate":"number|null",'
        '"batch_size":"number|null","image_size":"number|null","epochs":"number|null","weight_decay":"number|null",'
        '"scheduler":"string|null","mixup_alpha":"number|null","cutmix_alpha":"number|null",'
        '"random_erasing_prob":"number|null","loss_name":"string|null",'
        '"focal_gamma":"number|null","label_smoothing":"number|null","aux_logits":"boolean|null",'
        '"neck_name":"string|null","head_name":"string|null"},"reason":"string"}\n'
        "changes 里只填写实际变更的字段；未变更字段保持为 null。"
    )


def build_policy_block_text(policy_payload: dict[str, Any]) -> str:
    """Render one policy block from structured policy data."""
    return build_payload_block_text("策略上下文", policy_payload)


def get_prompt_block_title(block_name: str) -> str:
    """Return the localized title for one prompt block."""
    return PROPOSAL_BLOCK_TITLES.get(block_name, block_name.replace("_", " ").title())


def build_payload_block_text(title: str, payload: Any) -> str:
    """Render one block title plus a JSON payload."""
    return f"{title}:\n{json.dumps(payload, ensure_ascii=False)}"
