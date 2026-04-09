"""Prompt helpers for auto-train search summaries."""

from __future__ import annotations

from typing import Any

from app.prompts.context_blocks import build_payload_block_text


def build_search_summary_system_prompt() -> str:
    """Return the stable system rules for one search-summary request."""
    return (
        "你负责总结一个图像分类搜索任务当前为止的结果。"
        "只返回 JSON，且只能包含一个键：summary_text。"
        "summary_text 必须使用简洁、客观、自然的中文。"
        "不要提到自己是 AI。不要输出列表、标题、额外字段或解释。"
        "这是一段结果摘要，不是下一步 proposal，不要给具体参数建议或执行建议。"
        "请基于提供的 stop reason、base、source、current stage 和 past stages 生成总结。"
        "其中，base 表示初始 baseline；source 表示当前最佳参考实验；"
        "current stage 表示当前 source 之后最近一阶段的尝试；past stages 表示更早阶段的阶段总结。"
        "请优先覆盖以下信息："
        "第一，搜索为何结束，以及本次搜索的大致范围；"
        "第二，当前领先实验是谁，以及最关键的结果指标；"
        "第三，本次搜索中哪些方向带来了收益、推进了 best，或表现更稳定；"
        "第四，哪些方向无效、不稳定，或反复失败。"
        "在表达搜索范围时，可以结合实验总数、已完成轮数、best 更新次数或阶段数来描述，"
        "但不要机械罗列所有数字，只有在这些数字能帮助理解搜索强度或推进路径时才提及。"
        "如果当前最佳始终没有超过 baseline，要明确说出这一点。"
        "如果搜索过程中 best 多次更新，也应概括这条推进路径。"
        "如果证据不足以支持明确的正向或负向结论，请直接说明“暂未形成明确结论”，不要编造趋势。"
        "状态值、decision 和其他内部字段可能是英文，你可以直接理解，不需要逐字翻译字段名。"
        "尽量避免重复同一指标、同一实验 id 或同一结论。"
    )


def build_search_summary_user_prompt(blocks: list[tuple[str, Any]]) -> str:
    """Render one search-summary user prompt from titled payload blocks."""
    return "\n\n".join(build_payload_block_text(title, payload) for title, payload in blocks)
