"""Prompt helpers for compare-result summaries."""

from __future__ import annotations

import json

from app.schemas.run import ModelCompareSummary


def build_compare_summary_prompt(summary: ModelCompareSummary) -> tuple[str, str]:
    """Build the prompt pair for one compare-result summary."""
    candidate_payload = [
        {
            "model_name": candidate.model_name,
            "status": candidate.status,
            "top1_acc": candidate.top1_acc,
            "latency_ms": candidate.latency_ms,
            "parameter_count_million": candidate.parameter_count_million,
            "normalized_config_notes": candidate.normalized_config_notes,
        }
        for candidate in summary.candidate_results
    ]
    system_prompt = (
        "你负责总结一个模型对比任务的结果。"
        "只返回 JSON，且只能包含一个键：summary_text。"
        "summary_text 必须使用简洁、客观、自然的中文。"
        "不要提到自己是 AI。不要给下一步建议。不要输出列表、标题或额外字段。"
        "如果存在成功候选，需要指出当前领先模型，并概括最关键的准确率和延迟信息。"
        "如果全部候选都失败，也要明确说明。"
        "状态值和字段名可能是英文，可以直接理解，不需要逐字翻译。"
    )
    user_prompt = (
        "请为一个 workspace results panel 总结下面的 compare 结果。\n"
        f"共享基线配置:\n{json.dumps(summary.shared_baseline_config, ensure_ascii=False)}\n"
        f"候选结果:\n{json.dumps(candidate_payload, ensure_ascii=False)}\n"
    )
    return system_prompt, user_prompt
