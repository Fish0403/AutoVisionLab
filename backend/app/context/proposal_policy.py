"""Build structured policy payloads for proposal prompting."""

from __future__ import annotations

from typing import Any


def build_policy_prompt_payload(
    *,
    allowed_field_definitions: dict[str, Any],
    epoch_policy_instruction: str,
    require_non_basic_change: bool,
    max_changed_fields: int | None,
) -> dict[str, Any]:
    """Build the structured proposal policy payload."""
    return {
        "allowed_field_definitions": allowed_field_definitions,
        "epoch_policy_instruction": epoch_policy_instruction,
        "task_rules": [
            "当前 source experiment 的 config 就是下一轮 follow-up 的完整起点；请把它当作 baseline state，只返回 changes 里的增量字段。",
            "每个值都必须严格遵守 allowed field definitions。",
            "如果当前 parameter space 开启了组件级搜索，请在 changes 里使用 neck_name 和 head_name，而不是输出 recipe 结构补丁。",
            "如果你修改 image_size，它必须是 allowed parameter space 内的正整数。",
            "如果这样能让下一步更高效，通常优先选择不大于当前 source experiment image_size 的值，但这只是搜索偏好，不是硬性规则。",
            "你可以自由选择下一步搜索方向，但不要机械地重复最近几轮几乎相同的建议。",
        ],
        "require_non_basic_change": require_non_basic_change,
        "max_changed_fields": max_changed_fields,
    }
