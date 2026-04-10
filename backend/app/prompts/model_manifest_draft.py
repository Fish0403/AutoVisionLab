"""Prompt builder for AI-assisted model manifest drafting."""

from __future__ import annotations

import json
from pathlib import Path

from app.model_catalog import loader as model_catalog_loader


_EXAMPLE_FILENAMES = ("mobilenet_v2.yaml", "resnet18.yaml")


def _load_manifest_examples() -> list[dict[str, str]]:
    """Load a small set of current manifest examples for prompting."""
    example_dir = model_catalog_loader.MODEL_MANIFEST_ROOT / "classification"
    examples: list[dict[str, str]] = []
    for filename in _EXAMPLE_FILENAMES:
        example_path = example_dir / filename
        if not example_path.exists():
            continue
        examples.append(
            {
                "path": str(example_path.relative_to(Path(__file__).resolve().parents[3])),
                "content": example_path.read_text(encoding="utf-8").strip(),
            }
        )
    return examples


def build_model_manifest_draft_prompt(*, query: str, candidate_model_names: list[str]) -> tuple[str, str]:
    """Return the prompt pair for one manifest-drafting request."""
    system_prompt = (
        "你是 AutoVisionLab 的模型 manifest 起草器。"
        "你的任务是根据用户输入的模型名字，生成一份 classification 模型 manifest 最小草稿。"
        "只返回 JSON，且只能包含三个键：resolved_model_name、draft_spec、warnings。"
        "resolved_model_name 必须是最终推断出的模型名。"
        "draft_spec 必须是对象，不能是字符串。"
        "warnings 必须是字符串数组。"
        "不要生成 YAML，不要生成 Python 代码，不要生成解释性段落。"
        "draft_spec 只包含这些字段：model_name、label、model_family、builder、recipe_profile、parameter_space_profile、supports_compare、supports_search、is_default、display_order。"
        "builder.type 只能使用 torchvision_classifier 或 googlenet_classifier。"
        "recipe_profile 只能使用 torchvision_classifier_v1 或 googlenet_classifier_v1。"
        "parameter_space_profile 只能使用 classification_standard_v1、classification_memory_safe_v1、googlenet_aux_v1。"
        "如果用户输入疑似拼写错误，请先纠正成最可能的标准模型名。"
        "优先选择 torchvision 现成分类模型。"
        "除非有明确理由，supports_compare 和 supports_search 设为 true，is_default 设为 false，display_order 设为 120。"
        "不要在 draft_spec 中生成 default_model_recipe 或 parameter_space，后端会自动补全。"
        "如果不确定，请在 warnings 中写清楚假设。"
    )
    user_prompt = (
        f"用户输入:\n{query}\n\n"
        "候选 torchvision 模型名（按接近程度排序）:\n"
        f"{json.dumps(candidate_model_names, ensure_ascii=False)}\n\n"
        "当前项目 manifest 示例:\n"
        f"{json.dumps(_load_manifest_examples(), ensure_ascii=False)}\n\n"
        "输出要求:\n"
        "1. 返回合法 JSON。\n"
        "2. draft_spec.model_name 要与 resolved_model_name 一致。\n"
        "3. 如果模型属于 resnet / mobilenet / efficientnet 这类常见族，model_family 请写对应族名。\n"
        "4. 如果模型更适合作为独立 family，例如 vgg，请直接使用对应 family 名。\n"
        "5. 如果用了 torchvision_classifier，请补齐 builder 所需字段。\n"
        "6. 常规 torchvision 分类模型优先使用 recipe_profile=torchvision_classifier_v1。\n"
        "7. 常规 torchvision 分类模型优先使用 parameter_space_profile=classification_standard_v1；"
        "如果模型明显更重、更占显存，例如 vgg，可使用 classification_memory_safe_v1；"
        "GoogLeNet 使用 googlenet_aux_v1。"
    )
    return system_prompt, user_prompt
