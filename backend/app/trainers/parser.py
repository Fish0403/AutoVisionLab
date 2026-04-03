"""Trainer-side YAML parser helpers for unified manifests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from app.trainers.manifest import TrainerManifest


LAYER_SECTION_NAMES = ("backbone", "neck", "head")
LEGACY_TOP_LEVEL_SECTION_NAMES = ("model_recipe", "train_hyp", "dataset_recipe", "search_policy", "ranking_policy")
LEGACY_RUNTIME_FIELD_NAMES = ("use_demo_mode", "participates_in_ranking")


def parse_model_layer_payload(layer_payload: Any) -> dict[str, Any]:
    """Normalize one YOLO-style layer entry into the internal mapping form."""
    if isinstance(layer_payload, dict):
        normalized_payload = deepcopy(layer_payload)
        if "number" in normalized_payload and "repeat" not in normalized_payload:
            normalized_payload["repeat"] = normalized_payload.pop("number")
        return normalized_payload
    if isinstance(layer_payload, (list, tuple)):
        if len(layer_payload) not in {4, 5}:
            raise ValueError("Architecture layer list form must contain 4 or 5 items")
        from_indices, repeat, module_name, module_args = layer_payload[:4]
        tag = layer_payload[4] if len(layer_payload) == 5 else None
        return {
            "from": from_indices,
            "repeat": repeat,
            "module": module_name,
            "args": module_args,
            "tag": tag,
        }
    raise ValueError(f"Unsupported architecture layer payload: {layer_payload!r}")


def parse_model_recipe_payload(model_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one model recipe payload before schema validation."""
    normalized_payload = deepcopy(model_payload)
    if "architecture" in normalized_payload:
        raise ValueError("Legacy model.architecture is no longer supported")
    for section_name in LAYER_SECTION_NAMES:
        section_layers = normalized_payload.get(section_name)
        if isinstance(section_layers, list):
            normalized_payload[section_name] = [parse_model_layer_payload(layer_payload) for layer_payload in section_layers]
    return normalized_payload


def parse_trainer_manifest_payload(manifest_payload: dict[str, Any]) -> TrainerManifest:
    """Normalize one raw payload and parse it into the unified trainer manifest."""
    if not isinstance(manifest_payload, dict):
        raise ValueError("Trainer manifest payload must be a mapping")

    normalized_payload = deepcopy(manifest_payload)
    legacy_sections = sorted(
        field_name for field_name in LEGACY_TOP_LEVEL_SECTION_NAMES if field_name in normalized_payload
    )
    if legacy_sections:
        legacy_section_text = ", ".join(legacy_sections)
        raise ValueError(f"Legacy manifest sections are no longer supported: {legacy_section_text}")

    legacy_runtime_fields = sorted(
        field_name for field_name in LEGACY_RUNTIME_FIELD_NAMES if field_name in normalized_payload
    )
    if legacy_runtime_fields:
        legacy_runtime_text = ", ".join(legacy_runtime_fields)
        raise ValueError(
            "Legacy top-level runtime fields are no longer supported. "
            f"Move them under runtime: {legacy_runtime_text}"
        )

    model_payload = normalized_payload.get("model")
    if isinstance(model_payload, dict):
        normalized_payload["model"] = parse_model_recipe_payload(model_payload)

    return TrainerManifest.model_validate(normalized_payload)


def load_trainer_manifest(manifest_path: str | Path) -> TrainerManifest:
    """Load one unified trainer manifest from a YAML file."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as manifest_file:
        manifest_payload = yaml.safe_load(manifest_file) or {}
    return parse_trainer_manifest_payload(manifest_payload)
