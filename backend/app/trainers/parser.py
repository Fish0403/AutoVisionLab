"""Trainer-side YAML parser helpers for unified manifests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from app.trainers.manifest import TrainerManifest


LAYER_SECTION_NAMES = ("backbone", "neck", "head")
TOP_LEVEL_SECTION_ALIASES = {
    "model_recipe": "model",
    "train_hyp": "train",
    "dataset_recipe": "data",
    "search_policy": "search",
    "ranking_policy": "ranking",
}


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
    for section_name in LAYER_SECTION_NAMES:
        section_layers = normalized_payload.get(section_name)
        if isinstance(section_layers, list):
            normalized_payload[section_name] = [parse_model_layer_payload(layer_payload) for layer_payload in section_layers]

    architecture_payload = normalized_payload.get("architecture")
    if not isinstance(architecture_payload, dict):
        return normalized_payload

    for section_name in LAYER_SECTION_NAMES:
        section_layers = architecture_payload.get(section_name) or []
        architecture_payload[section_name] = [parse_model_layer_payload(layer_payload) for layer_payload in section_layers]
    return normalized_payload


def parse_trainer_manifest_payload(manifest_payload: dict[str, Any]) -> TrainerManifest:
    """Normalize one raw payload and parse it into the unified trainer manifest."""
    if not isinstance(manifest_payload, dict):
        raise ValueError("Trainer manifest payload must be a mapping")

    normalized_payload = deepcopy(manifest_payload)
    for legacy_name, canonical_name in TOP_LEVEL_SECTION_ALIASES.items():
        if canonical_name not in normalized_payload and legacy_name in normalized_payload:
            normalized_payload[canonical_name] = normalized_payload.pop(legacy_name)

    runtime_payload = dict(normalized_payload.get("runtime") or {})
    if "use_demo_mode" in normalized_payload and "use_demo_mode" not in runtime_payload:
        runtime_payload["use_demo_mode"] = normalized_payload.pop("use_demo_mode")
    if "participates_in_ranking" in normalized_payload and "participates_in_ranking" not in runtime_payload:
        runtime_payload["participates_in_ranking"] = normalized_payload.pop("participates_in_ranking")
    if runtime_payload:
        normalized_payload["runtime"] = runtime_payload

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
