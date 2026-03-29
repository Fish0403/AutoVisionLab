"""Load built-in trainer recipe templates from repository YAML files."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


TRAINER_ROOT = Path(__file__).resolve().parent
TRAINER_RECIPE_ROOT = TRAINER_ROOT / "recipes"

BUILTIN_MODEL_RECIPE_PATHS = {
    "mobilenet_v3_small": TRAINER_RECIPE_ROOT / "classification" / "mobilenet_v3_small.yaml",
}


def has_builtin_model_recipe(model_name: str) -> bool:
    """Return whether one model has a built-in recipe template file."""
    return model_name in BUILTIN_MODEL_RECIPE_PATHS


def get_builtin_model_recipe_path(model_name: str) -> Path:
    """Return the YAML path for one built-in model recipe template."""
    try:
        return BUILTIN_MODEL_RECIPE_PATHS[model_name]
    except KeyError as exc:
        raise ValueError(f"No built-in model recipe template for {model_name}") from exc


@lru_cache(maxsize=None)
def _load_builtin_model_recipe_payload(model_name: str) -> dict[str, Any]:
    """Read one built-in model recipe YAML file into a payload dictionary."""
    recipe_path = get_builtin_model_recipe_path(model_name)
    with recipe_path.open("r", encoding="utf-8") as recipe_file:
        payload = yaml.safe_load(recipe_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Built-in model recipe template must be a mapping: {recipe_path}")
    return payload


def load_builtin_model_recipe_payload(model_name: str) -> dict[str, Any]:
    """Return one defensive copy of a built-in recipe payload."""
    return deepcopy(_load_builtin_model_recipe_payload(model_name))
