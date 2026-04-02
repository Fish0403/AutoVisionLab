"""Backbone defaults for classification builder compatibility."""

from __future__ import annotations

def build_default_backbone_layers(base_model: str) -> list[dict[str, Any]]:
    """Return default backbone layer payloads for one supported base model."""
    del base_model
    return []
