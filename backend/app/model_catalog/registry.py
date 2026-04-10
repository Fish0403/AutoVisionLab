"""Runtime registry for externally declared models."""

from __future__ import annotations

from functools import lru_cache

from app.model_catalog.loader import iter_manifest_paths, load_manifest_from_path
from app.model_catalog.schemas import ModelCatalogEntry, ModelSummary


def _sort_key(entry: ModelCatalogEntry) -> tuple[int, str]:
    return (entry.display_order, entry.label.lower())


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, ModelCatalogEntry]:
    catalog: dict[str, ModelCatalogEntry] = {}
    for manifest_path in iter_manifest_paths(task_type="classification"):
        manifest = load_manifest_from_path(manifest_path)
        entry = ModelCatalogEntry.from_manifest(manifest)
        if entry.model_name in catalog:
            raise ValueError(f"Duplicate model_name in model catalog: {entry.model_name}")
        catalog[entry.model_name] = entry
    return catalog


def list_model_catalog_entries(*, task_type: str = "classification") -> list[ModelCatalogEntry]:
    """Return all catalog entries for one task type."""
    return sorted(
        (entry for entry in _load_catalog().values() if entry.task_type == task_type),
        key=_sort_key,
    )


def list_model_summaries(*, task_type: str = "classification") -> list[ModelSummary]:
    """Return frontend-facing model summaries for one task type."""
    return [ModelSummary.from_entry(entry) for entry in list_model_catalog_entries(task_type=task_type)]


def get_model_catalog_entry(model_name: str) -> ModelCatalogEntry | None:
    """Return one catalog entry by model name."""
    return _load_catalog().get(model_name)

def list_compare_candidate_model_names(*, task_type: str = "classification") -> list[str]:
    """Return the default compare-model candidate list."""
    return [
        entry.model_name
        for entry in list_model_catalog_entries(task_type=task_type)
        if entry.supports_compare
    ]


def reload_model_catalog(*, task_type: str = "classification") -> list[ModelCatalogEntry]:
    """Clear the cached catalog and reload it from disk."""
    _load_catalog.cache_clear()
    return list_model_catalog_entries(task_type=task_type)
