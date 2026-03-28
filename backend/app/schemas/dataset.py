"""Schemas for local dataset discovery."""

from pydantic import BaseModel, Field


class LocalDatasetSummary(BaseModel):
    """One locally discovered dataset and its readiness status."""

    name: str
    source_dir: str | None = None
    classification_dir: str | None = None
    has_source_dir: bool = False
    has_prepared_source_dir: bool = False
    train_manifest_exists: bool = False
    val_manifest_exists: bool = False
    test_manifest_exists: bool = False
    is_ready_for_training: bool = False
    original_image_size: int | None = None
    image_size_options: list[int] = Field(default_factory=list)
    message: str | None = None
