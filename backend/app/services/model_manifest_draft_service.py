"""Services for AI-assisted model manifest drafting and commit."""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torchvision.models as torchvision_models
import yaml

from app.llm.aihubmix_client import AIHubMixClient
from app.model_catalog import loader as model_catalog_loader
from app.model_catalog.registry import get_model_catalog_entry, reload_model_catalog
from app.model_catalog.schemas import ModelManifest
from app.prompts.model_manifest_draft import build_model_manifest_draft_prompt
from app.schemas.model_manifest_draft import (
    MinimalModelManifestDraftSpec,
    ModelManifestCommitRequest,
    ModelManifestCommitResponse,
    ModelManifestDraftRequest,
    ModelManifestDraftResponse,
    ModelManifestValidationResult,
)
from app.trainers.classification.model_builder_registry import build_classification_model_from_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRY_RUN_IMAGE_SIZE = 224
DEFAULT_DRY_RUN_NUM_CLASSES = 2
STANDARD_BATCH_SIZE_CHOICES = [32, 64, 128, 256]
MEMORY_SAFE_BATCH_SIZE_CHOICES = [16, 32, 64, 128]


@lru_cache(maxsize=1)
def _list_torchvision_model_names() -> list[str]:
    """Return the available torchvision model names."""
    return sorted(torchvision_models.list_models(module=torchvision_models))


def _normalize_search_token(value: str) -> str:
    """Normalize one free-text token for fuzzy model-name matching."""
    return re.sub(r"[\s_-]+", "", value.strip().lower())


def _normalize_manifest_filename_token(value: str) -> str:
    """Normalize one model name into a safe manifest filename stem."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return normalized.strip("_") or "draft_model"


def _collect_candidate_model_names(query: str, *, limit: int = 8) -> list[str]:
    """Return a compact list of likely torchvision model names for one query."""
    normalized_query = _normalize_search_token(query)
    if not normalized_query:
        return _list_torchvision_model_names()[:limit]

    all_model_names = _list_torchvision_model_names()
    direct_matches = [
        model_name
        for model_name in all_model_names
        if normalized_query in _normalize_search_token(model_name)
    ]
    normalized_name_map = {
        _normalize_search_token(model_name): model_name
        for model_name in all_model_names
    }
    fuzzy_tokens = difflib.get_close_matches(
        normalized_query,
        list(normalized_name_map.keys()),
        n=limit,
        cutoff=0.45,
    )
    fuzzy_matches = [normalized_name_map[token] for token in fuzzy_tokens]
    ordered_candidates = direct_matches + [name for name in fuzzy_matches if name not in direct_matches]
    return ordered_candidates[:limit] or all_model_names[:limit]


def _build_manifest_path(model_name: str, *, task_type: str = "classification") -> Path:
    """Return the target manifest path for one model name."""
    return model_catalog_loader.MODEL_MANIFEST_ROOT / task_type / f"{_normalize_manifest_filename_token(model_name)}.yaml"


def _format_manifest_display_path(path: Path) -> str:
    """Return a stable human-readable path for API responses."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _normalize_yaml_text(yaml_text: str) -> tuple[str, dict[str, Any]]:
    """Parse and re-dump YAML text into a stable preview form."""
    payload = yaml.safe_load(yaml_text) or {}
    if not isinstance(payload, dict):
        raise ValueError("Draft YAML must parse into a mapping at the document root.")
    normalized_yaml_text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    return normalized_yaml_text, payload


def _serialize_manifest_yaml_text(manifest: ModelManifest) -> str:
    """Serialize one validated manifest into stable YAML text."""
    return yaml.safe_dump(
        manifest.model_dump(mode="python", by_alias=True),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )


def _build_common_parameter_space_payload(*, model_name: str, batch_size_choices: list[int]) -> dict[str, Any]:
    """Return the common editable parameter payload shared by standard classifiers."""
    return {
        "model_name": model_name,
        "version": f"{model_name}@v1",
        "editable_params": {
            "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
            "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
            "batch_size": {"type": "discrete_values", "choices": batch_size_choices},
            "image_size": {"type": "number_range", "min": 1, "max": 10000},
            "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
            "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
            "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
            "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
            "loss_name": {
                "type": "enum",
                "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"],
            },
            "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
        },
    }


def _build_parameter_space_payload(draft_spec: MinimalModelManifestDraftSpec) -> dict[str, Any]:
    """Expand one parameter-space profile into a full editable parameter payload."""
    if draft_spec.parameter_space_profile == "classification_standard_v1":
        parameter_space_payload = _build_common_parameter_space_payload(
            model_name=draft_spec.model_name,
            batch_size_choices=STANDARD_BATCH_SIZE_CHOICES,
        )
    elif draft_spec.parameter_space_profile == "classification_memory_safe_v1":
        parameter_space_payload = _build_common_parameter_space_payload(
            model_name=draft_spec.model_name,
            batch_size_choices=MEMORY_SAFE_BATCH_SIZE_CHOICES,
        )
    elif draft_spec.parameter_space_profile == "googlenet_aux_v1":
        parameter_space_payload = _build_common_parameter_space_payload(
            model_name=draft_spec.model_name,
            batch_size_choices=STANDARD_BATCH_SIZE_CHOICES,
        )
        parameter_space_payload["editable_params"]["aux_logits"] = {
            "type": "enum",
            "choices": [True, False],
        }
        return parameter_space_payload
    else:  # pragma: no cover - defensive union guard
        raise ValueError(f"Unsupported parameter_space_profile: {draft_spec.parameter_space_profile}")

    if draft_spec.builder.type == "torchvision_classifier":
        if len(draft_spec.builder.supported_neck_names) > 1:
            parameter_space_payload["editable_params"]["neck_name"] = {
                "type": "enum",
                "choices": draft_spec.builder.supported_neck_names,
            }
        if len(draft_spec.builder.supported_head_names) > 1:
            parameter_space_payload["editable_params"]["head_name"] = {
                "type": "enum",
                "choices": draft_spec.builder.supported_head_names,
            }
    return parameter_space_payload


def _build_default_model_recipe_payload(draft_spec: MinimalModelManifestDraftSpec) -> dict[str, Any]:
    """Expand one recipe profile into a full default model recipe payload."""
    if draft_spec.recipe_profile == "googlenet_classifier_v1":
        return {
            "version": "model_recipe@v1",
            "task_type": "classification",
            "model_family": draft_spec.model_family,
            "base_model": draft_spec.model_name,
            "input_channels": 3,
            "width_multiple": 1.0,
            "components": {
                "backbone": {"name": "googlenet_native", "params": {}},
                "neck": {"name": "avg_pool", "params": {}},
                "head": {"name": "native_classifier", "params": {}},
            },
            "backbone_config": {
                "stem_variant": "standard",
                "attention_module": "none",
                "last_channel_multiplier": 1.0,
            },
            "head_config": {
                "pooling_type": "avg",
                "classifier_dropout": 0.0,
                "classifier_type": "linear",
            },
            "modules": {"aux_logits": False},
            "metadata": {
                "notes": f"Default {draft_spec.label} recipe using the native torchvision implementation.",
            },
        }

    if draft_spec.builder.type != "torchvision_classifier":
        raise ValueError(
            f"recipe_profile {draft_spec.recipe_profile} requires a torchvision_classifier builder."
        )
    return {
        "version": "model_recipe@v1",
        "task_type": "classification",
        "model_family": draft_spec.model_family,
        "base_model": draft_spec.model_name,
        "input_channels": 3,
        "width_multiple": 1.0,
        "components": {
            "backbone": {"name": draft_spec.builder.backbone_component_name, "params": {}},
            "neck": {"name": draft_spec.builder.default_neck_name, "params": {}},
            "head": {"name": draft_spec.builder.default_head_name, "params": {}},
        },
        "backbone_config": {
            "stem_variant": "standard",
            "attention_module": "none",
            "last_channel_multiplier": 1.0,
        },
        "head_config": {
            "pooling_type": "avg",
            "classifier_dropout": draft_spec.builder.default_dropout,
            "classifier_type": "linear",
        },
        "modules": {},
        "metadata": {
            "notes": f"Default {draft_spec.label} recipe using the torchvision backbone and classifier.",
        },
    }


def _expand_draft_spec_to_manifest(draft_spec: MinimalModelManifestDraftSpec) -> ModelManifest:
    """Expand one minimal AI draft into a full validated manifest."""
    if draft_spec.recipe_profile == "googlenet_classifier_v1" and draft_spec.builder.type != "googlenet_classifier":
        raise ValueError("googlenet_classifier_v1 requires builder.type=googlenet_classifier")
    if draft_spec.parameter_space_profile == "googlenet_aux_v1" and draft_spec.builder.type != "googlenet_classifier":
        raise ValueError("googlenet_aux_v1 requires builder.type=googlenet_classifier")
    if draft_spec.builder.type == "googlenet_classifier":
        if draft_spec.recipe_profile != "googlenet_classifier_v1":
            raise ValueError("googlenet_classifier builder must use recipe_profile=googlenet_classifier_v1")
        if draft_spec.parameter_space_profile != "googlenet_aux_v1":
            raise ValueError("googlenet_classifier builder must use parameter_space_profile=googlenet_aux_v1")

    manifest_payload = {
        "model_name": draft_spec.model_name,
        "label": draft_spec.label,
        "task_type": "classification",
        "model_family": draft_spec.model_family,
        "builder": draft_spec.builder.model_dump(mode="python"),
        "loss_adapter": "googlenet_aux" if draft_spec.builder.type == "googlenet_classifier" else "default",
        "supports_compare": draft_spec.supports_compare,
        "supports_search": draft_spec.supports_search,
        "is_default": draft_spec.is_default,
        "display_order": draft_spec.display_order,
        "default_model_recipe": _build_default_model_recipe_payload(draft_spec),
        "parameter_space": _build_parameter_space_payload(draft_spec),
    }
    return ModelManifest.model_validate(manifest_payload)


def _extract_provider_draft(
    response_payload: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    """Return resolved model name, AI preview text, full YAML text, and provider warnings."""
    provider_warnings = response_payload.get("warnings") or []
    if not isinstance(provider_warnings, list) or not all(isinstance(item, str) for item in provider_warnings):
        raise ValueError("Draft provider returned an invalid warnings field.")

    resolved_model_name = response_payload.get("resolved_model_name")
    if not isinstance(resolved_model_name, str) or not resolved_model_name.strip():
        raise ValueError("Draft provider returned an empty resolved_model_name.")

    yaml_text = response_payload.get("yaml_text")
    if isinstance(yaml_text, str) and yaml_text.strip():
        normalized_yaml_text, _ = _normalize_yaml_text(yaml_text)
        return resolved_model_name.strip(), normalized_yaml_text, normalized_yaml_text, provider_warnings

    draft_spec_payload = response_payload.get("draft_spec")
    if not isinstance(draft_spec_payload, dict):
        raise ValueError("Draft provider must return either yaml_text or draft_spec.")
    draft_spec = MinimalModelManifestDraftSpec.model_validate(draft_spec_payload)
    if draft_spec.model_name != resolved_model_name.strip():
        raise ValueError("draft_spec.model_name must match resolved_model_name.")
    expanded_manifest = _expand_draft_spec_to_manifest(draft_spec)
    ai_preview_text = yaml.safe_dump(
        draft_spec.model_dump(mode="python"),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    return (
        draft_spec.model_name,
        ai_preview_text,
        _serialize_manifest_yaml_text(expanded_manifest),
        provider_warnings,
    )


def _extract_primary_tensor(output: object) -> torch.Tensor:
    """Return the primary tensor from one model forward output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and isinstance(getattr(output, "logits"), torch.Tensor):
        return getattr(output, "logits")
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("Model forward output did not contain a primary tensor.")


def _validate_manifest_yaml_text(
    yaml_text: str,
    *,
    expected_model_name: str | None = None,
) -> tuple[ModelManifestValidationResult, ModelManifest | None, str]:
    """Validate one drafted manifest YAML text and return the parsed manifest when possible."""
    errors: list[str] = []
    warnings: list[str] = []
    manifest: ModelManifest | None = None
    normalized_yaml_text = yaml_text
    yaml_parse_ok = False
    schema_ok = False
    dry_run_build_ok = False
    dry_run_forward_ok = False

    try:
        normalized_yaml_text, payload = _normalize_yaml_text(yaml_text)
        yaml_parse_ok = True
    except Exception as error:  # pragma: no cover - defensive normalization guard
        payload = None
        errors.append(f"YAML parse failed: {error}")

    if isinstance(payload, dict):
        try:
            manifest = ModelManifest.model_validate(payload)
            schema_ok = True
        except Exception as error:
            errors.append(f"Manifest schema validation failed: {error}")

    if manifest is not None:
        if expected_model_name and manifest.model_name != expected_model_name:
            errors.append(
                f"Manifest model_name {manifest.model_name!r} does not match the expected model name {expected_model_name!r}."
            )

        if manifest.model_name != _normalize_manifest_filename_token(manifest.model_name):
            warnings.append(
                "model_name contains characters that will be normalized in the manifest filename."
            )

        target_path = _build_manifest_path(manifest.model_name, task_type=manifest.task_type)
        if target_path.exists() or get_model_catalog_entry(manifest.model_name) is not None:
            errors.append(f"Model manifest already exists for {manifest.model_name}.")

        try:
            model = build_classification_model_from_manifest(
                manifest,
                num_classes=DEFAULT_DRY_RUN_NUM_CLASSES,
            )
            dry_run_build_ok = True
            model.eval()
            input_channels = manifest.default_model_recipe.input_channels or 3
            with torch.no_grad():
                output = model(
                    torch.zeros(
                        (1, input_channels, DEFAULT_DRY_RUN_IMAGE_SIZE, DEFAULT_DRY_RUN_IMAGE_SIZE),
                        dtype=torch.float32,
                    )
                )
            primary_output = _extract_primary_tensor(output)
            if primary_output.ndim != 2:
                raise ValueError(f"Expected a rank-2 logits tensor, but got shape {tuple(primary_output.shape)}.")
            if primary_output.shape[0] != 1:
                raise ValueError(f"Expected batch dimension 1, but got shape {tuple(primary_output.shape)}.")
            if primary_output.shape[1] != DEFAULT_DRY_RUN_NUM_CLASSES:
                raise ValueError(
                    "Expected logits dimension "
                    f"{DEFAULT_DRY_RUN_NUM_CLASSES}, but got shape {tuple(primary_output.shape)}."
                )
            dry_run_forward_ok = True
        except Exception as error:
            errors.append(f"Dry-run validation failed: {error}")

    validation = ModelManifestValidationResult(
        is_valid=not errors,
        yaml_parse_ok=yaml_parse_ok,
        schema_ok=schema_ok,
        dry_run_build_ok=dry_run_build_ok,
        dry_run_forward_ok=dry_run_forward_ok,
        errors=errors,
        warnings=warnings,
    )
    return validation, manifest, normalized_yaml_text


def draft_model_manifest(request: ModelManifestDraftRequest) -> ModelManifestDraftResponse:
    """Generate one manifest draft preview from a user query."""
    candidate_model_names = _collect_candidate_model_names(request.query)
    system_prompt, user_prompt = build_model_manifest_draft_prompt(
        query=request.query,
        candidate_model_names=candidate_model_names,
    )
    response_payload, _provider_metadata = AIHubMixClient().create_json_completion_with_metadata(
        system_prompt,
        user_prompt,
    )
    resolved_model_name, ai_preview_text, yaml_text, provider_warnings = _extract_provider_draft(response_payload)

    validation, manifest, normalized_yaml_text = _validate_manifest_yaml_text(yaml_text)
    effective_model_name = manifest.model_name if manifest is not None else resolved_model_name.strip()
    target_path = _build_manifest_path(effective_model_name)
    return ModelManifestDraftResponse(
        query=request.query,
        resolved_model_name=effective_model_name,
        ai_preview_text=ai_preview_text,
        yaml_text=normalized_yaml_text,
        target_path=_format_manifest_display_path(target_path),
        provider_warnings=provider_warnings,
        validation=validation,
    )


def commit_model_manifest(request: ModelManifestCommitRequest) -> ModelManifestCommitResponse:
    """Persist one approved manifest draft after final validation."""
    validation, manifest, normalized_yaml_text = _validate_manifest_yaml_text(
        request.yaml_text,
        expected_model_name=request.expected_model_name,
    )
    if manifest is None or not validation.is_valid:
        raise ValueError("; ".join(validation.errors) or "Manifest draft is invalid.")

    manifest_path = _build_manifest_path(manifest.model_name, task_type=manifest.task_type)
    if manifest_path.exists():
        raise FileExistsError(f"Manifest file already exists: {manifest_path.name}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(normalized_yaml_text, encoding="utf-8")
    try:
        reloaded_entries = reload_model_catalog(task_type=manifest.task_type)
    except Exception as error:
        if manifest_path.exists():
            manifest_path.unlink()
        reload_model_catalog(task_type=manifest.task_type)
        raise ValueError(f"Manifest commit failed during catalog reload: {error}") from error

    return ModelManifestCommitResponse(
        model_name=manifest.model_name,
        manifest_path=_format_manifest_display_path(manifest_path),
        reloaded_model_count=len(reloaded_entries),
    )
