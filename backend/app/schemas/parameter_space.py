"""Editable parameter space schemas."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.ranking_policy import RankingPolicy


DEFAULT_BASIC_HPARAM_SEARCH_FIELDS = [
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "scheduler",
    "label_smoothing",
]

RecipeTaskType = Literal["classification", "detection", "segmentation"]


class EnumParamDefinition(BaseModel):
    """Parameter with a fixed choice set."""

    type: Literal["enum"]
    choices: list[str | int | bool]


class NumberRangeParamDefinition(BaseModel):
    """Parameter with a numeric range."""

    type: Literal["number_range"]
    min: float
    max: float


class DiscreteValuesParamDefinition(BaseModel):
    """Parameter with discrete allowed numeric values."""

    type: Literal["discrete_values"]
    choices: list[int | float]


ParameterDefinition = EnumParamDefinition | NumberRangeParamDefinition | DiscreteValuesParamDefinition


class EditableParameterSpace(BaseModel):
    """White-listed editable parameter space for a model."""

    model_name: str
    version: str
    editable_params: dict[str, ParameterDefinition]


class LossParams(BaseModel):
    """Structured parameters for supported loss functions."""

    focal_gamma: float = Field(default=2.0, gt=0)


class AugmentationParams(BaseModel):
    """Structured parameters for supported augmentation strategies."""

    mixup_alpha: float = Field(default=0.0, ge=0)
    cutmix_alpha: float = Field(default=0.0, ge=0)
    random_erasing_prob: float = Field(default=0.0, ge=0, le=1)


def _normalize_null_label_smoothing(payload: Any) -> Any:
    """Treat explicit null label smoothing as the default disabled value."""
    if not isinstance(payload, dict):
        return payload
    normalized_payload = deepcopy(payload)
    if normalized_payload.get("label_smoothing") is None:
        normalized_payload["label_smoothing"] = 0.0
    return normalized_payload

class ExperimentParams(BaseModel):
    """Structured training parameters allowed in the MVP."""

    optimizer: str
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    weight_decay: float = Field(ge=0)
    scheduler: str
    augmentation_params: AugmentationParams = Field(default_factory=AugmentationParams)
    loss_name: Literal["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"] = (
        "cross_entropy_with_label_smoothing"
    )
    loss_params: LossParams = Field(default_factory=LossParams)
    label_smoothing: float = Field(ge=0, le=0.2)
    aux_logits: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_null_label_smoothing(cls, payload: Any) -> Any:
        """Treat explicit null label smoothing as the default disabled value."""
        return _normalize_null_label_smoothing(payload)


class SearchPolicy(BaseModel):
    """Controls what the AI is allowed to search automatically."""

    allow_basic_hparam_search: bool = True
    allowed_basic_hparam_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_BASIC_HPARAM_SEARCH_FIELDS))
    allow_strategy_search: bool = False
    allow_loss_search: bool = False
    allow_augmentation_search: bool = False
    allow_model_module_search: bool = False
    require_manual_approval_for_high_impact_changes: bool = True


class ModelRecipeBackbone(BaseModel):
    """Structured backbone-level recipe config."""

    stem_variant: str = "standard"
    attention_module: str = "none"
    last_channel_multiplier: float = Field(default=1.0, gt=0)


class ModelRecipeHead(BaseModel):
    """Structured head-level recipe config."""

    pooling_type: str = "avg"
    classifier_dropout: float = Field(default=0.0, ge=0, le=1)
    classifier_type: str = "linear"


class ModelRecipeMetadata(BaseModel):
    """Human-readable metadata for one model recipe."""

    notes: str | None = None


class ModelRecipeComponentSlot(BaseModel):
    """One named component slot in the higher-level recipe view."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelRecipeComponents(BaseModel):
    """Coarse component slots exposed to search and UI layers."""

    backbone: ModelRecipeComponentSlot
    neck: ModelRecipeComponentSlot
    head: ModelRecipeComponentSlot


class ModelRecipeArchitectureLayer(BaseModel):
    """One YOLO-style architecture layer entry."""

    model_config = ConfigDict(populate_by_name=True)

    from_indices: int | list[int] = Field(alias="from", serialization_alias="from")
    repeat: int = Field(default=1, gt=0)
    module: str
    args: list[Any] = Field(default_factory=list)
    tag: str | None = None


class ModelRecipe(BaseModel):
    """Structured model recipe attached to one experiment config."""

    model_config = ConfigDict(extra="forbid")

    version: str = "model_recipe@v1"
    task_type: RecipeTaskType = "classification"
    model_family: str
    base_model: str
    nc: int | None = Field(default=None, gt=0)
    input_channels: int = Field(default=3, gt=0)
    width_multiple: float = Field(default=1.0, gt=0)
    components: ModelRecipeComponents | None = None
    backbone_config: ModelRecipeBackbone = Field(default_factory=ModelRecipeBackbone)
    backbone: list[ModelRecipeArchitectureLayer] = Field(default_factory=list)
    neck: list[ModelRecipeArchitectureLayer] = Field(default_factory=list)
    head_config: ModelRecipeHead = Field(default_factory=ModelRecipeHead)
    head: list[ModelRecipeArchitectureLayer] = Field(default_factory=list)
    modules: dict[str, Any] = Field(default_factory=dict)
    metadata: ModelRecipeMetadata = Field(default_factory=ModelRecipeMetadata)


class TrainHypAugmentation(BaseModel):
    """Structured augmentation block for one train hyp recipe."""

    mixup: float = Field(default=0.0, ge=0)
    cutmix: float = Field(default=0.0, ge=0)
    random_erasing: float = Field(default=0.0, ge=0, le=1)


class TrainHypLoss(BaseModel):
    """Structured loss block for one train hyp recipe."""

    name: str = "cross_entropy_with_label_smoothing"


class TrainHypRuntime(BaseModel):
    """Structured runtime flags for one train hyp recipe."""

    amp: bool = False
    grad_clip_norm: float | None = Field(default=None, gt=0)


class TrainHypMetadata(BaseModel):
    """Human-readable metadata for one train hyp recipe."""

    notes: str | None = None


class TrainHyp(BaseModel):
    """Structured training hyperparameter recipe."""

    version: str = "train_hyp@v1"
    task_type: RecipeTaskType = "classification"
    optimizer: str
    lr0: float = Field(gt=0)
    lrf: float = Field(default=0.01, ge=0)
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(ge=0)
    warmup_epochs: float = Field(default=0.0, ge=0)
    scheduler: str
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0, le=1)
    label_smoothing: float = Field(default=0.0, ge=0, le=0.2)
    fl_gamma: float = Field(default=0.0, ge=0)
    augmentation: TrainHypAugmentation = Field(default_factory=TrainHypAugmentation)
    loss: TrainHypLoss = Field(default_factory=TrainHypLoss)
    runtime: TrainHypRuntime = Field(default_factory=TrainHypRuntime)
    metadata: TrainHypMetadata = Field(default_factory=TrainHypMetadata)

    @model_validator(mode="before")
    @classmethod
    def normalize_null_label_smoothing(cls, payload: Any) -> Any:
        """Treat explicit null label smoothing as the default disabled value."""
        return _normalize_null_label_smoothing(payload)

    def to_experiment_params(self, *, aux_logits: bool | None = None) -> "ExperimentParams":
        """Convert the training recipe into the legacy result payload shape."""
        return ExperimentParams(
            optimizer=self.optimizer,
            learning_rate=self.lr0,
            batch_size=self.batch_size,
            image_size=self.image_size,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            scheduler=self.scheduler,
            augmentation_params=AugmentationParams(
                mixup_alpha=self.augmentation.mixup,
                cutmix_alpha=self.augmentation.cutmix,
                random_erasing_prob=self.augmentation.random_erasing,
            ),
            loss_name=self.loss.name,
            loss_params=LossParams(
                focal_gamma=self.fl_gamma if self.loss.name == "focal_loss" else 2.0,
            ),
            label_smoothing=self.label_smoothing,
            aux_logits=aux_logits,
        )


class DatasetRecipeSource(BaseModel):
    """Structured source paths for one dataset recipe."""

    root_dir: str
    prepared_source_dir: str | None = None


class DatasetRecipeSplits(BaseModel):
    """Structured split paths for one dataset recipe."""

    train_manifest: str
    val_manifest: str
    test_manifest: str | None = None


class DatasetRecipeMetadata(BaseModel):
    """Human-readable metadata for one dataset recipe."""

    image_size_options: list[int] = Field(default_factory=list)
    notes: str | None = None


class DatasetRecipe(BaseModel):
    """Structured dataset recipe for one experiment config."""

    version: str = "dataset_recipe@v1"
    task_type: RecipeTaskType = "classification"
    dataset_name: str
    class_names: list[str] = Field(default_factory=list)
    source: DatasetRecipeSource
    splits: DatasetRecipeSplits
    metadata: DatasetRecipeMetadata = Field(default_factory=DatasetRecipeMetadata)


def _infer_dataset_class_names(dataset_recipe: DatasetRecipe) -> list[str]:
    """Infer dataset class names from available split manifests when possible."""
    from app.trainers.classification.data_loading import collect_manifest_classes
    from app.services.dataset_paths import resolve_dataset_dir

    raw_manifest_paths = [
        Path(dataset_recipe.splits.train_manifest),
        Path(dataset_recipe.splits.val_manifest),
    ]
    if dataset_recipe.splits.test_manifest:
        raw_manifest_paths.append(Path(dataset_recipe.splits.test_manifest))

    discovered_class_names: set[str] = set()
    for manifest_path in raw_manifest_paths:
        if not manifest_path.exists() and len(manifest_path.parts) >= 3:
            candidate_parent = resolve_dataset_dir(
                manifest_path.parent.parent,
                dataset_recipe.dataset_name,
                strict=False,
            )
            candidate_path = candidate_parent / manifest_path.name
            if candidate_path.exists():
                manifest_path = candidate_path
        if not manifest_path.exists():
            continue
        discovered_class_names.update(collect_manifest_classes(manifest_path))
    return sorted(discovered_class_names)


class ExperimentConfig(BaseModel):
    """Final config consumed by the trainer."""

    task_type: Literal["classification"]
    dataset: str
    model_family: str
    model_name: str
    parameter_space_version: str
    use_demo_mode: bool = False
    participates_in_ranking: bool = True
    search_policy: SearchPolicy = Field(default_factory=SearchPolicy)
    ranking_policy: RankingPolicy = Field(default_factory=RankingPolicy)
    params: ExperimentParams | None = None
    model_recipe: ModelRecipe
    train_hyp: TrainHyp
    dataset_recipe: DatasetRecipe

    def use_aux_logits(self) -> bool:
        """Return whether the active model should enable auxiliary logits."""
        from app.model_catalog.registry import get_model_catalog_entry

        model_entry = get_model_catalog_entry(self.model_name)
        if model_entry is None or model_entry.loss_adapter != "googlenet_aux":
            return False
        return bool(self.model_recipe.modules.get("aux_logits", False))

    def result_params(self) -> ExperimentParams:
        """Return the normalized result payload derived from the active recipes."""
        if self.params is None:
            raise ValueError("ExperimentConfig.params must be hydrated from train_hyp before use")
        return self.params

    @model_validator(mode="after")
    def validate_and_hydrate_structured_config(self) -> "ExperimentConfig":
        """Validate cross-field consistency for the structured experiment config."""
        from app.model_catalog.registry import get_model_catalog_entry

        model_entry = get_model_catalog_entry(self.model_name)
        if model_entry is None:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if model_entry.task_type != self.task_type:
            raise ValueError("model_name task_type must match ExperimentConfig.task_type")

        self.model_recipe = hydrate_model_recipe(self.model_recipe)

        if self.model_recipe.task_type != self.task_type:
            raise ValueError("model_recipe.task_type must match ExperimentConfig.task_type")
        if self.train_hyp.task_type != self.task_type:
            raise ValueError("train_hyp.task_type must match ExperimentConfig.task_type")
        if self.dataset_recipe.task_type != self.task_type:
            raise ValueError("dataset_recipe.task_type must match ExperimentConfig.task_type")
        if self.model_recipe.base_model != self.model_name:
            raise ValueError("model_recipe.base_model must match ExperimentConfig.model_name")
        if self.model_recipe.model_family != self.model_family:
            raise ValueError("model_recipe.model_family must match ExperimentConfig.model_family")
        if self.model_family != model_entry.model_family:
            raise ValueError("ExperimentConfig.model_family must match the registered model family")
        if self.dataset_recipe.dataset_name != self.dataset:
            raise ValueError("dataset_recipe.dataset_name must match ExperimentConfig.dataset")

        if not self.dataset_recipe.class_names:
            self.dataset_recipe.class_names = _infer_dataset_class_names(self.dataset_recipe)
        if self.model_recipe.nc is None and self.dataset_recipe.class_names:
            self.model_recipe.nc = len(self.dataset_recipe.class_names)
        self.params = self.train_hyp.to_experiment_params(aux_logits=self.use_aux_logits())
        return self


def apply_proposal_changes_to_train_hyp(train_hyp_payload: dict[str, Any], proposal_changes: dict[str, Any]) -> TrainHyp:
    """Apply legacy proposal fields onto one train_hyp payload."""
    updated_payload = deepcopy(train_hyp_payload)
    for field_name, value in proposal_changes.items():
        if value is None:
            continue
        if field_name == "optimizer":
            updated_payload["optimizer"] = value
        elif field_name == "learning_rate":
            updated_payload["lr0"] = value
        elif field_name == "batch_size":
            updated_payload["batch_size"] = value
        elif field_name == "image_size":
            updated_payload["image_size"] = value
        elif field_name == "epochs":
            updated_payload["epochs"] = value
        elif field_name == "weight_decay":
            updated_payload["weight_decay"] = value
        elif field_name == "scheduler":
            updated_payload["scheduler"] = value
        elif field_name == "mixup_alpha":
            updated_payload.setdefault("augmentation", {})["mixup"] = value
        elif field_name == "cutmix_alpha":
            updated_payload.setdefault("augmentation", {})["cutmix"] = value
        elif field_name == "random_erasing_prob":
            updated_payload.setdefault("augmentation", {})["random_erasing"] = value
        elif field_name == "loss_name":
            updated_payload.setdefault("loss", {})["name"] = value
        elif field_name == "focal_gamma":
            updated_payload["fl_gamma"] = value
        elif field_name == "label_smoothing":
            updated_payload["label_smoothing"] = value
    return TrainHyp.model_validate(updated_payload)


def build_train_hyp_change_payload(proposal_changes: dict[str, Any]) -> dict[str, Any]:
    """Project legacy proposal fields into a partial train_hyp payload."""
    train_hyp_changes: dict[str, Any] = {}
    augmentation_changes: dict[str, Any] = {}
    loss_changes: dict[str, Any] = {}

    for field_name, value in proposal_changes.items():
        if value is None:
            continue
        if field_name == "optimizer":
            train_hyp_changes["optimizer"] = value
        elif field_name == "learning_rate":
            train_hyp_changes["lr0"] = value
        elif field_name == "batch_size":
            train_hyp_changes["batch_size"] = value
        elif field_name == "image_size":
            train_hyp_changes["image_size"] = value
        elif field_name == "epochs":
            train_hyp_changes["epochs"] = value
        elif field_name == "weight_decay":
            train_hyp_changes["weight_decay"] = value
        elif field_name == "scheduler":
            train_hyp_changes["scheduler"] = value
        elif field_name == "mixup_alpha":
            augmentation_changes["mixup"] = value
        elif field_name == "cutmix_alpha":
            augmentation_changes["cutmix"] = value
        elif field_name == "random_erasing_prob":
            augmentation_changes["random_erasing"] = value
        elif field_name == "loss_name":
            loss_changes["name"] = value
        elif field_name == "focal_gamma":
            train_hyp_changes["fl_gamma"] = value
        elif field_name == "label_smoothing":
            train_hyp_changes["label_smoothing"] = value

    if augmentation_changes:
        train_hyp_changes["augmentation"] = augmentation_changes
    if loss_changes:
        train_hyp_changes["loss"] = loss_changes
    return train_hyp_changes


def _merge_partial_payload(base_payload: dict[str, Any], partial_changes: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge one partial structured payload onto a base payload."""
    merged_payload = deepcopy(base_payload)
    for field_name, value in partial_changes.items():
        if isinstance(value, dict) and isinstance(merged_payload.get(field_name), dict):
            merged_payload[field_name] = _merge_partial_payload(merged_payload.get(field_name) or {}, value)
        else:
            merged_payload[field_name] = value
    return merged_payload


def _normalize_model_recipe_components_payload(model_recipe_payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure model_recipe.components is complete before schema validation."""
    normalized_payload = deepcopy(model_recipe_payload)
    base_model = normalized_payload.get("base_model")
    if not isinstance(base_model, str) or not base_model.strip():
        return normalized_payload

    default_components = build_default_model_recipe_components(base_model=base_model)
    legacy_neck_name = normalized_payload.pop("neck_name", None)
    if isinstance(legacy_neck_name, str) and legacy_neck_name.strip():
        normalized_payload.setdefault("components", {}).setdefault("neck", {})["name"] = legacy_neck_name
        if legacy_neck_name == "gem_pool":
            normalized_payload.setdefault("head_config", {})["pooling_type"] = "gem"
        elif legacy_neck_name == "avg_pool":
            normalized_payload.setdefault("head_config", {})["pooling_type"] = "avg"

    legacy_head_name = normalized_payload.pop("head_name", None)
    if isinstance(legacy_head_name, str) and legacy_head_name.strip():
        normalized_payload.setdefault("components", {}).setdefault("head", {})["name"] = legacy_head_name
        normalized_payload = _merge_partial_payload(
            normalized_payload,
            _build_head_component_compatibility_payload(legacy_head_name),
        )
    for slot_name in ("backbone", "neck", "head"):
        slot_payload = normalized_payload.get(slot_name)
        if isinstance(slot_payload, dict) and isinstance(slot_payload.get("name"), str):
            normalized_payload.setdefault("components", {})[slot_name] = slot_payload
            normalized_payload.pop(slot_name, None)
    current_components = normalized_payload.get("components")
    if isinstance(current_components, dict):
        normalized_payload["components"] = _merge_partial_payload(default_components, current_components)
    else:
        normalized_payload["components"] = default_components
    return normalized_payload


def build_default_model_recipe(*, model_name: str, task_type: str, model_family: str) -> ModelRecipe:
    """Build one default model recipe, using a built-in template when available."""
    from app.model_catalog.registry import get_model_catalog_entry

    model_entry = get_model_catalog_entry(model_name)
    if model_entry is not None:
        recipe_payload = model_entry.default_model_recipe.model_dump(mode="python", by_alias=True)
        recipe_payload["task_type"] = task_type
        recipe_payload["model_family"] = model_family
        recipe_payload["base_model"] = model_name
        return ModelRecipe.model_validate(recipe_payload)

    recipe = ModelRecipe(
        task_type=task_type,
        model_family=model_family,
        base_model=model_name,
        components=ModelRecipeComponents.model_validate(build_default_model_recipe_components(base_model=model_name)),
        backbone_config=ModelRecipeBackbone(),
    )
    if model_name == "mobilenet_v2":
        recipe.head_config.classifier_dropout = 0.2
    return recipe


def hydrate_model_recipe(model_recipe: ModelRecipe) -> ModelRecipe:
    """Fill one partial model recipe with the built-in architecture template when available."""
    from app.model_catalog.registry import get_model_catalog_entry

    model_entry = get_model_catalog_entry(model_recipe.base_model)
    if model_entry is None:
        return model_recipe
    builtin_payload = model_entry.default_model_recipe.model_dump(mode="python", by_alias=True)
    merged_payload = _merge_partial_payload(
        builtin_payload,
        model_recipe.model_dump(exclude_none=True, exclude_unset=True),
    )
    return ModelRecipe.model_validate(merged_payload)


def build_default_model_recipe_components(base_model: str, *, pooling_type: str = "avg") -> dict[str, Any]:
    """Return the default component slots for one base model."""
    if base_model == "mobilenet_v2":
        return {
            "backbone": {"name": "mobilenet_v2_native", "params": {}},
            "neck": {"name": "avg_pool", "params": {}},
            "head": {"name": "native_classifier", "params": {}},
        }
    if base_model in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        neck_name = "gem_pool" if pooling_type == "gem" else "avg_pool"
        return {
            "backbone": {"name": f"{base_model}_native", "params": {}},
            "neck": {"name": neck_name, "params": {}},
            "head": {"name": "native_classifier", "params": {}},
        }
    if base_model == "efficientnet_b0":
        neck_name = "gem_pool" if pooling_type == "gem" else "avg_pool"
        return {
            "backbone": {"name": f"{base_model}_native", "params": {}},
            "neck": {"name": neck_name, "params": {}},
            "head": {"name": "native_classifier", "params": {}},
        }
    if base_model == "googlenet":
        return {
            "backbone": {"name": "googlenet_native", "params": {}},
            "neck": {"name": "avg_pool", "params": {}},
            "head": {"name": "native_classifier", "params": {}},
        }
    if base_model in {"resnet18", "resnet34"}:
        return {
            "backbone": {"name": f"{base_model}_native", "params": {}},
            "neck": {"name": "avg_pool", "params": {}},
            "head": {"name": "native_classifier", "params": {}},
        }
    return {
        "backbone": {"name": f"{base_model}_native", "params": {}},
        "neck": {"name": "identity", "params": {}},
        "head": {"name": "native_classifier", "params": {}},
    }


def apply_train_hyp_change_payload(train_hyp_payload: dict[str, Any], train_hyp_changes: dict[str, Any]) -> TrainHyp:
    """Apply a partial structured train_hyp payload onto one base train_hyp object."""
    return TrainHyp.model_validate(_merge_partial_payload(train_hyp_payload, train_hyp_changes))


def _build_head_component_compatibility_payload(head_name: str) -> dict[str, Any]:
    """Return one compatibility patch for component-level head choices."""
    if head_name == "linear":
        return {"head_config": {"classifier_type": "linear", "classifier_dropout": 0.0}}
    if head_name == "dropout_linear":
        return {"head_config": {"classifier_type": "linear", "classifier_dropout": 0.2}}
    if head_name == "native_classifier":
        return {"head_config": {"classifier_type": "linear"}}
    return {}


def apply_proposal_changes_to_model_recipe(
    model_recipe_payload: dict[str, Any],
    proposal_changes: dict[str, Any],
) -> ModelRecipe:
    """Apply legacy proposal fields onto one model_recipe payload."""
    updated_payload = deepcopy(model_recipe_payload)
    for field_name, value in proposal_changes.items():
        if value is None:
            continue
        if field_name == "width_multiple":
            updated_payload["width_multiple"] = value
        elif field_name == "pooling_type":
            updated_payload.setdefault("head_config", {})["pooling_type"] = value
            updated_payload.setdefault("components", {}).setdefault("neck", {})["name"] = "gem_pool" if value == "gem" else "avg_pool"
        elif field_name == "classifier_dropout":
            updated_payload.setdefault("head_config", {})["classifier_dropout"] = value
        elif field_name == "neck_name":
            updated_payload.setdefault("components", {}).setdefault("neck", {})["name"] = value
            if value == "gem_pool":
                updated_payload.setdefault("head_config", {})["pooling_type"] = "gem"
            elif value == "avg_pool":
                updated_payload.setdefault("head_config", {})["pooling_type"] = "avg"
        elif field_name == "head_name":
            updated_payload.setdefault("components", {}).setdefault("head", {})["name"] = value
            updated_payload = _merge_partial_payload(
                updated_payload,
                _build_head_component_compatibility_payload(str(value)),
            )
        elif field_name == "aux_logits":
            updated_payload.setdefault("modules", {})["aux_logits"] = value
    return ModelRecipe.model_validate(_normalize_model_recipe_components_payload(updated_payload))


def build_model_recipe_change_payload(proposal_changes: dict[str, Any]) -> dict[str, Any]:
    """Project legacy proposal fields into a partial model_recipe payload."""
    recipe_changes: dict[str, Any] = {}
    modules_changes: dict[str, Any] = {}

    for field_name, value in proposal_changes.items():
        if value is None:
            continue
        if field_name == "width_multiple":
            recipe_changes["width_multiple"] = value
        elif field_name == "pooling_type":
            recipe_changes.setdefault("head_config", {})["pooling_type"] = value
            recipe_changes.setdefault("components", {}).setdefault("neck", {})["name"] = (
                "gem_pool" if value == "gem" else "avg_pool"
            )
        elif field_name == "classifier_dropout":
            recipe_changes.setdefault("head_config", {})["classifier_dropout"] = value
        elif field_name == "neck_name":
            recipe_changes.setdefault("components", {}).setdefault("neck", {})["name"] = value
            if value == "gem_pool":
                recipe_changes.setdefault("head_config", {})["pooling_type"] = "gem"
            elif value == "avg_pool":
                recipe_changes.setdefault("head_config", {})["pooling_type"] = "avg"
        elif field_name == "head_name":
            recipe_changes.setdefault("components", {}).setdefault("head", {})["name"] = value
            recipe_changes = _merge_partial_payload(
                recipe_changes,
                _build_head_component_compatibility_payload(str(value)),
            )
        elif field_name == "aux_logits":
            modules_changes["aux_logits"] = value

    if modules_changes:
        recipe_changes["modules"] = modules_changes
    return recipe_changes


def apply_model_recipe_change_payload(
    model_recipe_payload: dict[str, Any],
    recipe_changes: dict[str, Any],
) -> ModelRecipe:
    """Apply a partial structured model_recipe payload onto one base model recipe object."""
    merged_payload = _merge_partial_payload(model_recipe_payload, recipe_changes)
    return ModelRecipe.model_validate(_normalize_model_recipe_components_payload(merged_payload))
