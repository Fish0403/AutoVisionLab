"""Parameter space access and validation helpers."""

from app.model_catalog.registry import get_model_catalog_entry
from app.schemas.ai import ProposalSchema
from app.schemas.parameter_space import (
    DiscreteValuesParamDefinition,
    EditableParameterSpace,
    EnumParamDefinition,
    NumberRangeParamDefinition,
    SearchPolicy,
)

AI_BLOCKED_PROPOSAL_FIELDS: set[str] = set()
BASIC_HPARAM_SEARCH_FIELDS = {
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
    "scheduler",
    "label_smoothing",
    "image_size",
}
STRATEGY_SEARCH_FIELDS: set[str] = set()
LOSS_SEARCH_FIELDS = {"loss_name", "focal_gamma"}
AUGMENTATION_SEARCH_FIELDS = {"mixup_alpha", "cutmix_alpha", "random_erasing_prob"}
MODEL_MODULE_SEARCH_FIELDS = {
    "aux_logits",
    "neck_name",
    "head_name",
}


def get_parameter_space(model_name: str) -> EditableParameterSpace | None:
    """Return the declared parameter space for a supported model."""
    model_entry = get_model_catalog_entry(model_name)
    if model_entry is None:
        return None
    return model_entry.parameter_space


def get_allowed_ai_search_fields(
    search_policy: SearchPolicy | None,
    *,
    parameter_space: EditableParameterSpace | None = None,
) -> set[str]:
    """Return the effective field-level AI search white-list."""
    policy = search_policy or SearchPolicy()
    allowed_fields: set[str] = set()
    if policy.allow_basic_hparam_search:
        allowed_fields.update(set(policy.allowed_basic_hparam_fields) & BASIC_HPARAM_SEARCH_FIELDS)
    if policy.allow_strategy_search:
        allowed_fields.update(STRATEGY_SEARCH_FIELDS)
    if policy.allow_loss_search:
        allowed_fields.update(LOSS_SEARCH_FIELDS)
    if policy.allow_augmentation_search:
        allowed_fields.update(AUGMENTATION_SEARCH_FIELDS)
    if policy.allow_model_module_search:
        if parameter_space is not None:
            allowed_fields.update(MODEL_MODULE_SEARCH_FIELDS & set(parameter_space.editable_params.keys()))
        else:
            allowed_fields.update(MODEL_MODULE_SEARCH_FIELDS)
    return allowed_fields - AI_BLOCKED_PROPOSAL_FIELDS


def is_epoch_search_enabled(search_policy: SearchPolicy | None) -> bool:
    """Return whether the current policy explicitly allows epoch search."""
    policy = search_policy or SearchPolicy()
    return policy.allow_basic_hparam_search and "epochs" in set(policy.allowed_basic_hparam_fields)


def _is_value_allowed_by_definition(value: object, definition: object) -> bool:
    if isinstance(definition, EnumParamDefinition):
        return value in definition.choices
    if isinstance(definition, DiscreteValuesParamDefinition):
        return value in definition.choices
    if isinstance(definition, NumberRangeParamDefinition):
        if not isinstance(value, (int, float)):
            return False
        return definition.min <= float(value) <= definition.max
    return False


def explain_proposal_rejection(
    proposal: ProposalSchema,
    search_policy: SearchPolicy | None = None,
    parameter_space: EditableParameterSpace | None = None,
) -> str | None:
    """Return a human-readable rejection reason, or None if the proposal is valid."""
    effective_parameter_space = parameter_space or get_parameter_space(proposal.model_name)
    if effective_parameter_space is None:
        return f"parameter space is missing for model {proposal.model_name}"

    allowed_fields = get_allowed_ai_search_fields(search_policy, parameter_space=effective_parameter_space)
    proposal_changes = proposal.changes.model_dump()
    changed_fields = {field_name: value for field_name, value in proposal_changes.items() if value is not None}
    if any(field_name in AI_BLOCKED_PROPOSAL_FIELDS for field_name in changed_fields):
        blocked_fields = sorted(field_name for field_name in changed_fields if field_name in AI_BLOCKED_PROPOSAL_FIELDS)
        return f"blocked fields: {', '.join(blocked_fields)}"

    for field_name, value in changed_fields.items():
        if field_name not in allowed_fields:
            return f"field not allowed by search policy: {field_name}"
        definition = effective_parameter_space.editable_params.get(field_name)
        if definition is None:
            return f"field not declared in parameter space: {field_name}"
        if not _is_value_allowed_by_definition(value, definition):
            return f"value out of allowed range for {field_name}: {value}"
    return None
