"""Parameter space access and validation helpers."""

from app.config_spaces.classification import (
    DENSENET121_PARAMETER_SPACE,
    GOOGLENET_PARAMETER_SPACE,
    MOBILENET_V2_PARAMETER_SPACE,
    RESNET18_PARAMETER_SPACE,
    RESNET34_PARAMETER_SPACE,
)
from app.schemas.ai import ProposalSchema
from app.schemas.parameter_space import (
    DiscreteValuesParamDefinition,
    EditableParameterSpace,
    EnumParamDefinition,
    NumberRangeParamDefinition,
    SearchPolicy,
)


PARAMETER_SPACES = {
    "mobilenet_v2": MOBILENET_V2_PARAMETER_SPACE,
    "googlenet": GOOGLENET_PARAMETER_SPACE,
    "resnet18": RESNET18_PARAMETER_SPACE,
    "resnet34": RESNET34_PARAMETER_SPACE,
    "densenet121": DENSENET121_PARAMETER_SPACE,
}

AI_BLOCKED_PROPOSAL_FIELDS = {"epochs"}
BASIC_HPARAM_SEARCH_FIELDS = {
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "scheduler",
    "label_smoothing",
    "image_size",
}
STRATEGY_SEARCH_FIELDS = {"aux_logits"}
LOSS_SEARCH_FIELDS = {"loss_name", "focal_gamma"}
AUGMENTATION_SEARCH_FIELDS = {"augmentation_policy", "mixup_alpha", "cutmix_alpha", "random_erasing_prob"}


def get_parameter_space(model_name: str) -> EditableParameterSpace | None:
    """Return the declared parameter space for a supported model."""
    return PARAMETER_SPACES.get(model_name)


def get_allowed_ai_search_fields(search_policy: SearchPolicy | None) -> set[str]:
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
    return allowed_fields - AI_BLOCKED_PROPOSAL_FIELDS


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


def validate_proposal_against_space(
    proposal: ProposalSchema,
    search_policy: SearchPolicy | None = None,
    parameter_space: EditableParameterSpace | None = None,
) -> bool:
    """Validate proposal fields against both policy and parameter space."""
    return explain_proposal_rejection(proposal, search_policy, parameter_space) is None


def explain_proposal_rejection(
    proposal: ProposalSchema,
    search_policy: SearchPolicy | None = None,
    parameter_space: EditableParameterSpace | None = None,
) -> str | None:
    """Return a human-readable rejection reason, or None if the proposal is valid."""
    effective_parameter_space = parameter_space or PARAMETER_SPACES.get(proposal.model_name)
    if effective_parameter_space is None:
        return f"parameter space is missing for model {proposal.model_name}"

    allowed_fields = get_allowed_ai_search_fields(search_policy)
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
