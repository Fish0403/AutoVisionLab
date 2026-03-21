"""Parameter space access and validation helpers."""

from app.config_spaces.classification import (
    GOOGLENET_PARAMETER_SPACE,
    MOBILENET_V2_PARAMETER_SPACE,
)
from app.schemas.ai import ProposalSchema
from app.schemas.parameter_space import EditableParameterSpace


PARAMETER_SPACES = {
    "mobilenet_v2": MOBILENET_V2_PARAMETER_SPACE,
    "googlenet": GOOGLENET_PARAMETER_SPACE,
}

AI_BLOCKED_PROPOSAL_FIELDS = {"epochs"}


def get_parameter_space(model_name: str) -> EditableParameterSpace | None:
    """Return the declared parameter space for a supported model."""
    return PARAMETER_SPACES.get(model_name)


def validate_proposal_against_space(proposal: ProposalSchema) -> bool:
    """Reserve a strict white-list validation entrypoint."""
    if proposal.model_name not in PARAMETER_SPACES:
        return False
    proposal_changes = proposal.changes.model_dump()
    return all(proposal_changes.get(field_name) is None for field_name in AI_BLOCKED_PROPOSAL_FIELDS)
