"""Proposal generation services."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.aihubmix_client import AIHubMixClient
from app.models.experiment import ExperimentModel
from app.models.run import RunModel
from app.schemas.ai import ProposalChanges, ProposalSchema
from app.services.parameter_space import AI_BLOCKED_PROPOSAL_FIELDS, validate_proposal_against_space


def get_latest_experiment_payload(db: Session, run_id: str) -> dict | None:
    """Return the latest experiment payload for a run."""
    experiment = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.desc())
    ).first()
    if experiment is None:
        return None
    return {
        "id": experiment.id,
        "status": experiment.status,
        "config": experiment.experiment_config,
        "result": experiment.result,
        "proposal": experiment.proposal,
    }


def sanitize_blocked_proposal_fields(proposal: ProposalSchema) -> ProposalSchema:
    """Clear blocked AI-controlled fields instead of failing the whole proposal."""
    changes_payload = proposal.changes.model_dump()
    sanitized_changes = {
        key: (None if key in AI_BLOCKED_PROPOSAL_FIELDS else value)
        for key, value in changes_payload.items()
    }
    return proposal.model_copy(update={"changes": ProposalChanges.model_validate(sanitized_changes)})


def generate_aihubmix_proposal(db: Session, run_id: str) -> ProposalSchema:
    """Generate a structured proposal using AIHubMix."""
    run = db.get(RunModel, run_id)
    if run is None:
        raise ValueError("Run not found")

    latest_experiment = get_latest_experiment_payload(db, run_id)
    if latest_experiment is None:
        raise ValueError("No experiment is available for this run")

    system_prompt = (
        "You are generating the next classification training proposal. "
        "Return JSON only. "
        "You must follow this exact schema: "
        '{"task_type":"classification","model_name":"string","based_on_experiment_ids":["string"],'
        '"hypothesis":"string","changes":{"optimizer":"string|null","learning_rate":"number|null",'
        '"batch_size":"number|null","image_size":"number|null","epochs":"number|null","weight_decay":"number|null",'
        '"scheduler":"string|null","augmentation_level":"string|null","label_smoothing":"number|null","aux_logits":"boolean|null"},'
        '"reason":"string","risk":"low|medium|high"}'
    )
    user_prompt = (
        f"Run:\n{json.dumps({'id': run.id, 'name': run.name, 'dataset': run.dataset, 'model_name': run.model_name}, ensure_ascii=True)}\n"
        f"Latest experiment:\n{json.dumps(latest_experiment, ensure_ascii=True)}\n"
        "Generate the next proposal for the same run. "
        "Keep task_type as classification. "
        "Do not change model_name. "
        "Do not modify epochs; epochs is fixed and cannot be tuned by AI. "
        "Only propose structured parameter changes."
    )
    client = AIHubMixClient()
    proposal_payload = client.create_json_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    proposal = ProposalSchema.model_validate(proposal_payload)
    if proposal.model_name != run.model_name:
        raise ValueError("Proposal model_name does not match the run model")
    proposal = sanitize_blocked_proposal_fields(proposal)
    if not validate_proposal_against_space(proposal):
        raise ValueError("Proposal contains blocked parameter changes")
    return proposal


def test_aihubmix_connection() -> dict:
    """Run a minimal AIHubMix connectivity test."""
    client = AIHubMixClient()
    return client.create_json_completion(
        system_prompt='Return JSON only: {"ok": true, "message": "string"}',
        user_prompt="Reply with a tiny JSON object confirming connectivity.",
    )
