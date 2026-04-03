"""Unified trainer manifest schema and config conversion helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.parameter_space import (
    DatasetRecipe,
    ExperimentConfig,
    ModelRecipe,
    RecipeTaskType,
    SearchPolicy,
    TrainHyp,
)
from app.schemas.ranking_policy import RankingPolicy


class TrainerManifestRuntime(BaseModel):
    """Runtime-only flags that stay outside model and data recipes."""

    use_demo_mode: bool = False
    participates_in_ranking: bool = True


class TrainerManifest(BaseModel):
    """Unified manifest view used by trainer-side parser, validator, and builder."""

    version: str = "trainer_manifest@v1"
    task_type: RecipeTaskType = "classification"
    parameter_space_version: str
    model: ModelRecipe
    train: TrainHyp
    data: DatasetRecipe
    search: SearchPolicy = Field(default_factory=SearchPolicy)
    ranking: RankingPolicy = Field(default_factory=RankingPolicy)
    runtime: TrainerManifestRuntime = Field(default_factory=TrainerManifestRuntime)

    @property
    def model_name(self) -> str:
        """Return the active base model name."""
        return self.model.base_model

    @property
    def model_family(self) -> str:
        """Return the active model family."""
        return self.model.model_family

    @property
    def dataset_name(self) -> str:
        """Return the dataset name declared by the data recipe."""
        return self.data.dataset_name

    @classmethod
    def from_experiment_config(cls, config: ExperimentConfig) -> "TrainerManifest":
        """Project one experiment config into the unified trainer manifest view."""
        return cls(
            task_type=config.task_type,
            parameter_space_version=config.parameter_space_version,
            model=config.model_recipe,
            train=config.train_hyp,
            data=config.dataset_recipe,
            search=config.search_policy,
            ranking=config.ranking_policy,
            runtime=TrainerManifestRuntime(
                use_demo_mode=config.use_demo_mode,
                participates_in_ranking=config.participates_in_ranking,
            ),
        )

    def to_experiment_config(self) -> ExperimentConfig:
        """Convert the unified manifest back into the persisted experiment config shape."""
        aux_logits = bool(self.model.modules.get("aux_logits", False))
        return ExperimentConfig.model_validate(
            {
                "task_type": self.task_type,
                "dataset": self.data.dataset_name,
                "model_family": self.model.model_family,
                "model_name": self.model.base_model,
                "parameter_space_version": self.parameter_space_version,
                "use_demo_mode": self.runtime.use_demo_mode,
                "participates_in_ranking": self.runtime.participates_in_ranking,
                "search_policy": self.search.model_dump(),
                "ranking_policy": self.ranking.model_dump(),
                "params": self.train.to_experiment_params(aux_logits=aux_logits).model_dump(),
                "model_recipe": self.model.model_dump(by_alias=True),
                "train_hyp": self.train.model_dump(),
                "dataset_recipe": self.data.model_dump(),
            }
        )

    @model_validator(mode="after")
    def validate_nested_task_types(self) -> "TrainerManifest":
        """Keep the top-level task type aligned with nested recipe sections."""
        nested_task_types = {
            "model": self.model.task_type,
            "train": self.train.task_type,
            "data": self.data.task_type,
        }
        mismatched_sections = [
            section_name
            for section_name, section_task_type in nested_task_types.items()
            if section_task_type != self.task_type
        ]
        if mismatched_sections:
            mismatched_text = ", ".join(sorted(mismatched_sections))
            raise ValueError(f"Trainer manifest task_type mismatch in sections: {mismatched_text}")
        return self
