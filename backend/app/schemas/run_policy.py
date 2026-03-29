"""Run-level policy schemas."""

from pydantic import BaseModel, Field


class RunPolicy(BaseModel):
    """Centralized hard constraints for one run."""

    default_max_changed_fields: int = Field(default=1, ge=1)
    max_changed_fields_after_stagnation: int = Field(default=2, ge=1)
    stagnation_rounds_for_combined_changes: int = Field(default=2, ge=0)
    consecutive_failures_before_field_cooldown: int = Field(default=2, ge=1)
    field_cooldown_rounds: int = Field(default=2, ge=1)
    stagnation_rounds_for_dimension_switch: int = Field(default=3, ge=1)
    auto_train_non_basic_change_after_round: int = Field(default=3, ge=0)
    auto_train_early_stop_stagnation_rounds: int = Field(default=6, ge=1)
    auto_train_min_successful_attempts_per_dimension: int = Field(default=2, ge=1)
