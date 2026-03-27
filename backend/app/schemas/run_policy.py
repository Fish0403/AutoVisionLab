"""Run-level policy schemas."""

from pydantic import BaseModel, Field


class RunPolicy(BaseModel):
    """Centralized hard constraints for one run."""

    min_top1_acc_promotion_delta: float = Field(default=0.01, ge=0)
    top1_acc_parity_epsilon: float = Field(default=0.0005, ge=0)
    min_val_loss_promotion_delta: float = Field(default=0.01, ge=0)
    default_max_changed_fields: int = Field(default=1, ge=1)
    max_changed_fields_after_stagnation: int = Field(default=2, ge=1)
    stagnation_rounds_for_combined_changes: int = Field(default=2, ge=0)
    consecutive_failures_before_field_cooldown: int = Field(default=2, ge=1)
    field_cooldown_rounds: int = Field(default=2, ge=1)
    stagnation_rounds_for_dimension_switch: int = Field(default=3, ge=1)
    auto_train_non_basic_change_after_round_ratio: float = Field(default=0.5, ge=0, le=1)
