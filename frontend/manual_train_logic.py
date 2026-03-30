"""Helpers for deciding manual training behavior in the Streamlit UI."""

from __future__ import annotations

from typing import Literal


ManualTrainAction = Literal["create_run", "append_experiment"]


def resolve_manual_train_action(
    *,
    selected_run_id: str,
    selected_run_model_name: str | None,
    requested_model_name: str,
) -> ManualTrainAction:
    """Decide whether one manual train request should create a run or append an experiment."""
    if selected_run_id == "__all__":
        return "create_run"
    if selected_run_model_name != requested_model_name:
        return "create_run"
    return "append_experiment"
