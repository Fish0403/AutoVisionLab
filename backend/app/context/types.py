"""Constants used by proposal context builders."""

from __future__ import annotations

from typing import Final, Literal, get_args

from app.schemas.prompt_context import ChangeType, PromptBlockName


PromptMode = Literal["proposal"]

DEFAULT_STAGE_HISTORY_KEEP: Final[int] = 10
DEFAULT_STAGE_COMPACT_THRESHOLD: Final[int] = 20

PROPOSAL_BLOCK_NAMES: Final[tuple[PromptBlockName, ...]] = get_args(PromptBlockName)
CHANGE_TYPES: Final[tuple[ChangeType, ...]] = get_args(ChangeType)
