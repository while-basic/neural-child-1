"""Developmental stages package.

This package provides functionality for managing developmental stages
and related features in the Neural Child project.
"""

from .stages import DevelopmentalStage, DevelopmentalSystem
from .prompts import get_stage_prompt, STAGE_PROMPTS

__all__ = [
    'DevelopmentalStage',
    'DevelopmentalSystem',
    'get_stage_prompt',
    'STAGE_PROMPTS'
] 