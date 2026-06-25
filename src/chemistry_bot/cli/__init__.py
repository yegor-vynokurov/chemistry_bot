"""CLI-facing package exports for the current bot variants."""

from __future__ import annotations

from .legacy_bot import CliBot as LegacyCliBot
from .rag_bot import CliBot, StudentContext, TeacherContext

__all__ = [
    "CliBot",
    "LegacyCliBot",
    "StudentContext",
    "TeacherContext",
]
