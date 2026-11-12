"""Standalone LangGraph chemistry tutor bot.

This package is intentionally independent from ``chemistry_bot.cli.rag_bot``.
The current RAG bot can change, break, or disappear without affecting this
experimental tutor graph package.

Imports are intentionally lazy: storage helpers can be imported without loading
LangChain/LangGraph dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "QuestionClassification",
    "RoutingDecision",
    "TutorAnswer",
    "TutorGraphConfig",
    "build_tutor_graph",
]


def __getattr__(name: str) -> Any:
    if name in {"QuestionClassification", "RoutingDecision", "TutorAnswer"}:
        from .schemas import QuestionClassification, RoutingDecision, TutorAnswer

        return {
            "QuestionClassification": QuestionClassification,
            "RoutingDecision": RoutingDecision,
            "TutorAnswer": TutorAnswer,
        }[name]
    if name in {"TutorGraphConfig", "build_tutor_graph"}:
        from .graph import TutorGraphConfig, build_tutor_graph

        return {"TutorGraphConfig": TutorGraphConfig, "build_tutor_graph": build_tutor_graph}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
