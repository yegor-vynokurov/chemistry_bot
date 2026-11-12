"""LangGraph state definitions for the standalone tutor graph."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from .schemas import QuestionClassification, RoutingDecision, TutorAnswer


class TutorState(TypedDict, total=False):
    """State passed between graph nodes.

    The graph state stays small and explicit. Durable storage is handled outside
    the graph through tutor_graph storage/tools modules.
    """

    student_id: str
    session_id: str
    user_text: str
    messages: list[BaseMessage]
    session_state: dict[str, Any]
    external_context: dict[str, Any] | None
    classification: QuestionClassification | None
    routing: RoutingDecision | None
    response: TutorAnswer | None
    error: str | None
    debug: dict[str, Any]
