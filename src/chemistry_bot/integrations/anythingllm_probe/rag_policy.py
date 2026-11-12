"""Pure policy for deciding when AnythingLLM user documents should be searched."""

from __future__ import annotations

from typing import Literal

from chemistry_bot.tutor_graph.schemas import QuestionClassification, RoutingDecision


UserDocumentPolicy = Literal["classifier", "always", "never"]


def should_call_user_document_rag(
    *,
    classification: QuestionClassification,
    routing: RoutingDecision,
    rag_tool_available: bool,
    policy: UserDocumentPolicy,
) -> tuple[bool, str]:
    if not rag_tool_available:
        return False, "AnythingLLM did not expose the rag-memory tool."
    if policy == "never":
        return False, "User-document retrieval is disabled by configuration."
    if routing.route in {"safety_redirect", "off_topic", "clarification"}:
        return False, f"Route {routing.route!r} should answer without user-document retrieval."
    if policy == "always":
        return True, "User-document retrieval is forced by configuration."
    if classification.needs_documents_dev:
        return True, "Developer override needs_documents_dev forced user-document retrieval."
    if classification.needs_user_documents:
        return True, "Python detected an explicit request to use uploaded documents."
    return False, "No explicit user-document request was detected."


__all__ = ["UserDocumentPolicy", "should_call_user_document_rag"]
