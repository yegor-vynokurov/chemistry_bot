"""AnythingLLM Desktop integration for the LangGraph chemistry tutor.

The heavy FastAPI/LangGraph bridge objects are imported lazily. This keeps the
pure conversation, tool-cycle, and context-adapter helpers usable in migration
scripts and unit tests that do not load Ollama dependencies.
"""

from __future__ import annotations

from typing import Any

from .context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
    adapt_anythingllm_tool_results,
)
from .conversation_resolver import (
    AnythingLLMConversationResolver,
    ConversationMessage,
    ConversationResolution,
    ConversationResolutionError,
    ParsedConversation,
    conversation_fingerprint,
    parse_openai_conversation,
)
from .rag_policy import UserDocumentPolicy, should_call_user_document_rag
from .tool_cycle import (
    ToolCycleInspection,
    ToolInteractionLog,
    inspect_tool_cycle,
    interaction_id_from_tool_call_id,
    make_interaction_id,
    make_tool_call_id,
)


def __getattr__(name: str) -> Any:
    if name in {"ProbeSettings", "create_app"}:
        from .app import ProbeSettings, create_app

        return {"ProbeSettings": ProbeSettings, "create_app": create_app}[name]
    if name in {"AnythingLLMRoutingPreview", "RoutingPreview"}:
        from .routing_preview import AnythingLLMRoutingPreview, RoutingPreview

        return {
            "AnythingLLMRoutingPreview": AnythingLLMRoutingPreview,
            "RoutingPreview": RoutingPreview,
        }[name]
    if name in {"TutorBridgeResult", "TutorGraphBridge"}:
        from .tutor_bridge import TutorBridgeResult, TutorGraphBridge

        return {
            "TutorBridgeResult": TutorBridgeResult,
            "TutorGraphBridge": TutorGraphBridge,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AdaptedRAGContext",
    "AnythingLLMConversationResolver",
    "AnythingLLMRoutingPreview",
    "ConversationMessage",
    "ConversationResolution",
    "ConversationResolutionError",
    "DocumentContextChunk",
    "ParsedConversation",
    "ProbeSettings",
    "RoutingPreview",
    "ToolCycleInspection",
    "ToolInteractionLog",
    "TutorBridgeResult",
    "TutorGraphBridge",
    "UserDocumentPolicy",
    "adapt_anythingllm_tool_results",
    "conversation_fingerprint",
    "create_app",
    "inspect_tool_cycle",
    "interaction_id_from_tool_call_id",
    "make_interaction_id",
    "make_tool_call_id",
    "parse_openai_conversation",
    "should_call_user_document_rag",
]
