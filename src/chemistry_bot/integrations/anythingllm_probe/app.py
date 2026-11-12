"""OpenAI-compatible AnythingLLM Desktop bridge for the LangGraph tutor.

I0.3.1B.4 keeps the proven conversation resolver and two-request
``rag-memory`` cycle, but replaces diagnostic answers with real LangGraph tutor
answers. The tutor model now returns plain text only; Python assembles TutorAnswer and an optional small-model check question. A lightweight classifier runs before retrieval. When it requests local
RAG, AnythingLLM searches the user's workspace documents and returns a standard
``role=tool`` message. The result is adapted into a bounded, source-tagged,
untrusted context block before the graph sees it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Mapping
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chemistry_bot.tutor_graph.graph import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_TUTOR_MODEL,
    TutorGraphConfig,
)
from chemistry_bot.tutor_graph.storage import DEFAULT_DB_PATH, SQLiteTutorStore
from chemistry_bot.tutor_graph.semantic_safety import LazyOllamaEmbeddingProvider
from chemistry_bot.tutor_graph.schemas import TutorAnswer

from .context_adapter import AdaptedRAGContext, adapt_anythingllm_tool_results
from .model_catalog import (
    OllamaCatalogUnavailable,
    OllamaModelCatalog,
    TutorModelUnavailable,
    load_tutor_model_configuration,
)
from .conversation_resolver import (
    DEFAULT_MAX_CANDIDATE_SESSIONS,
    DEFAULT_MAX_MATCH_MESSAGES,
    DEFAULT_STUDENT_ID,
    AnythingLLMConversationResolver,
    ConversationResolution,
    ParsedConversation,
    parse_openai_conversation,
)
from .rag_query_expansion import (
    RAGQueryExpansionConfig,
    RAGQueryExpansionResult,
    choose_query_expansion,
)
from .rag_relevance_gate import (
    RAGRelevanceGateConfig,
    RAGRelevanceGateResult,
    apply_rag_relevance_gate,
)
from .request_log import RequestLogSettings, RequestProbeLog, sha256_json, utc_now_iso
from .routing_preview import (
    AnythingLLMRoutingPreview,
    RoutingPreview,
    UserDocumentPolicy,
)
from .tool_cycle import (
    DEFAULT_RAG_TOOL_NAME,
    InteractionLogSettings,
    ToolInteractionLog,
    inspect_tool_cycle,
    make_interaction_id,
    make_tool_call_id,
    make_retry_tool_call_id,
    parsed_conversation_to_dict,
)
from .tutor_bridge import DEFAULT_CONTEXT_TURNS, TutorBridgeResult, TutorGraphBridge


DEFAULT_MODEL_ID = "chemistry-tutor"
DEFAULT_LOG_DIR = Path("data/anythingllm_probe")
DEFAULT_MAX_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_RAG_CHUNKS = 6
DEFAULT_MAX_RAG_CONTEXT_CHARS = 14000
DEFAULT_TUTOR_MODEL_OPTIONS, _DEFAULT_SUPPORT_MODELS = load_tutor_model_configuration()
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"


def _environment_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _environment_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, min(maximum, float(raw)))
    except ValueError:
        return default


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _environment_policy(name: str, default: UserDocumentPolicy) -> UserDocumentPolicy:
    raw = os.getenv(name, default).strip().lower()
    return raw if raw in {"classifier", "always", "never"} else default  # type: ignore[return-value]


@dataclass(frozen=True)
class ProbeSettings:
    """Runtime settings for the local AnythingLLM tutor bridge."""

    model_id: str = DEFAULT_MODEL_ID
    log_dir: Path = DEFAULT_LOG_DIR
    log_content: bool = True
    api_key: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    tutor_model_options: tuple[str, ...] = DEFAULT_TUTOR_MODEL_OPTIONS
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    resolve_conversations: bool = True
    tutor_db_path: Path = DEFAULT_DB_PATH
    student_id: str = DEFAULT_STUDENT_ID
    max_candidate_sessions: int = DEFAULT_MAX_CANDIDATE_SESSIONS
    max_match_messages: int = DEFAULT_MAX_MATCH_MESSAGES
    rag_tool_probe: bool = True
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME
    user_document_policy: UserDocumentPolicy = "classifier"
    needs_documents_dev: bool = False
    max_rag_chunks: int = DEFAULT_MAX_RAG_CHUNKS
    max_rag_context_chars: int = DEFAULT_MAX_RAG_CONTEXT_CHARS
    rag_relevance_gate_enabled: bool = True
    rag_relevance_embedding_model_name: str = "embeddinggemma:latest"
    rag_relevance_medium_threshold: float = 0.45
    rag_relevance_high_threshold: float = 0.58
    rag_relevance_top_k: int = 2
    rag_query_expansion_enabled: bool = True
    rag_query_expansion_retry_enabled: bool = True
    rag_query_expansion_max_added_terms_light: int = 8
    rag_query_expansion_max_added_terms_targeted: int = 14
    tutor_model_name: str = DEFAULT_TUTOR_MODEL
    tutor_temperature: float = 0.0
    tutor_num_ctx: int = 8192
    tutor_num_predict: int = 1536
    classifier_enabled: bool = True
    classifier_model_name: str = DEFAULT_CLASSIFIER_MODEL
    classifier_temperature: float = 0.0
    classification_confidence_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD
    classifier_fallback_to_rules: bool = True
    turn_relation_enabled: bool = True
    turn_relation_model_name: str = DEFAULT_CLASSIFIER_MODEL
    turn_relation_temperature: float = 0.0
    turn_relation_num_ctx: int = 2048
    turn_relation_fallback_to_rules: bool = True
    semantic_safety_enabled: bool = True
    safety_embedding_model_name: str = "nomic-embed-text"
    safety_hazard_threshold: float = 0.60
    safety_procedure_threshold: float = 0.60
    safety_fuzzy_threshold: float = 86.0
    followup_enabled: bool = True
    followup_model_name: str = DEFAULT_CLASSIFIER_MODEL
    followup_temperature: float = 0.0
    context_turns: int = DEFAULT_CONTEXT_TURNS

    @classmethod
    def from_environment(cls) -> "ProbeSettings":
        return cls(
            model_id=os.getenv("CHEM_TUTOR_PROBE_MODEL_ID", DEFAULT_MODEL_ID).strip()
            or DEFAULT_MODEL_ID,
            log_dir=Path(os.getenv("CHEM_TUTOR_PROBE_LOG_DIR", str(DEFAULT_LOG_DIR))),
            log_content=_environment_bool("CHEM_TUTOR_PROBE_LOG_CONTENT", True),
            api_key=os.getenv("CHEM_TUTOR_PROBE_API_KEY") or None,
            ollama_base_url=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/"),
            tutor_model_options=tuple(
                dict.fromkeys(
                    model.strip()
                    for model in os.getenv(
                        "CHEM_TUTOR_MODEL_OPTIONS", ",".join(DEFAULT_TUTOR_MODEL_OPTIONS)
                    ).split(",")
                    if model.strip()
                )
            ) or DEFAULT_TUTOR_MODEL_OPTIONS,
            max_body_bytes=_environment_int(
                "CHEM_TUTOR_PROBE_MAX_BODY_BYTES",
                DEFAULT_MAX_BODY_BYTES,
                minimum=1024,
            ),
            resolve_conversations=_environment_bool(
                "CHEM_TUTOR_PROBE_RESOLVE_CONVERSATIONS", True
            ),
            tutor_db_path=Path(os.getenv("CHEM_TUTOR_DB_PATH", str(DEFAULT_DB_PATH))),
            student_id=os.getenv("CHEM_TUTOR_STUDENT_ID", DEFAULT_STUDENT_ID).strip()
            or DEFAULT_STUDENT_ID,
            max_candidate_sessions=_environment_int(
                "CHEM_TUTOR_MAX_CANDIDATE_SESSIONS",
                DEFAULT_MAX_CANDIDATE_SESSIONS,
                minimum=1,
            ),
            max_match_messages=_environment_int(
                "CHEM_TUTOR_MAX_MATCH_MESSAGES",
                DEFAULT_MAX_MATCH_MESSAGES,
                minimum=2,
            ),
            rag_tool_probe=_environment_bool("CHEM_TUTOR_RAG_TOOL_PROBE", True),
            rag_tool_name=os.getenv("CHEM_TUTOR_RAG_TOOL_NAME", DEFAULT_RAG_TOOL_NAME).strip()
            or DEFAULT_RAG_TOOL_NAME,
            user_document_policy=_environment_policy(
                "CHEM_TUTOR_USER_DOCUMENT_POLICY", "classifier"
            ),
            needs_documents_dev=_environment_bool(
                "CHEM_TUTOR_NEEDS_DOCUMENTS_DEV", False
            ),
            max_rag_chunks=_environment_int(
                "CHEM_TUTOR_MAX_RAG_CHUNKS", DEFAULT_MAX_RAG_CHUNKS, minimum=1
            ),
            max_rag_context_chars=_environment_int(
                "CHEM_TUTOR_MAX_RAG_CONTEXT_CHARS",
                DEFAULT_MAX_RAG_CONTEXT_CHARS,
                minimum=1000,
            ),
            rag_relevance_gate_enabled=_environment_bool(
                "CHEM_TUTOR_RAG_RELEVANCE_GATE_ENABLED", True
            ),
            rag_relevance_embedding_model_name=os.getenv(
                "CHEM_TUTOR_RAG_RELEVANCE_EMBEDDING_MODEL", "embeddinggemma:latest"
            ).strip() or "embeddinggemma:latest",
            rag_relevance_medium_threshold=_environment_float(
                "CHEM_TUTOR_RAG_RELEVANCE_MEDIUM_THRESHOLD",
                0.45,
                minimum=-1.0,
                maximum=1.0,
            ),
            rag_relevance_high_threshold=_environment_float(
                "CHEM_TUTOR_RAG_RELEVANCE_HIGH_THRESHOLD",
                0.58,
                minimum=-1.0,
                maximum=1.0,
            ),
            rag_relevance_top_k=_environment_int(
                "CHEM_TUTOR_RAG_RELEVANCE_TOP_K", 2, minimum=1
            ),
            rag_query_expansion_enabled=_environment_bool(
                "CHEM_TUTOR_RAG_QUERY_EXPANSION_ENABLED", True
            ),
            rag_query_expansion_retry_enabled=_environment_bool(
                "CHEM_TUTOR_RAG_QUERY_EXPANSION_RETRY_ENABLED", True
            ),
            rag_query_expansion_max_added_terms_light=_environment_int(
                "CHEM_TUTOR_RAG_QUERY_EXPANSION_MAX_ADDED_TERMS_LIGHT", 8, minimum=0
            ),
            rag_query_expansion_max_added_terms_targeted=_environment_int(
                "CHEM_TUTOR_RAG_QUERY_EXPANSION_MAX_ADDED_TERMS_TARGETED", 14, minimum=0
            ),
            tutor_model_name=os.getenv("CHEM_TUTOR_MODEL", DEFAULT_TUTOR_MODEL).strip()
            or DEFAULT_TUTOR_MODEL,
            tutor_temperature=_environment_float(
                "CHEM_TUTOR_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
            ),
            tutor_num_ctx=_environment_int(
                "CHEM_TUTOR_NUM_CTX", 8192, minimum=1024
            ),
            tutor_num_predict=_environment_int(
                "CHEM_TUTOR_NUM_PREDICT", 1536, minimum=128
            ),
            classifier_enabled=_environment_bool("CHEM_TUTOR_CLASSIFIER_ENABLED", True),
            classifier_model_name=os.getenv(
                "CHEM_TUTOR_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL
            ).strip()
            or DEFAULT_CLASSIFIER_MODEL,
            classifier_temperature=_environment_float(
                "CHEM_TUTOR_CLASSIFIER_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
            ),
            classification_confidence_threshold=_environment_float(
                "CHEM_TUTOR_CLASSIFICATION_THRESHOLD",
                DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
                minimum=0.0,
                maximum=1.0,
            ),
            classifier_fallback_to_rules=_environment_bool(
                "CHEM_TUTOR_CLASSIFIER_FALLBACK_TO_RULES", True
            ),
            turn_relation_enabled=_environment_bool(
                "CHEM_TUTOR_TURN_RELATION_ENABLED", True
            ),
            turn_relation_model_name=(
                os.getenv("CHEM_TUTOR_TURN_RELATION_MODEL")
                or os.getenv("CHEM_TUTOR_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)
            ).strip() or DEFAULT_CLASSIFIER_MODEL,
            turn_relation_temperature=_environment_float(
                "CHEM_TUTOR_TURN_RELATION_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
            ),
            turn_relation_num_ctx=_environment_int(
                "CHEM_TUTOR_TURN_RELATION_NUM_CTX", 2048, minimum=512
            ),
            turn_relation_fallback_to_rules=_environment_bool(
                "CHEM_TUTOR_TURN_RELATION_FALLBACK_TO_RULES", True
            ),
            semantic_safety_enabled=_environment_bool(
                "CHEM_TUTOR_SEMANTIC_SAFETY_ENABLED", True
            ),
            safety_embedding_model_name=os.getenv(
                "CHEM_TUTOR_SAFETY_EMBEDDING_MODEL", "nomic-embed-text"
            ).strip() or "nomic-embed-text",
            safety_hazard_threshold=_environment_float(
                "CHEM_TUTOR_SAFETY_HAZARD_THRESHOLD", 0.60, minimum=-1.0, maximum=1.0
            ),
            safety_procedure_threshold=_environment_float(
                "CHEM_TUTOR_SAFETY_PROCEDURE_THRESHOLD", 0.60, minimum=-1.0, maximum=1.0
            ),
            safety_fuzzy_threshold=_environment_float(
                "CHEM_TUTOR_SAFETY_FUZZY_THRESHOLD", 86.0, minimum=0.0, maximum=100.0
            ),
            followup_enabled=_environment_bool("CHEM_TUTOR_FOLLOWUP_ENABLED", True),
            followup_model_name=(
                os.getenv("CHEM_TUTOR_FOLLOWUP_MODEL")
                or os.getenv("CHEM_TUTOR_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)
            ).strip() or DEFAULT_CLASSIFIER_MODEL,
            followup_temperature=_environment_float(
                "CHEM_TUTOR_FOLLOWUP_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
            ),
            context_turns=_environment_int(
                "CHEM_TUTOR_CONTEXT_TURNS", DEFAULT_CONTEXT_TURNS, minimum=1
            ),
        )


def _openai_error(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key")


def _require_api_key(request: Request, settings: ProbeSettings) -> None:
    if settings.api_key is not None and _extract_bearer_token(request) != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _validate_chat_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("The 'messages' field must be a non-empty JSON array.")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be a JSON object.")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"messages[{index}].role must be a string.")
        assistant_tool_call = role == "assistant" and isinstance(message.get("tool_calls"), list)
        if "content" not in message and not assistant_tool_call:
            raise ValueError(
                f"messages[{index}] must contain content unless it is an assistant tool-call message."
            )
    return body


def _usage_estimate(body: dict[str, Any], response_text: str) -> dict[str, int]:
    prompt_chars = len(json.dumps(body.get("messages", []), ensure_ascii=False, default=str))
    completion_chars = len(response_text)
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = max(1, completion_chars // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _non_stream_text_completion(
    *, request_id: str, model: str, created: int, response_text: str, body: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_estimate(body, response_text),
        "system_fingerprint": "chemistry-tutor-i0.3.1b3",
    }


def _non_stream_tool_call_completion(
    *,
    request_id: str,
    model: str,
    created: int,
    tool_call_id: str,
    tool_name: str,
    arguments_json: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments_json},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": _usage_estimate(body, arguments_json),
        "system_fingerprint": "chemistry-tutor-i0.3.1b3",
    }


async def _stream_text_completion(
    *,
    request_id: str,
    model: str,
    created: int,
    response_text: str,
    include_usage: bool,
    body: dict[str, Any],
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{request_id}"

    def event(
        delta: dict[str, Any],
        finish_reason: str | None = None,
        *,
        choices: bool = True,
        usage: dict[str, int] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                if choices
                else []
            ),
            "system_fingerprint": "chemistry-tutor-i0.3.1b3",
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"

    yield event({"role": "assistant", "content": ""})
    await asyncio.sleep(0)
    # I0.3.1B.4 obtains the programmatically assembled answer before sending it as one
    # SSE content delta, which AnythingLLM renders correctly.
    yield event({"content": response_text})
    await asyncio.sleep(0)
    yield event({}, finish_reason="stop")
    if include_usage:
        yield event({}, choices=False, usage=_usage_estimate(body, response_text))
    yield "data: [DONE]\n\n"


async def _stream_tool_call_completion(
    *,
    request_id: str,
    model: str,
    created: int,
    tool_call_id: str,
    tool_name: str,
    arguments_json: str,
    include_usage: bool,
    body: dict[str, Any],
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{request_id}"

    def event(
        delta: dict[str, Any],
        finish_reason: str | None = None,
        *,
        choices: bool = True,
        usage: dict[str, int] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                if choices
                else []
            ),
            "system_fingerprint": "chemistry-tutor-i0.3.1b3",
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"

    yield event({"role": "assistant"})
    await asyncio.sleep(0)
    yield event(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": ""},
                }
            ]
        }
    )
    await asyncio.sleep(0)
    yield event(
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments_json}}]}
    )
    await asyncio.sleep(0)
    yield event({}, finish_reason="tool_calls")
    if include_usage:
        yield event({}, choices=False, usage=_usage_estimate(body, arguments_json))
    yield "data: [DONE]\n\n"


def _stream_include_usage(body: Mapping[str, Any]) -> bool:
    options = body.get("stream_options")
    return isinstance(options, dict) and bool(options.get("include_usage"))


def _interaction_resolution(interaction: Mapping[str, Any] | None) -> ConversationResolution | None:
    if not isinstance(interaction, Mapping) or not isinstance(interaction.get("resolution"), dict):
        return None
    try:
        return ConversationResolution.from_dict(interaction["resolution"])
    except (KeyError, TypeError, ValueError):
        return None


def _interaction_parsed_conversation(interaction: Mapping[str, Any] | None) -> ParsedConversation | None:
    if not isinstance(interaction, Mapping) or not isinstance(
        interaction.get("parsed_conversation"), dict
    ):
        return None
    try:
        return ParsedConversation.from_dict(interaction["parsed_conversation"])
    except (TypeError, ValueError):
        return None


def _interaction_routing_preview(interaction: Mapping[str, Any] | None) -> RoutingPreview | None:
    if not isinstance(interaction, Mapping) or not isinstance(
        interaction.get("routing_preview"), dict
    ):
        return None
    try:
        return RoutingPreview.from_dict(interaction["routing_preview"])
    except (KeyError, TypeError, ValueError):
        return None


def _safe_bridge_error_text() -> str:
    return (
        "The local chemistry tutor could not finish this response. "
        "Please check the tutor service log and confirm that Ollama and the configured models are running."
    )


def _explicit_document_request(routing_preview: RoutingPreview | None) -> bool:
    if routing_preview is None:
        return False
    return bool(
        routing_preview.classification.needs_user_documents
        and not routing_preview.classification.needs_documents_dev
        and routing_preview.user_document_policy != "always"
    )


def _document_not_found_text() -> str:
    return (
        "I do not see a direct answer in the retrieved book fragments, so I should not "
        "present this as book-supported. Try adding the exact term, chapter, page, or a "
        "screenshot, and I can check the document evidence again."
    )


def _deterministic_document_not_found_answer() -> TutorAnswer:
    return TutorAnswer(
        tutor_move="direct_answer",
        visible_answer=_document_not_found_text(),
        next_question=None,
        concept_ids=["document_retrieval"],
        safety_note=None,
    )


def _persist_deterministic_document_response(
    *,
    conversation_resolver: AnythingLLMConversationResolver,
    tutor_store: SQLiteTutorStore,
    resolution: ConversationResolution,
    parsed: ParsedConversation,
    routing_preview: RoutingPreview,
    request_id: str,
    request_fingerprint: str,
    interaction_id: str | None,
    rag_gate_result: RAGRelevanceGateResult,
) -> dict[str, Any]:
    answer = _deterministic_document_not_found_answer()
    response_text = answer.to_cli_text()
    persisted_turn_ids = conversation_resolver.append_exchange(
        resolution=resolution,
        parsed=parsed,
        response_text=response_text,
        request_id=request_id,
        request_fingerprint=request_fingerprint,
        integration_stage="I0.3.1D",
        extra_metadata={
            "anythingllm_interaction_id": interaction_id,
            "rag_relevance_gate_decision": rag_gate_result.decision,
        },
        response_json=answer.model_dump(),
        tutor_metadata={
            "source": "deterministic_rag_relevance_gate",
            "classification": routing_preview.classification.model_dump(),
            "routing": routing_preview.routing.model_dump(),
            "rag_relevance_gate": rag_gate_result.as_dict(include_context_text=False),
            "model_invoked": False,
        },
    )
    tutor_store.update_session_state(
        resolution.session_id,
        {
            "awaiting_student_answer": False,
            "last_bot_question": None,
            "active_concept_id": None,
            "last_tutor_move": answer.tutor_move,
            "last_concept_ids": answer.concept_ids,
            "last_route": routing_preview.routing.route,
            "last_route_reason": routing_preview.routing.reason,
            "last_question_classification": routing_preview.classification.main_type,
            "last_turn_relation": routing_preview.classification.turn_relation,
            "last_classification_safety_level": routing_preview.classification.safety_level,
            "last_needs_course_rag": routing_preview.classification.needs_course_rag,
            "last_needs_user_documents": routing_preview.classification.needs_user_documents,
            "last_needs_documents_dev": routing_preview.classification.needs_documents_dev,
            "last_needs_wikipedia": routing_preview.classification.needs_wikipedia,
            "last_needs_diagnostic_question": routing_preview.classification.needs_diagnostic_question,
            "last_context_source": "anythingllm_document_not_found",
            "last_anythingllm_rag_relevance": rag_gate_result.final_relevance,
            "last_anythingllm_rag_score": rag_gate_result.relevance_score,
            "last_anythingllm_sources": [],
            "last_student_turn_id": persisted_turn_ids.get("student"),
            "last_tutor_turn_id": persisted_turn_ids.get("tutor"),
        },
    )
    return persisted_turn_ids


def create_app(settings: ProbeSettings | None = None) -> FastAPI:
    settings = settings or ProbeSettings.from_environment()
    if not settings.resolve_conversations:
        raise ValueError("I0.3.1D requires conversation resolution and tutor SQLite storage.")

    model_catalog = OllamaModelCatalog(
        base_url=settings.ollama_base_url,
        allowed_models=settings.tutor_model_options,
    )

    probe_log = RequestProbeLog(
        RequestLogSettings(log_dir=settings.log_dir, log_content=settings.log_content)
    )
    interaction_log = ToolInteractionLog(
        InteractionLogSettings(log_dir=settings.log_dir, log_content=settings.log_content)
    )
    tutor_store = SQLiteTutorStore(settings.tutor_db_path)
    conversation_resolver = AnythingLLMConversationResolver(
        tutor_store,
        student_id=settings.student_id,
        max_candidate_sessions=settings.max_candidate_sessions,
        max_match_messages=settings.max_match_messages,
    )
    graph_config = TutorGraphConfig(
        model_name=settings.tutor_model_name,
        temperature=settings.tutor_temperature,
        tutor_num_ctx=settings.tutor_num_ctx,
        tutor_num_predict=settings.tutor_num_predict,
        classifier_enabled=settings.classifier_enabled,
        classifier_model_name=settings.classifier_model_name,
        classifier_temperature=settings.classifier_temperature,
        classification_confidence_threshold=settings.classification_confidence_threshold,
        classifier_fallback_to_rules=settings.classifier_fallback_to_rules,
        turn_relation_enabled=settings.turn_relation_enabled,
        turn_relation_model_name=settings.turn_relation_model_name,
        turn_relation_temperature=settings.turn_relation_temperature,
        turn_relation_num_ctx=settings.turn_relation_num_ctx,
        turn_relation_fallback_to_rules=settings.turn_relation_fallback_to_rules,
        semantic_safety_enabled=settings.semantic_safety_enabled,
        safety_embedding_model_name=settings.safety_embedding_model_name,
        safety_hazard_threshold=settings.safety_hazard_threshold,
        safety_procedure_threshold=settings.safety_procedure_threshold,
        safety_fuzzy_threshold=settings.safety_fuzzy_threshold,
        needs_documents_dev=settings.needs_documents_dev,
        followup_enabled=settings.followup_enabled,
        followup_model_name=settings.followup_model_name,
        followup_temperature=settings.followup_temperature,
    )
    routing_preview_service = AnythingLLMRoutingPreview(
        classifier_enabled=settings.classifier_enabled,
        classifier_model_name=settings.classifier_model_name,
        classifier_temperature=settings.classifier_temperature,
        confidence_threshold=settings.classification_confidence_threshold,
        fallback_to_rules=settings.classifier_fallback_to_rules,
        turn_relation_enabled=settings.turn_relation_enabled,
        turn_relation_model_name=settings.turn_relation_model_name,
        turn_relation_temperature=settings.turn_relation_temperature,
        turn_relation_num_ctx=settings.turn_relation_num_ctx,
        turn_relation_fallback_to_rules=settings.turn_relation_fallback_to_rules,
        semantic_safety_enabled=settings.semantic_safety_enabled,
        safety_embedding_model_name=settings.safety_embedding_model_name,
        safety_hazard_threshold=settings.safety_hazard_threshold,
        safety_procedure_threshold=settings.safety_procedure_threshold,
        safety_fuzzy_threshold=settings.safety_fuzzy_threshold,
        user_document_policy=settings.user_document_policy,
        needs_documents_dev=settings.needs_documents_dev,
    )
    tutor_bridge = TutorGraphBridge(
        store=tutor_store,
        conversation_resolver=conversation_resolver,
        graph_config=graph_config,
        context_turns=settings.context_turns,
    )
    rag_relevance_gate_config = RAGRelevanceGateConfig(
        enabled=settings.rag_relevance_gate_enabled,
        medium_threshold=settings.rag_relevance_medium_threshold,
        high_threshold=settings.rag_relevance_high_threshold,
        top_k=settings.rag_relevance_top_k,
        embedding_model_name=settings.rag_relevance_embedding_model_name,
    )
    rag_relevance_embedder = LazyOllamaEmbeddingProvider(
        settings.rag_relevance_embedding_model_name
    )
    rag_query_expansion_config = RAGQueryExpansionConfig(
        enabled=settings.rag_query_expansion_enabled,
        retry_enabled=settings.rag_query_expansion_retry_enabled,
        max_added_terms_light=settings.rag_query_expansion_max_added_terms_light,
        max_added_terms_targeted=settings.rag_query_expansion_max_added_terms_targeted,
    )

    app = FastAPI(
        title="Chemistry Tutor AnythingLLM LangGraph Bridge",
        version="I0.3.1D",
        description=(
            "OpenAI-compatible local tutor endpoint with durable conversation resolution, "
            "thin-label classification, deterministic source policy, AnythingLLM RAG, LangGraph routing, and SQLite audit logs."
        ),
    )
    app.state.probe_settings = settings
    app.state.model_catalog = model_catalog
    app.state.probe_log = probe_log
    app.state.interaction_log = interaction_log
    app.state.tutor_store = tutor_store
    app.state.conversation_resolver = conversation_resolver
    app.state.routing_preview_service = routing_preview_service
    app.state.tutor_bridge = tutor_bridge

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "Chemistry Tutor AnythingLLM LangGraph Bridge",
            "version": "I0.3.1D",
            "status": "ready",
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "request_index": "/probe/requests",
            "interaction_index": "/probe/interactions",
            "conversation_index": "/probe/conversations",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "chemistry-tutor-anythingllm-bridge",
            "version": "I0.3.1D",
            "model_id": settings.model_id,
            "tutor_model_name": settings.tutor_model_name,
            "tutor_num_ctx": settings.tutor_num_ctx,
            "tutor_num_predict": settings.tutor_num_predict,
            "classifier_enabled": settings.classifier_enabled,
            "classifier_model_name": settings.classifier_model_name,
            "classifier_contract": "main_type_only",
            "turn_relation_enabled": settings.turn_relation_enabled,
            "turn_relation_model_name": settings.turn_relation_model_name,
            "turn_relation_num_ctx": settings.turn_relation_num_ctx,
            "turn_relation_contract": "one_state_allowed_label_when_reply_awaited",
            "semantic_safety_enabled": settings.semantic_safety_enabled,
            "safety_embedding_model_name": settings.safety_embedding_model_name,
            "safety_hazard_threshold": settings.safety_hazard_threshold,
            "safety_procedure_threshold": settings.safety_procedure_threshold,
            "safety_fuzzy_threshold": settings.safety_fuzzy_threshold,
            "tutor_output_contract": "plain_text_only",
            "tutor_answer_assembled_by": "python",
            "needs_documents_dev": settings.needs_documents_dev,
            "followup_enabled": settings.followup_enabled,
            "followup_model_name": settings.followup_model_name,
            "classification_threshold_deprecated": settings.classification_confidence_threshold,
            "tutor_graph_connected": True,
            "conversation_resolver_enabled": True,
            "rag_tool_enabled": settings.rag_tool_probe,
            "rag_tool_name": settings.rag_tool_name,
            "user_document_policy": settings.user_document_policy,
            "max_rag_chunks": settings.max_rag_chunks,
            "max_rag_context_chars": settings.max_rag_context_chars,
            "rag_relevance_gate_enabled": settings.rag_relevance_gate_enabled,
            "rag_relevance_embedding_model_name": settings.rag_relevance_embedding_model_name,
            "rag_relevance_medium_threshold": settings.rag_relevance_medium_threshold,
            "rag_relevance_high_threshold": settings.rag_relevance_high_threshold,
            "rag_relevance_top_k": settings.rag_relevance_top_k,
            "rag_query_expansion_enabled": settings.rag_query_expansion_enabled,
            "rag_query_expansion_retry_enabled": settings.rag_query_expansion_retry_enabled,
            "rag_query_expansion_max_added_terms_light": settings.rag_query_expansion_max_added_terms_light,
            "rag_query_expansion_max_added_terms_targeted": settings.rag_query_expansion_max_added_terms_targeted,
            "log_dir": str(settings.log_dir.resolve()),
            "log_content": settings.log_content,
            "api_key_required": settings.api_key is not None,
            "tutor_db_path": str(settings.tutor_db_path.resolve()),
            "student_id": settings.student_id,
        }

    @app.get("/v1/models", response_model=None)
    async def list_models(request: Request) -> dict[str, Any] | JSONResponse:
        _require_api_key(request, settings)
        try:
            return await asyncio.to_thread(model_catalog.openai_model_list)
        except OllamaCatalogUnavailable as error:
            return _openai_error(
                str(error),
                status_code=503,
                error_type="server_error",
                code="model_backend_unavailable",
            )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        try:
            _require_api_key(request, settings)
        except HTTPException:
            return _openai_error(
                "Invalid API key.",
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_body_bytes:
                    return _openai_error(
                        f"Request body exceeds {settings.max_body_bytes} bytes.",
                        status_code=413,
                        code="request_too_large",
                    )
            except ValueError:
                pass

        raw_body = await request.body()
        if len(raw_body) > settings.max_body_bytes:
            return _openai_error(
                f"Request body exceeds {settings.max_body_bytes} bytes.",
                status_code=413,
                code="request_too_large",
            )
        try:
            body = _validate_chat_body(json.loads(raw_body.decode("utf-8")))
        except UnicodeDecodeError:
            return _openai_error(
                "Request body must be UTF-8 encoded JSON.", status_code=400, code="invalid_json"
            )
        except json.JSONDecodeError as error:
            return _openai_error(
                f"Invalid JSON: {error.msg}.", status_code=400, code="invalid_json"
            )
        except ValueError as error:
            return _openai_error(str(error), status_code=400, code="invalid_request")

        request_id = uuid4().hex
        received_at = utc_now_iso()
        created = int(time.time())
        requested_model = str(body.get("model") or settings.model_id)
        try:
            effective_model = await asyncio.to_thread(
                model_catalog.resolve_requested_model,
                requested_model,
                default_model=settings.tutor_model_name,
                legacy_model_id=settings.model_id,
            )
        except OllamaCatalogUnavailable as error:
            return _openai_error(
                str(error),
                status_code=503,
                error_type="server_error",
                code="model_backend_unavailable",
            )
        except TutorModelUnavailable as error:
            return _openai_error(
                str(error), status_code=400, code="model_not_available", param="model"
            )
        stream = bool(body.get("stream", False))
        include_usage = _stream_include_usage(body)
        request_fingerprint = sha256_json(body)
        inspection = inspect_tool_cycle(body, rag_tool_name=settings.rag_tool_name)

        resolution: ConversationResolution | None = None
        parsed_conversation: ParsedConversation | None = None
        routing_preview: RoutingPreview | None = None
        interaction: dict[str, Any] | None = None
        interaction_id = inspection.interaction_id
        tool_call_id: str | None = None
        adapted_context: AdaptedRAGContext | None = None
        gated_context: AdaptedRAGContext | None = None
        rag_gate_result: RAGRelevanceGateResult | None = None
        rag_query_expansion: RAGQueryExpansionResult | None = None
        retry_query_expansion: RAGQueryExpansionResult | None = None
        retrieval_query = ""
        previous_query_expansion_payload: Mapping[str, Any] | None = None
        retry_emitted = False
        external_context_used_as_evidence = False
        persisted_turn_ids: dict[str, Any] | None = None
        bridge_result: TutorBridgeResult | None = None
        bridge_error: str | None = None
        response_kind = "text"
        response_text = ""

        if inspection.phase == "tool_result_return" and interaction_id is not None:
            interaction = interaction_log.read(interaction_id)
            resolution = _interaction_resolution(interaction)
            parsed_conversation = _interaction_parsed_conversation(interaction)
            routing_preview = _interaction_routing_preview(interaction)

        try:
            if parsed_conversation is None:
                parsed_conversation = parse_openai_conversation(body["messages"])
            if resolution is None:
                resolution, resolved_parsed = conversation_resolver.resolve(body["messages"])
                if parsed_conversation is None:
                    parsed_conversation = resolved_parsed

            session_state = tutor_store.get_session_state(resolution.session_id)
            if routing_preview is None:
                routing_preview = await asyncio.to_thread(
                    routing_preview_service.classify,
                    user_text=parsed_conversation.current_user_text,
                    session_state=session_state,
                    rag_tool_available=(
                        settings.rag_tool_probe
                        and settings.rag_tool_name in inspection.available_tools
                    ),
                )

            if (
                inspection.phase == "initial_request"
                and routing_preview.should_call_user_documents
            ):
                interaction_id = make_interaction_id()
                tool_call_id = make_tool_call_id(interaction_id)
                explicit_document_request = _explicit_document_request(routing_preview)
                rag_query_expansion = choose_query_expansion(
                    original_query=parsed_conversation.current_user_text,
                    should_call_user_documents=routing_preview.should_call_user_documents,
                    explicit_document_request=explicit_document_request,
                    route=routing_preview.routing.route,
                    safety_level=routing_preview.classification.safety_level,
                    config=rag_query_expansion_config,
                )
                retrieval_query = rag_query_expansion.expanded_query
                arguments_json = json.dumps(
                    {
                        "action": "search",
                        "content": retrieval_query,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                interaction = interaction_log.start(
                    interaction_id=interaction_id,
                    tool_call_id=tool_call_id,
                    created_at=received_at,
                    session_id=resolution.session_id,
                    student_id=resolution.student_id,
                    initial_request_id=request_id,
                    initial_request_fingerprint=request_fingerprint,
                    query=retrieval_query,
                    original_query=parsed_conversation.current_user_text,
                    query_expansion=rag_query_expansion.as_dict(),
                    resolution=resolution.as_dict(),
                    parsed_conversation=parsed_conversation_to_dict(parsed_conversation),
                    routing_preview=routing_preview.as_dict(),
                )
                response_kind = "tool_call"
                response_text = arguments_json
            else:
                if inspection.phase == "tool_result_return":
                    latest_tool_results = inspection.tool_results[-1:] or inspection.tool_results
                    adapted_context = adapt_anythingllm_tool_results(
                        query=parsed_conversation.current_user_text,
                        tool_results=latest_tool_results,
                        max_chunks=settings.max_rag_chunks,
                        max_total_chars=settings.max_rag_context_chars,
                    )
                    explicit_document_request = _explicit_document_request(routing_preview)
                    rag_gate_result = await asyncio.to_thread(
                        apply_rag_relevance_gate,
                        query=parsed_conversation.current_user_text,
                        context=adapted_context,
                        explicit_user_document_request=explicit_document_request,
                        config=rag_relevance_gate_config,
                        embed_texts=rag_relevance_embedder.embed_documents,
                    )
                    previous_expansion = (
                        interaction.get("retry_query_expansion")
                        or interaction.get("rag_query_expansion")
                        if isinstance(interaction, Mapping)
                        else None
                    )
                    previous_query_expansion_payload = (
                        previous_expansion if isinstance(previous_expansion, Mapping) else None
                    )
                    retry_query_expansion = choose_query_expansion(
                        original_query=parsed_conversation.current_user_text,
                        should_call_user_documents=routing_preview.should_call_user_documents,
                        explicit_document_request=explicit_document_request,
                        route=routing_preview.routing.route,
                        safety_level=routing_preview.classification.safety_level,
                        config=rag_query_expansion_config,
                        first_gate_result=rag_gate_result,
                        previous_expansion=previous_expansion,
                    )
                    should_retry = bool(
                        retry_query_expansion.changed
                        and retry_query_expansion.mode == "retry_targeted"
                        and interaction_id is not None
                    )
                    if should_retry:
                        retry_count = int(interaction.get("retry_count") or 0) + 1 if isinstance(interaction, Mapping) else 1
                        tool_call_id = make_retry_tool_call_id(interaction_id, retry_count)
                        retrieval_query = retry_query_expansion.expanded_query
                        arguments_json = json.dumps(
                            {
                                "action": "search",
                                "content": retrieval_query,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        interaction_log.record_retry(
                            interaction_id=interaction_id,
                            request_id=request_id,
                            retry_tool_call_id=tool_call_id,
                            retry_query=retrieval_query,
                            query_expansion=retry_query_expansion.as_dict(),
                            rag_relevance_gate=rag_gate_result.as_dict(include_context_text=False),
                        )
                        response_kind = "tool_call"
                        response_text = arguments_json
                        retry_emitted = True
                    else:
                        gated_context = rag_gate_result.context_for_prompt
                        external_context_used_as_evidence = gated_context is not None
                        if rag_gate_result.decision == "explicit_document_not_found":
                            answer = _deterministic_document_not_found_answer()
                            response_text = answer.to_cli_text()
                            persisted_turn_ids = _persist_deterministic_document_response(
                                conversation_resolver=conversation_resolver,
                                tutor_store=tutor_store,
                                resolution=resolution,
                                parsed=parsed_conversation,
                                routing_preview=routing_preview,
                                request_id=request_id,
                                request_fingerprint=request_fingerprint,
                                interaction_id=interaction_id,
                                rag_gate_result=rag_gate_result,
                            )
                        else:
                            bridge_result = await asyncio.to_thread(
                                tutor_bridge.invoke,
                                resolution=resolution,
                                parsed=parsed_conversation,
                                routing_preview=routing_preview,
                                request_id=request_id,
                                request_fingerprint=request_fingerprint,
                                model_name=effective_model,
                                external_context=gated_context,
                                interaction_id=interaction_id,
                            )
                            response_text = bridge_result.response_text
                            persisted_turn_ids = bridge_result.persisted_turn_ids
                else:
                    bridge_result = await asyncio.to_thread(
                        tutor_bridge.invoke,
                        resolution=resolution,
                        parsed=parsed_conversation,
                        routing_preview=routing_preview,
                        request_id=request_id,
                        request_fingerprint=request_fingerprint,
                        model_name=effective_model,
                        external_context=None,
                        interaction_id=interaction_id,
                    )
                    response_text = bridge_result.response_text
                    persisted_turn_ids = bridge_result.persisted_turn_ids
                if interaction_id is not None and inspection.phase == "tool_result_return" and not retry_emitted:
                    interaction = interaction_log.complete(
                        interaction_id=interaction_id,
                        result_request_id=request_id,
                        completed_at=received_at,
                        inspection=inspection,
                        final_response_text=response_text,
                        persisted_turn_ids=persisted_turn_ids,
                        completion_error=None,
                        adapted_context=(
                            {
                                **adapted_context.as_dict(include_text=settings.log_content),
                                "retrieved_context_relevance": adapted_context.relevance,
                                "retrieved_context_score": adapted_context.relevance_score,
                                "retrieved_chunk_count": len(adapted_context.chunks),
                                "gated_context_relevance": (
                                    rag_gate_result.final_relevance
                                    if rag_gate_result is not None
                                    else None
                                ),
                                "gated_context_score": (
                                    rag_gate_result.relevance_score
                                    if rag_gate_result is not None
                                    else None
                                ),
                                "passed_chunk_count": (
                                    rag_gate_result.passed_chunk_count
                                    if rag_gate_result is not None
                                    else None
                                ),
                                "rejected_chunk_count": (
                                    rag_gate_result.rejected_chunk_count
                                    if rag_gate_result is not None
                                    else None
                                ),
                                "external_context_used_as_evidence": external_context_used_as_evidence,
                                "rag_query_expansion": (
                                    retry_query_expansion.as_dict()
                                    if retry_query_expansion is not None and retry_query_expansion.changed
                                    else (
                                        rag_query_expansion.as_dict()
                                        if rag_query_expansion is not None
                                        else previous_query_expansion_payload
                                    )
                                ),
                            }
                            if adapted_context is not None
                            else None
                        ),
                        rag_relevance_gate=(
                            rag_gate_result.as_dict(include_context_text=settings.log_content)
                            if rag_gate_result is not None
                            else None
                        ),
                        external_context_used_as_evidence=external_context_used_as_evidence,
                        graph_result=(
                            bridge_result.as_dict(include_context_text=settings.log_content)
                            if bridge_result is not None
                            else {
                                "deterministic_response": True,
                                "model_invoked": False,
                                "external_context_used_as_evidence": external_context_used_as_evidence,
                                "rag_query_expansion": (
                                    retry_query_expansion.as_dict()
                                    if retry_query_expansion is not None and retry_query_expansion.changed
                                    else (
                                        rag_query_expansion.as_dict()
                                        if rag_query_expansion is not None
                                        else previous_query_expansion_payload
                                    )
                                ),
                                "response_text": response_text if settings.log_content else None,
                                "rag_relevance_gate": (
                                    rag_gate_result.as_dict(
                                        include_context_text=settings.log_content
                                    )
                                    if rag_gate_result is not None
                                    else None
                                ),
                            }
                        ),
                    )
        except Exception as error:
            bridge_error = f"{type(error).__name__}: {error}"
            response_text = _safe_bridge_error_text()
            if interaction_id is not None:
                interaction_log.fail(
                    interaction_id=interaction_id,
                    request_id=request_id,
                    error=bridge_error,
                )

        client = {
            "host": request.client.host if request.client else None,
            "port": request.client.port if request.client else None,
        }
        record = probe_log.make_record(
            request_id=request_id,
            received_at=received_at,
            method=request.method,
            path=request.url.path,
            client=client,
            headers=dict(request.headers),
            body=body,
            raw_body_bytes=len(raw_body),
            response_mode=(
                "tool_call_stream"
                if response_kind == "tool_call" and stream
                else "tool_call_json"
                if response_kind == "tool_call"
                else "stream"
                if stream
                else "json"
            ),
            response_model=effective_model,
        )
        record["requested_model"] = requested_model
        record["selected_tutor_model"] = effective_model
        record["schema_version"] = 5
        record["integration_stage"] = "I0.3.1D"
        record["request_fingerprint"] = request_fingerprint
        record["tool_cycle"] = inspection.as_dict(include_content=settings.log_content)
        record["tool_cycle"].update(
            {
                "response_kind": response_kind,
                "interaction_id": interaction_id,
                "emitted_tool_call_id": tool_call_id,
                "interaction_status": interaction.get("status") if interaction else None,
            }
        )
        if resolution is not None:
            record["conversation_resolution"] = resolution.as_dict()
        if routing_preview is not None:
            record["routing_preview"] = routing_preview.as_dict()
        if adapted_context is not None:
            record["adapted_context"] = adapted_context.as_dict(
                include_text=settings.log_content
            )
            record["retrieved_context_relevance"] = adapted_context.relevance
            record["retrieved_context_score"] = adapted_context.relevance_score
            record["retrieved_chunk_count"] = len(adapted_context.chunks)
        if rag_query_expansion is not None:
            record["rag_query_expansion"] = rag_query_expansion.as_dict()
            record["retrieval_query"] = rag_query_expansion.expanded_query if settings.log_content else None
            record["original_user_query"] = parsed_conversation.current_user_text if parsed_conversation is not None and settings.log_content else None
        if retry_query_expansion is not None:
            record["retry_query_expansion"] = retry_query_expansion.as_dict()
            if retry_query_expansion.changed:
                record["retrieval_query"] = retry_query_expansion.expanded_query if settings.log_content else None
        if previous_query_expansion_payload is not None:
            record["previous_rag_query_expansion"] = dict(previous_query_expansion_payload)
        record["rag_query_retry_emitted"] = retry_emitted
        if rag_gate_result is not None:
            record["rag_relevance_gate"] = rag_gate_result.as_dict(
                include_context_text=settings.log_content
            )
            record["gated_context_relevance"] = rag_gate_result.final_relevance
            record["gated_context_score"] = rag_gate_result.relevance_score
            record["gated_decision"] = rag_gate_result.decision
            record["gated_passed_chunk_count"] = rag_gate_result.passed_chunk_count
            record["gated_rejected_chunk_count"] = rag_gate_result.rejected_chunk_count
        record["external_context_received"] = adapted_context is not None
        record["external_context_used_as_evidence"] = external_context_used_as_evidence
        if bridge_result is not None:
            record["tutor_graph_result"] = bridge_result.as_dict(
                include_context_text=settings.log_content
            )
        elif rag_gate_result is not None and response_text:
            if retry_emitted:
                record["tutor_graph_result"] = {
                    "tool_retry_emitted": True,
                    "model_invoked": False,
                    "external_context_used_as_evidence": external_context_used_as_evidence,
                    "rag_relevance_gate": rag_gate_result.as_dict(
                        include_context_text=settings.log_content
                    ),
                    "retry_query_expansion": (
                        retry_query_expansion.as_dict()
                        if retry_query_expansion is not None
                        else None
                    ),
                }
            else:
                record["tutor_graph_result"] = {
                    "deterministic_response": True,
                    "model_invoked": False,
                    "external_context_used_as_evidence": external_context_used_as_evidence,
                    "rag_relevance_gate": rag_gate_result.as_dict(
                        include_context_text=settings.log_content
                    ),
                    "response_text": response_text if settings.log_content else None,
                }
        if bridge_error is not None:
            record["tutor_bridge_error"] = bridge_error
        log_path = probe_log.persist(record)

        print(
            "[ANYTHINGLLM_I0_3_1D] "
            f"request_id={request_id} interaction={interaction_id or 'none'} "
            f"phase={inspection.phase} requested_model={requested_model} "
            f"selected_tutor_model={effective_model} response={response_kind} "
            f"session={resolution.session_id if resolution else 'unresolved'} "
            f"route={routing_preview.routing.route if routing_preview else 'unresolved'} "
            f"rag={routing_preview.should_call_user_documents if routing_preview else False} "
            f"context={adapted_context.relevance if adapted_context else 'none'} "
            f"error={bridge_error or 'none'} log={log_path}"
        )

        common_headers = {
            "X-Request-ID": request_id,
            "X-Chemistry-Tutor-Probe": "I0.3.1D",
            "X-Chemistry-Tutor-Tool-Phase": inspection.phase,
        }
        if interaction_id is not None:
            common_headers["X-Chemistry-Tutor-Interaction"] = interaction_id
        if resolution is not None:
            common_headers["X-Chemistry-Tutor-Session"] = resolution.session_id
            common_headers["X-Chemistry-Tutor-Resolution"] = resolution.resolution_method
        if routing_preview is not None:
            common_headers["X-Chemistry-Tutor-Route"] = routing_preview.routing.route
        if adapted_context is not None:
            common_headers["X-Chemistry-Tutor-RAG-Relevance"] = adapted_context.relevance
        if rag_gate_result is not None:
            common_headers["X-Chemistry-Tutor-RAG-Gated-Relevance"] = (
                rag_gate_result.final_relevance
            )
            common_headers["X-Chemistry-Tutor-RAG-Evidence"] = (
                "used" if external_context_used_as_evidence else "suppressed"
            )

        if response_kind == "tool_call":
            assert tool_call_id is not None
            if stream:
                return StreamingResponse(
                    _stream_tool_call_completion(
                        request_id=request_id,
                        model=effective_model,
                        created=created,
                        tool_call_id=tool_call_id,
                        tool_name=settings.rag_tool_name,
                        arguments_json=response_text,
                        include_usage=include_usage,
                        body=body,
                    ),
                    media_type="text/event-stream",
                    headers={
                        **common_headers,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            return JSONResponse(
                content=_non_stream_tool_call_completion(
                    request_id=request_id,
                    model=effective_model,
                    created=created,
                    tool_call_id=tool_call_id,
                    tool_name=settings.rag_tool_name,
                    arguments_json=response_text,
                    body=body,
                ),
                headers=common_headers,
            )

        if stream:
            return StreamingResponse(
                _stream_text_completion(
                    request_id=request_id,
                    model=effective_model,
                    created=created,
                    response_text=response_text,
                    include_usage=include_usage,
                    body=body,
                ),
                media_type="text/event-stream",
                headers={
                    **common_headers,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return JSONResponse(
            content=_non_stream_text_completion(
                request_id=request_id,
                model=effective_model,
                created=created,
                response_text=response_text,
                body=body,
            ),
            headers=common_headers,
        )

    @app.get("/probe/requests")
    async def recent_requests(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
        include_body: bool = Query(default=False),
    ) -> dict[str, Any]:
        _require_api_key(request, settings)
        rows = probe_log.recent(limit=limit, include_body=include_body)
        return {"count": len(rows), "requests": rows}

    @app.get("/probe/requests/{request_id}")
    async def read_request(request_id: str, request: Request) -> dict[str, Any]:
        _require_api_key(request, settings)
        row = probe_log.read(request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Probe request was not found.")
        return row

    @app.get("/probe/interactions")
    async def recent_interactions(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
        include_content: bool = Query(default=False),
    ) -> dict[str, Any]:
        _require_api_key(request, settings)
        rows = interaction_log.recent(limit=limit, include_content=include_content)
        return {"count": len(rows), "interactions": rows}

    @app.get("/probe/interactions/{interaction_id}")
    async def read_interaction(interaction_id: str, request: Request) -> dict[str, Any]:
        _require_api_key(request, settings)
        row = interaction_log.read(interaction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Tool interaction was not found.")
        return row

    @app.get("/probe/conversations")
    async def recent_conversations(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
    ) -> dict[str, Any]:
        _require_api_key(request, settings)
        sessions = tutor_store.get_recent_sessions(student_id=settings.student_id, limit=limit)
        return {
            "count": len(sessions),
            "student_id": settings.student_id,
            "sessions": [
                {
                    "session_id": session.id,
                    "title": session.title,
                    "status": session.status,
                    "started_at": session.started_at,
                    "updated_at": session.updated_at,
                    "turn_count": tutor_store.count_session_turns(session.id),
                }
                for session in sessions
            ],
        }

    @app.get("/probe/conversations/{session_id}")
    async def read_conversation(session_id: str, request: Request) -> dict[str, Any]:
        _require_api_key(request, settings)
        session = tutor_store.get_session(session_id)
        if session is None or session.student_id != settings.student_id:
            raise HTTPException(status_code=404, detail="Conversation was not found.")
        turns = tutor_store.get_recent_turns(
            session_id=session_id,
            limit=500,
            roles=["student", "tutor"],
        )
        decisions = tutor_store.get_recent_decisions(session_id=session_id, limit=500)
        return {
            "session": {
                "session_id": session.id,
                "student_id": session.student_id,
                "title": session.title,
                "status": session.status,
                "started_at": session.started_at,
                "updated_at": session.updated_at,
                "state": tutor_store.get_session_state(session_id),
            },
            "turns": [
                {
                    "id": turn.id,
                    "role": turn.role,
                    "text": turn.text,
                    "response_json": turn.response_json,
                    "created_at": turn.created_at,
                    "metadata": turn.metadata,
                }
                for turn in turns
            ],
            "decisions": [
                {
                    "id": item.id,
                    "turn_id": item.turn_id,
                    "decision_type": item.decision_type,
                    "node_name": item.node_name,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "decision": item.decision,
                    "created_at": item.created_at,
                }
                for item in decisions
            ],
        }

    return app
