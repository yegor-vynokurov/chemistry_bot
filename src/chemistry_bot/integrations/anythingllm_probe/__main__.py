"""Command-line launcher for the AnythingLLM LangGraph bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from chemistry_bot.tutor_graph.graph import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_TUTOR_MODEL,
)
from chemistry_bot.tutor_graph.storage import DEFAULT_DB_PATH

from .app import (
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_RAG_CHUNKS,
    DEFAULT_MAX_RAG_CONTEXT_CHARS,
    DEFAULT_MODEL_ID,
    ProbeSettings,
    create_app,
)
from .conversation_resolver import (
    DEFAULT_MAX_CANDIDATE_SESSIONS,
    DEFAULT_MAX_MATCH_MESSAGES,
    DEFAULT_STUDENT_ID,
)
from .tutor_bridge import DEFAULT_CONTEXT_TURNS


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible AnythingLLM Desktop endpoint backed by the durable "
            "LangGraph chemistry tutor and classifier-gated user-document RAG."
        )
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CHEM_TUTOR_PROBE_HOST", "127.0.0.1"),
        help="Bind host. Keep 127.0.0.1 for a local Desktop integration.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_PROBE_PORT", "8765")),
        help="Bind port. Default: 8765.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("CHEM_TUTOR_PROBE_MODEL_ID", DEFAULT_MODEL_ID),
        help=f"Model name exposed to AnythingLLM. Default: {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--tutor-model",
        default=os.getenv("CHEM_TUTOR_MODEL", DEFAULT_TUTOR_MODEL),
        help=f"Ollama model used for tutor answers. Default: {DEFAULT_TUTOR_MODEL}.",
    )
    parser.add_argument(
        "--tutor-num-ctx",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_NUM_CTX", "8192")),
        help="Context window for tutor answer generation. Default: 8192.",
    )
    parser.add_argument(
        "--tutor-num-predict",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_NUM_PREDICT", "1536")),
        help="Maximum generated tokens for tutor answers. Default: 1536.",
    )
    parser.add_argument(
        "--classifier-model",
        default=os.getenv("CHEM_TUTOR_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
        help=f"Small Ollama routing model. Default: {DEFAULT_CLASSIFIER_MODEL}.",
    )
    parser.add_argument(
        "--classifier",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_CLASSIFIER_ENABLED", True),
        help="Use the small LLM classifier. --no-classifier uses deterministic rules.",
    )
    parser.add_argument(
        "--turn-relation-model",
        default=os.getenv("CHEM_TUTOR_TURN_RELATION_MODEL", ""),
        help=(
            "Small Ollama model used only to classify student_answer, "
            "clarification_choice, new_question, or unclear. "
            "Default: the selected classifier model."
        ),
    )
    parser.add_argument(
        "--turn-relation-num-ctx",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_TURN_RELATION_NUM_CTX", "2048")),
        help="Context window for the narrow turn-relation call. Default: 2048.",
    )
    parser.add_argument(
        "--turn-relation",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_TURN_RELATION_ENABLED", True),
        help="Enable the separate one-label dialogue-relation classifier.",
    )
    parser.add_argument(
        "--semantic-safety",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_SEMANTIC_SAFETY_ENABLED", True),
        help="Enable the RapidFuzz plus embedding safety gate.",
    )
    parser.add_argument(
        "--safety-embedding-model",
        default=os.getenv("CHEM_TUTOR_SAFETY_EMBEDDING_MODEL", "nomic-embed-text"),
        help="Ollama embedding model for the safety prototype dictionary.",
    )
    parser.add_argument(
        "--safety-hazard-threshold",
        type=float,
        default=float(os.getenv("CHEM_TUTOR_SAFETY_HAZARD_THRESHOLD", "0.60")),
        help="Cosine threshold for hazard prototypes. Calibrate on local examples.",
    )
    parser.add_argument(
        "--safety-procedure-threshold",
        type=float,
        default=float(os.getenv("CHEM_TUTOR_SAFETY_PROCEDURE_THRESHOLD", "0.60")),
        help="Cosine threshold for procedural-intent prototypes.",
    )
    parser.add_argument(
        "--safety-fuzzy-threshold",
        type=float,
        default=float(os.getenv("CHEM_TUTOR_SAFETY_FUZZY_THRESHOLD", "86")),
        help="RapidFuzz threshold from 0 to 100 for typo-tolerant aliases.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=float(
            os.getenv(
                "CHEM_TUTOR_CLASSIFICATION_THRESHOLD",
                str(DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD),
            )
        ),
        help="Deprecated compatibility option; I0.3.1B.4 does not use model confidence.",
    )
    parser.add_argument(
        "--user-document-policy",
        choices=["classifier", "always", "never"],
        default=os.getenv("CHEM_TUTOR_USER_DOCUMENT_POLICY", "classifier"),
        help=(
            "When to call AnythingLLM rag-memory. classifier uses the program-derived needs_user_documents flag, "
            "always is useful for diagnostics, never disables user-document retrieval."
        ),
    )
    parser.add_argument(
        "--needs-documents-dev",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_NEEDS_DOCUMENTS_DEV", False),
        help=(
            "Developer override. When true, needs_user_documents is set to true "
            "regardless of the classifier label or wording of the request. "
            "Protected routes can still block the actual tool call."
        ),
    )
    parser.add_argument(
        "--followup-model",
        default=os.getenv("CHEM_TUTOR_FOLLOWUP_MODEL", ""),
        help=(
            "Small Ollama model used only to generate one optional check question. "
            "Default: the selected classifier model."
        ),
    )
    parser.add_argument(
        "--followup",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_FOLLOWUP_ENABLED", True),
        help="Enable or disable the separate follow-up-question model call.",
    )
    parser.add_argument(
        "--rag-tool-name",
        default=os.getenv("CHEM_TUTOR_RAG_TOOL_NAME", "rag-memory"),
        help="AnythingLLM local-document tool name. Default: rag-memory.",
    )
    parser.add_argument(
        "--rag-tool",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_RAG_TOOL_PROBE", True),
        help="Enable or disable AnythingLLM user-document tool calls.",
    )
    parser.add_argument(
        "--max-rag-chunks",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_MAX_RAG_CHUNKS", str(DEFAULT_MAX_RAG_CHUNKS))),
        help=f"Maximum adapted document chunks. Default: {DEFAULT_MAX_RAG_CHUNKS}.",
    )
    parser.add_argument(
        "--max-rag-context-chars",
        type=int,
        default=int(
            os.getenv(
                "CHEM_TUTOR_MAX_RAG_CONTEXT_CHARS",
                str(DEFAULT_MAX_RAG_CONTEXT_CHARS),
            )
        ),
        help=f"Character budget for user-document excerpts. Default: {DEFAULT_MAX_RAG_CONTEXT_CHARS}.",
    )
    parser.add_argument(
        "--rag-relevance-gate",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_RAG_RELEVANCE_GATE_ENABLED", True),
        help="Enable the embedding relevance gate for AnythingLLM document chunks.",
    )
    parser.add_argument(
        "--rag-relevance-embedding-model",
        default=os.getenv(
            "CHEM_TUTOR_RAG_RELEVANCE_EMBEDDING_MODEL", "embeddinggemma:latest"
        ),
        help="Ollama embedding model for RAG chunk relevance. Default: embeddinggemma:latest.",
    )
    parser.add_argument(
        "--rag-relevance-medium-threshold",
        type=float,
        default=float(os.getenv("CHEM_TUTOR_RAG_RELEVANCE_MEDIUM_THRESHOLD", "0.45")),
        help="Minimum cosine score to pass a chunk as document evidence. Default: 0.45.",
    )
    parser.add_argument(
        "--rag-relevance-high-threshold",
        type=float,
        default=float(os.getenv("CHEM_TUTOR_RAG_RELEVANCE_HIGH_THRESHOLD", "0.58")),
        help="Cosine score for high document relevance diagnostics. Default: 0.58.",
    )
    parser.add_argument(
        "--rag-relevance-top-k",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_RAG_RELEVANCE_TOP_K", "2")),
        help="Maximum document chunks passed to the tutor prompt as evidence. Default: 2.",
    )
    parser.add_argument(
        "--rag-query-expansion",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_RAG_QUERY_EXPANSION_ENABLED", True),
        help="Enable deterministic anchor-aware expansion of the AnythingLLM search query.",
    )
    parser.add_argument(
        "--rag-query-expansion-retry",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_RAG_QUERY_EXPANSION_RETRY_ENABLED", True),
        help="After low/topic-adjacent evidence, emit one targeted expanded RAG retry.",
    )
    parser.add_argument(
        "--rag-query-expansion-max-added-terms-light",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_RAG_QUERY_EXPANSION_MAX_ADDED_TERMS_LIGHT", "8")),
        help="Maximum terms added by initial light query expansion. Default: 8.",
    )
    parser.add_argument(
        "--rag-query-expansion-max-added-terms-targeted",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_RAG_QUERY_EXPANSION_MAX_ADDED_TERMS_TARGETED", "14")),
        help="Maximum terms added by targeted retry query expansion. Default: 14.",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=int(os.getenv("CHEM_TUTOR_CONTEXT_TURNS", str(DEFAULT_CONTEXT_TURNS))),
        help=f"Operational history limit stored in session state. Default: {DEFAULT_CONTEXT_TURNS}.",
    )
    parser.add_argument(
        "--tutor-db",
        type=Path,
        default=Path(os.getenv("CHEM_TUTOR_DB_PATH", str(DEFAULT_DB_PATH))),
        help=f"Tutor SQLite database. Default: {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--student-id",
        default=os.getenv("CHEM_TUTOR_STUDENT_ID", DEFAULT_STUDENT_ID),
        help=f"Local student identifier. Default: {DEFAULT_STUDENT_ID}.",
    )
    parser.add_argument(
        "--max-candidate-sessions",
        type=int,
        default=int(
            os.getenv(
                "CHEM_TUTOR_MAX_CANDIDATE_SESSIONS",
                str(DEFAULT_MAX_CANDIDATE_SESSIONS),
            )
        ),
        help="Recent sessions inspected for an exact history suffix match.",
    )
    parser.add_argument(
        "--max-match-messages",
        type=int,
        default=int(
            os.getenv("CHEM_TUTOR_MAX_MATCH_MESSAGES", str(DEFAULT_MAX_MATCH_MESSAGES))
        ),
        help="Maximum user/assistant history messages used for session matching.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.getenv("CHEM_TUTOR_PROBE_LOG_DIR", str(DEFAULT_LOG_DIR))),
        help=f"Integration log directory. Default: {DEFAULT_LOG_DIR}.",
    )
    parser.add_argument(
        "--log-content",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("CHEM_TUTOR_PROBE_LOG_CONTENT", True),
        help="Store full prompt, RAG, and graph content in local diagnostics.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CHEM_TUTOR_PROBE_API_KEY"),
        help="Optional local API key enforced by the bridge.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(
            os.getenv("CHEM_TUTOR_PROBE_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
        ),
        help="Maximum accepted request size. Default: 20 MiB.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("CHEM_TUTOR_PROBE_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return parser


def main() -> None:
    """Build and serve the bridge with the active Ollama catalog settings.

    CLI arguments configure server and tutor behavior; OLLAMA_HOST and
    CHEM_TUTOR_MODEL_OPTIONS identify the backend and selectable tutor roster.
    The catalog must inspect the same Ollama instance used by the tutor graph.
    """
    args = build_parser().parse_args()
    active_backend_settings = ProbeSettings.from_environment()
    settings = ProbeSettings(
        model_id=args.model_id.strip() or DEFAULT_MODEL_ID,
        log_dir=args.log_dir,
        log_content=bool(args.log_content),
        api_key=args.api_key or None,
        ollama_base_url=active_backend_settings.ollama_base_url,
        tutor_model_options=active_backend_settings.tutor_model_options,
        max_body_bytes=max(1024, int(args.max_body_bytes)),
        resolve_conversations=True,
        tutor_db_path=args.tutor_db,
        student_id=args.student_id.strip() or DEFAULT_STUDENT_ID,
        max_candidate_sessions=max(1, int(args.max_candidate_sessions)),
        max_match_messages=max(2, int(args.max_match_messages)),
        rag_tool_probe=bool(args.rag_tool),
        rag_tool_name=args.rag_tool_name.strip() or "rag-memory",
        user_document_policy=args.user_document_policy,
        needs_documents_dev=bool(args.needs_documents_dev),
        max_rag_chunks=max(1, int(args.max_rag_chunks)),
        max_rag_context_chars=max(1000, int(args.max_rag_context_chars)),
        rag_relevance_gate_enabled=bool(args.rag_relevance_gate),
        rag_relevance_embedding_model_name=(
            args.rag_relevance_embedding_model.strip() or "embeddinggemma:latest"
        ),
        rag_relevance_medium_threshold=max(
            -1.0, min(1.0, float(args.rag_relevance_medium_threshold))
        ),
        rag_relevance_high_threshold=max(
            -1.0, min(1.0, float(args.rag_relevance_high_threshold))
        ),
        rag_relevance_top_k=max(1, int(args.rag_relevance_top_k)),
        rag_query_expansion_enabled=bool(args.rag_query_expansion),
        rag_query_expansion_retry_enabled=bool(args.rag_query_expansion_retry),
        rag_query_expansion_max_added_terms_light=max(0, int(args.rag_query_expansion_max_added_terms_light)),
        rag_query_expansion_max_added_terms_targeted=max(0, int(args.rag_query_expansion_max_added_terms_targeted)),
        tutor_model_name=args.tutor_model.strip() or DEFAULT_TUTOR_MODEL,
        tutor_num_ctx=max(1024, int(args.tutor_num_ctx)),
        tutor_num_predict=max(128, int(args.tutor_num_predict)),
        classifier_enabled=bool(args.classifier),
        classifier_model_name=args.classifier_model.strip() or DEFAULT_CLASSIFIER_MODEL,
        turn_relation_enabled=bool(args.turn_relation),
        turn_relation_model_name=(
            args.turn_relation_model.strip()
            or args.classifier_model.strip()
            or DEFAULT_CLASSIFIER_MODEL
        ),
        turn_relation_num_ctx=max(512, int(args.turn_relation_num_ctx)),
        semantic_safety_enabled=bool(args.semantic_safety),
        safety_embedding_model_name=(
            args.safety_embedding_model.strip() or "nomic-embed-text"
        ),
        safety_hazard_threshold=max(-1.0, min(1.0, float(args.safety_hazard_threshold))),
        safety_procedure_threshold=max(
            -1.0, min(1.0, float(args.safety_procedure_threshold))
        ),
        safety_fuzzy_threshold=max(0.0, min(100.0, float(args.safety_fuzzy_threshold))),
        followup_enabled=bool(args.followup),
        followup_model_name=(
            args.followup_model.strip()
            or args.classifier_model.strip()
            or DEFAULT_CLASSIFIER_MODEL
        ),
        classification_confidence_threshold=max(
            0.0, min(1.0, float(args.classification_threshold))
        ),
        context_turns=max(1, int(args.context_turns)),
    )
    application = create_app(settings)

    print("Chemistry Tutor AnythingLLM LangGraph Bridge I0.3.1D")
    print("Turn relation: one label constrained by the previous tutor move")
    print(f"  Base URL for AnythingLLM: http://{args.host}:{args.port}/v1")
    print(f"  Exposed model: {settings.model_id}")
    print(f"  Tutor model: {settings.tutor_model_name}")
    print(
        f"  Tutor generation: num_ctx={settings.tutor_num_ctx}, "
        f"num_predict={settings.tutor_num_predict}"
    )
    print(
        f"  Classifier: {settings.classifier_model_name if settings.classifier_enabled else 'rules only'}"
    )
    print("  Classifier contract: one main_type label; no score or JSON")
    print(
        f"  Turn relation: {settings.turn_relation_model_name if settings.turn_relation_enabled else 'rules only'}"
    )
    print("  Turn-relation contract: one label only when a tutor reply is awaited")
    print(f"  Turn-relation context window: {settings.turn_relation_num_ctx}")
    print(
        f"  Semantic safety: {settings.safety_embedding_model_name if settings.semantic_safety_enabled else 'RapidFuzz only'}"
    )
    print(
        f"  Safety thresholds: hazard={settings.safety_hazard_threshold:.2f}, "
        f"procedure={settings.safety_procedure_threshold:.2f}, "
        f"fuzzy={settings.safety_fuzzy_threshold:.1f}"
    )
    print("  Tutor contract: plain answer text; TutorAnswer assembled by Python")
    print(f"  Follow-up model: {settings.followup_model_name if settings.followup_enabled else 'disabled'}")
    print(f"  needs_documents_dev: {settings.needs_documents_dev}")
    print(f"  Tutor graph connected: True")
    print(f"  User-document policy: {settings.user_document_policy}")
    print(f"  RAG tool: {settings.rag_tool_name if settings.rag_tool_probe else 'disabled'}")
    print(
        "  RAG relevance gate: "
        f"{settings.rag_relevance_embedding_model_name if settings.rag_relevance_gate_enabled else 'disabled'} "
        f"medium={settings.rag_relevance_medium_threshold:.2f} "
        f"high={settings.rag_relevance_high_threshold:.2f} "
        f"top_k={settings.rag_relevance_top_k}"
    )
    print(
        "  RAG query expansion: "
        f"{'enabled' if settings.rag_query_expansion_enabled else 'disabled'} "
        f"retry={'enabled' if settings.rag_query_expansion_retry_enabled else 'disabled'} "
        f"light_terms={settings.rag_query_expansion_max_added_terms_light} "
        f"targeted_terms={settings.rag_query_expansion_max_added_terms_targeted}"
    )
    print(f"  Tutor database: {settings.tutor_db_path.resolve()}")
    print(f"  Student ID: {settings.student_id}")
    print(f"  Logs: {settings.log_dir.resolve()}")
    print(f"  Conversation viewer: http://127.0.0.1:{args.port}/probe/conversations")
    print(f"  Interaction viewer: http://127.0.0.1:{args.port}/probe/interactions")
    if settings.log_content:
        print("  WARNING: prompts, answers, and retrieved book text are stored locally.")

    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
