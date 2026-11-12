"""CLI entry point for the standalone LangGraph chemistry tutor bot.

Run from the project root:

    python -m chemistry_bot.tutor_graph.app

This bot is intentionally independent from ``chemistry_bot.cli.rag_bot``. It
does not import PromptGarden, Chroma RAG, Wikipedia helpers, or CLI bot modules.

PR 6 adds a separate lightweight LLM question classifier. PR 6.1 adds
classifier calibration/post-processing. PR 7 adds deterministic LangGraph
routing. I0.3.1B keeps classifier and tutor outputs thin: the classifier returns one label,
the tutor model returns plain text, and Python assembles ``TutorAnswer`` plus an
optional separately generated check question. Deterministic decisions remain in ``decision_log``. SQLite remains the normal source for conversation history, and
``session_state`` becomes the small operational memory that tells the tutor
whether the student is answering a previous question, which concept is active,
and what tutor move was used last.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .decision_logging import build_decision_log_events
from .graph import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASSIFIER_MODEL,
    TutorGraphConfig,
    build_tutor_graph,
)
from .schemas import QuestionClassification, RoutingDecision, TutorAnswer
from .storage import SQLiteTutorStore
from .storage.sqlite_store import DEFAULT_DB_PATH
from .tools import (
    build_get_recent_decisions_tool,
    build_get_recent_history_tool,
    build_get_session_state_tool,
    build_save_decision_tool,
    build_save_turn_tool,
    build_update_session_state_tool,
)


DEFAULT_CONTEXT_TURNS = 12


def _json_default(value: Any) -> Any:
    if isinstance(value, (TutorAnswer, QuestionClassification, RoutingDecision)):
        return value.model_dump()
    if isinstance(value, BaseMessage):
        return {"type": value.type, "content": value.content}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone LangGraph chemistry tutor bot with SQLite storage."
    )
    parser.add_argument(
        "--model",
        default="gemma4:12b",
        help="Ollama model name. Default: gemma4:12b",
    )
    parser.add_argument(
        "--tutor-num-ctx",
        type=int,
        default=8192,
        help="Context window for tutor answer generation. Default: 8192.",
    )
    parser.add_argument(
        "--tutor-num-predict",
        type=int,
        default=1536,
        help="Maximum generated tokens for tutor answers. Default: 1536.",
    )
    parser.add_argument(
        "--classifier-model",
        default=DEFAULT_CLASSIFIER_MODEL,
        help=(
            "Small Ollama model used only for question classification. "
            f"Default: {DEFAULT_CLASSIFIER_MODEL}. Good local choices are "
            "qwen2.5:1.5b or qwen2.5:3b."
        ),
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Disable the LLM classifier and use the rule-based fallback classifier.",
    )
    parser.add_argument(
        "--turn-relation-model",
        default="",
        help=(
            "Small Ollama model used only for student_answer / clarification_choice / "
            "new_question / unclear. Default: the selected classifier model."
        ),
    )
    parser.add_argument(
        "--turn-relation-num-ctx",
        type=int,
        default=2048,
        help="Context window for the narrow turn-relation call. Default: 2048.",
    )
    parser.add_argument(
        "--turn-relation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the separate one-label turn-relation call.",
    )
    parser.add_argument(
        "--semantic-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable embedding similarity in the safety gate.",
    )
    parser.add_argument(
        "--safety-embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model used for safety prototypes.",
    )
    parser.add_argument(
        "--safety-hazard-threshold",
        type=float,
        default=0.60,
        help="Cosine threshold for hazard prototypes.",
    )
    parser.add_argument(
        "--safety-procedure-threshold",
        type=float,
        default=0.60,
        help="Cosine threshold for procedural-intent prototypes.",
    )
    parser.add_argument(
        "--safety-fuzzy-threshold",
        type=float,
        default=86.0,
        help="RapidFuzz threshold from 0 to 100 for typo-tolerant aliases.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        help=(
            "Deprecated compatibility option; the thin classifier no longer emits confidence. "
            f"Default: {DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--needs-documents-dev",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Developer override. When enabled, needs_user_documents is true for every "
            "classification; protected routes can still block actual retrieval."
        ),
    )
    parser.add_argument(
        "--followup-model",
        default="",
        help=(
            "Small Ollama model used only to write one optional check question. "
            "Default: the selected classifier model."
        ),
    )
    parser.add_argument(
        "--followup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the separate small-model follow-up question call.",
    )
    parser.add_argument(
        "--student-id",
        default="student_001",
        help="Student id used in graph state and SQLite storage.",
    )
    parser.add_argument(
        "--session-id",
        default="session_001",
        help="Session id used in graph state and SQLite storage.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Run without SQLite persistence. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=DEFAULT_CONTEXT_TURNS,
        help=(
            "Number of previous persisted turns to load as model context. "
            f"Default: {DEFAULT_CONTEXT_TURNS}."
        ),
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the programmatically assembled TutorAnswer JSON after each turn.",
    )
    return parser


def _parse_positive_int(text: str, *, default: int, minimum: int = 1, maximum: int = 200) -> int:
    if not text.strip():
        return default
    value = int(text.strip())
    if value < minimum or value > maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return value


def _print_db_status(store: SQLiteTutorStore | None) -> None:
    if store is None:
        print("Bot: SQLite storage is disabled for this run.")
        return
    print("\nSQLite tutor storage:")
    print(json.dumps(store.status().as_dict(), indent=2, ensure_ascii=False))
    print()


def _print_tools(
    save_turn_tool: Any | None,
    get_recent_history_tool: Any | None,
    get_session_state_tool: Any | None,
    update_session_state_tool: Any | None,
    save_decision_tool: Any | None,
    get_recent_decisions_tool: Any | None,
) -> None:
    print("\nAvailable tutor tools:")
    tool_pairs = [
        ("save_turn", save_turn_tool),
        ("get_recent_history", get_recent_history_tool),
        ("get_session_state", get_session_state_tool),
        ("update_session_state", update_session_state_tool),
        ("save_decision", save_decision_tool),
        ("get_recent_decisions", get_recent_decisions_tool),
    ]
    for expected_name, tool in tool_pairs:
        if tool is None:
            print(f"  {expected_name}: disabled because SQLite storage is disabled")
        else:
            print(f"  {tool.name}: {tool.description}")
    print()


def _call_save_turn_tool(
    save_turn_tool: Any | None,
    *,
    student_id: str,
    session_id: str,
    role: str,
    text: str,
    response_json: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Persist a turn through the explicit save_turn tool."""
    if save_turn_tool is None:
        return None

    result = save_turn_tool.invoke(
        {
            "student_id": student_id,
            "session_id": session_id,
            "role": role,
            "text": text,
            "response_json": response_json,
            "metadata": metadata or {},
        }
    )
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"save_turn tool returned an unexpected result: {result!r}")
    return int(result["turn_id"])


def _call_get_recent_history_tool(
    get_recent_history_tool: Any | None,
    *,
    session_id: str,
    student_id: str | None = None,
    limit: int = DEFAULT_CONTEXT_TURNS,
    before_turn_id: int | None = None,
    roles: list[str] | None = None,
    include_response_json: bool = False,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Load persisted history through the explicit get_recent_history tool."""
    if get_recent_history_tool is None:
        return {"ok": False, "turns": [], "turn_count": 0, "tool": "get_recent_history"}

    result = get_recent_history_tool.invoke(
        {
            "session_id": session_id,
            "student_id": student_id,
            "limit": limit,
            "before_turn_id": before_turn_id,
            "roles": roles,
            "include_response_json": include_response_json,
            "include_metadata": include_metadata,
        }
    )
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            f"get_recent_history tool returned an unexpected result: {result!r}"
        )
    return result


def _call_get_session_state_tool(
    get_session_state_tool: Any | None,
    *,
    session_id: str,
) -> dict[str, Any]:
    """Load persisted operational session state through a tool."""
    if get_session_state_tool is None:
        return {"ok": False, "tool": "get_session_state", "session_id": session_id, "state": {}}

    result = get_session_state_tool.invoke({"session_id": session_id})
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            f"get_session_state tool returned an unexpected result: {result!r}"
        )
    return result


def _call_update_session_state_tool(
    update_session_state_tool: Any | None,
    *,
    session_id: str,
    patch: dict[str, Any],
    replace: bool = False,
) -> dict[str, Any]:
    """Update persisted operational session state through a tool."""
    if update_session_state_tool is None:
        return {
            "ok": False,
            "tool": "update_session_state",
            "session_id": session_id,
            "replace": replace,
            "state": patch if replace else {},
        }

    result = update_session_state_tool.invoke(
        {"session_id": session_id, "patch": patch, "replace": replace}
    )
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            f"update_session_state tool returned an unexpected result: {result!r}"
        )
    return result


def _call_save_decision_tool(
    save_decision_tool: Any | None,
    *,
    student_id: str,
    session_id: str,
    node_name: str,
    decision_type: str,
    decision: dict[str, Any],
    confidence: float | None,
    reason: str | None,
    turn_id: int | None,
) -> int | None:
    """Persist one graph decision through the explicit save_decision tool."""
    if save_decision_tool is None:
        return None
    result = save_decision_tool.invoke(
        {
            "student_id": student_id,
            "session_id": session_id,
            "node_name": node_name,
            "decision_type": decision_type,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "turn_id": turn_id,
        }
    )
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            f"save_decision tool returned an unexpected result: {result!r}"
        )
    return int(result["decision_id"])


def _call_get_recent_decisions_tool(
    get_recent_decisions_tool: Any | None,
    *,
    session_id: str,
    limit: int = 20,
    decision_types: list[str] | None = None,
) -> dict[str, Any]:
    """Load decision-log rows through the explicit read tool."""
    if get_recent_decisions_tool is None:
        return {
            "ok": False,
            "tool": "get_recent_decisions",
            "decision_count": 0,
            "decisions": [],
        }
    result = get_recent_decisions_tool.invoke(
        {
            "session_id": session_id,
            "student_id": None,
            "limit": limit,
            "before_decision_id": None,
            "decision_types": decision_types,
        }
    )
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            f"get_recent_decisions tool returned an unexpected result: {result!r}"
        )
    return result


def _persist_graph_decisions(
    save_decision_tool: Any | None,
    *,
    student_id: str,
    session_id: str,
    turn_id: int | None,
    classification: QuestionClassification | None,
    routing: RoutingDecision | None,
    debug: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist one turn's audit events without blocking the tutor answer."""
    if save_decision_tool is None:
        return {"ok": False, "disabled": True, "decision_ids": [], "error": None}
    if not isinstance(classification, QuestionClassification) or not isinstance(
        routing, RoutingDecision
    ):
        return {
            "ok": False,
            "disabled": False,
            "decision_ids": [],
            "error": "Missing valid classification or routing decision.",
        }

    decision_ids: list[int] = []
    try:
        events = build_decision_log_events(
            classification=classification, routing=routing, debug=debug
        )
        for event in events:
            decision_id = _call_save_decision_tool(
                save_decision_tool,
                student_id=student_id,
                session_id=session_id,
                node_name=event.node_name,
                decision_type=event.decision_type,
                decision=event.decision,
                confidence=event.confidence,
                reason=event.reason,
                turn_id=turn_id,
            )
            if decision_id is not None:
                decision_ids.append(decision_id)
        return {
            "ok": True,
            "disabled": False,
            "decision_ids": decision_ids,
            "decision_types": [event.decision_type for event in events],
            "error": None,
        }
    except Exception as error:  # Decision logging must not hide the tutor answer.
        return {
            "ok": False,
            "disabled": False,
            "decision_ids": decision_ids,
            "error": str(error),
        }


def _history_turns_to_messages(turns: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert stored student/tutor turns into LangChain chat messages.

    System/tool rows are intentionally ignored for model dialogue context at
    this stage. They stay available in SQLite and can be loaded later for debug
    or analysis.
    """
    messages: list[BaseMessage] = []
    for turn in turns:
        role = turn.get("role")
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        if role == "student":
            messages.append(HumanMessage(content=text))
        elif role == "tutor":
            messages.append(AIMessage(content=text))
    return messages


def _print_history_from_tool(
    get_recent_history_tool: Any | None,
    *,
    session_id: str,
    limit: int,
) -> None:
    if get_recent_history_tool is None:
        print("Bot: SQLite storage is disabled, so there is no persisted history.")
        return
    result = _call_get_recent_history_tool(
        get_recent_history_tool,
        session_id=session_id,
        limit=limit,
        roles=["student", "tutor", "system", "tool"],
    )
    turns = result.get("turns", [])
    if not turns:
        print("Bot: No persisted turns for this session yet.")
        return
    print(f"\nRecent persisted turns, limit={limit}:")
    for turn in turns:
        print(f"  [{turn['created_at']}] #{turn['id']} {turn['role']}: {turn['text']}")
    print()


def _print_decisions_from_tool(
    get_recent_decisions_tool: Any | None,
    *,
    session_id: str,
    limit: int,
) -> None:
    if get_recent_decisions_tool is None:
        print("Bot: SQLite storage is disabled, so there is no decision log.")
        return
    result = _call_get_recent_decisions_tool(
        get_recent_decisions_tool, session_id=session_id, limit=limit
    )
    decisions = result.get("decisions", [])
    if not decisions:
        print("Bot: No persisted decisions for this session yet.")
        return
    print(f"\nRecent decision_log rows, limit={limit}:")
    for row in decisions:
        confidence = row.get("confidence")
        confidence_text = f" confidence={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        print(
            f"  [{row['created_at']}] #{row['id']} turn={row.get('turn_id')} "
            f"{row['decision_type']} @ {row['node_name']}{confidence_text}"
        )
        if row.get("reason"):
            print(f"    reason: {row['reason']}")
        print(
            "    decision: "
            + json.dumps(row.get("decision", {}), ensure_ascii=False, default=_json_default)
        )
    print()


def _print_session_state_from_tool(
    get_session_state_tool: Any | None,
    *,
    session_id: str,
) -> None:
    if get_session_state_tool is None:
        print("Bot: SQLite storage is disabled, so there is no persisted session state.")
        return
    result = _call_get_session_state_tool(
        get_session_state_tool,
        session_id=session_id,
    )
    print("\nPersisted session_state:")
    print(json.dumps(result.get("state", {}), indent=2, ensure_ascii=False))
    print()


def _print_last_classification(last_state: dict[str, Any] | None) -> None:
    if last_state is None:
        print("Bot: No classification yet.")
        return
    classification = last_state.get("classification")
    if isinstance(classification, QuestionClassification):
        print("\nLast question classification:")
        print(classification.model_dump_json(indent=2))
    elif classification:
        print("\nLast question classification:")
        print(json.dumps(classification, indent=2, ensure_ascii=False, default=_json_default))
    else:
        print("Bot: Last graph state has no classification.")
    debug = last_state.get("debug", {})
    if debug:
        keys = [
            "classification_source",
            "classifier_model_name",
            "classification_error",
            "classifier_contract",
            "classifier_raw_output",
        ]
        compact = {key: debug.get(key) for key in keys if key in debug}
        if compact:
            print("\nClassification debug:")
            print(json.dumps(compact, indent=2, ensure_ascii=False))

        calibration = debug.get("classification_derivation")
        if isinstance(calibration, dict):
            print("\nClassification derivation:")
            print(json.dumps(calibration, indent=2, ensure_ascii=False))

        classifier_input = debug.get("classifier_input_payload")
        if isinstance(classifier_input, dict):
            print("\nClassifier input payload:")
            print(json.dumps(classifier_input, indent=2, ensure_ascii=False))
    print()



def _print_last_route(last_state: dict[str, Any] | None) -> None:
    if last_state is None:
        print("Bot: No routing decision yet.")
        return
    routing = last_state.get("routing")
    if isinstance(routing, RoutingDecision):
        print("\nLast routing decision:")
        print(routing.model_dump_json(indent=2))
    elif routing:
        print("\nLast routing decision:")
        print(json.dumps(routing, indent=2, ensure_ascii=False, default=_json_default))
    else:
        print("Bot: Last graph state has no routing decision.")
    debug = last_state.get("debug", {})
    compact = {
        key: debug.get(key)
        for key in [
            "routing_node",
            "selected_route",
            "routing_reason",
            "routing_clarification_fallback",
            "routing_safety_override",
            "node",
        ]
        if key in debug
    }
    if compact:
        print("\nRouting debug:")
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    print()

def _derive_session_state_patch_from_answer(
    answer: TutorAnswer,
    *,
    routing: RoutingDecision | None,
    context_source: str,
    context_turn_limit: int,
    loaded_history_turns: int,
) -> dict[str, Any]:
    """Create the operational session-state patch after a tutor answer."""
    next_question = answer.next_question.strip() if answer.next_question else None
    active_concept_id = answer.concept_ids[0] if answer.concept_ids else None
    return {
        "awaiting_student_answer": bool(next_question),
        "last_bot_question": next_question,
        "active_concept_id": active_concept_id,
        "last_tutor_move": answer.tutor_move,
        "last_concept_ids": answer.concept_ids,
        "last_route": routing.route if routing else None,
        "last_route_reason": routing.reason if routing else None,
        "last_context_source": context_source,
        "last_context_turn_limit": context_turn_limit,
        "last_loaded_history_turns": loaded_history_turns,
    }


def main() -> None:
    args = build_parser().parse_args()

    threshold = max(0.0, min(1.0, float(args.classification_threshold)))
    config = TutorGraphConfig(
        model_name=args.model,
        tutor_num_ctx=max(1024, int(args.tutor_num_ctx)),
        tutor_num_predict=max(128, int(args.tutor_num_predict)),
        classifier_enabled=not args.no_classifier,
        classifier_model_name=args.classifier_model,
        classification_confidence_threshold=threshold,
        turn_relation_enabled=bool(args.turn_relation),
        turn_relation_model_name=(args.turn_relation_model.strip() or args.classifier_model),
        turn_relation_num_ctx=max(512, int(args.turn_relation_num_ctx)),
        semantic_safety_enabled=bool(args.semantic_safety),
        safety_embedding_model_name=(args.safety_embedding_model.strip() or "nomic-embed-text"),
        safety_hazard_threshold=max(-1.0, min(1.0, float(args.safety_hazard_threshold))),
        safety_procedure_threshold=max(-1.0, min(1.0, float(args.safety_procedure_threshold))),
        safety_fuzzy_threshold=max(0.0, min(100.0, float(args.safety_fuzzy_threshold))),
        needs_documents_dev=bool(args.needs_documents_dev),
        followup_enabled=bool(args.followup),
        followup_model_name=(args.followup_model.strip() or args.classifier_model),
    )
    graph = build_tutor_graph(config)

    store: SQLiteTutorStore | None = None
    save_turn_tool: Any | None = None
    get_recent_history_tool: Any | None = None
    get_session_state_tool: Any | None = None
    update_session_state_tool: Any | None = None
    save_decision_tool: Any | None = None
    get_recent_decisions_tool: Any | None = None
    if not args.no_db:
        store = SQLiteTutorStore(Path(args.db_path))
        save_turn_tool = build_save_turn_tool(store)
        get_recent_history_tool = build_get_recent_history_tool(store)
        get_session_state_tool = build_get_session_state_tool(store)
        update_session_state_tool = build_update_session_state_tool(store)
        save_decision_tool = build_save_decision_tool(store)
        get_recent_decisions_tool = build_get_recent_decisions_tool(store)
        store.ensure_session(
            session_id=args.session_id,
            student_id=args.student_id,
            title="CLI tutor graph session",
        )
        _call_update_session_state_tool(
            update_session_state_tool,
            session_id=args.session_id,
            patch={
                "student_id": args.student_id,
                "session_id": args.session_id,
                "model_name": args.model,
                "storage_pr": "PR 8",
                "context_source": "sqlite_get_recent_history_tool",
                "context_turn_limit": args.context_turns,
                "session_state_source": "sqlite_session_state_tools",
                "classifier_enabled": not args.no_classifier,
                "classifier_model_name": args.classifier_model,
                "classifier_contract": "main_type_only",
                "turn_relation_enabled": bool(args.turn_relation),
                "turn_relation_model_name": (args.turn_relation_model.strip() or args.classifier_model),
                "turn_relation_num_ctx": max(512, int(args.turn_relation_num_ctx)),
                "turn_relation_contract": "one_state_allowed_label_when_reply_awaited",
                "semantic_safety_enabled": bool(args.semantic_safety),
                "safety_embedding_model_name": (args.safety_embedding_model.strip() or "nomic-embed-text"),
                "safety_hazard_threshold": max(-1.0, min(1.0, float(args.safety_hazard_threshold))),
                "safety_procedure_threshold": max(-1.0, min(1.0, float(args.safety_procedure_threshold))),
                "safety_fuzzy_threshold": max(0.0, min(100.0, float(args.safety_fuzzy_threshold))),
                "tutor_output_contract": "plain_text_only",
                "needs_documents_dev": bool(args.needs_documents_dev),
                "followup_enabled": bool(args.followup),
                "followup_model_name": (args.followup_model.strip() or args.classifier_model),
                "classification_threshold_deprecated": threshold,
            },
        )

    # When SQLite is enabled, this list is only a no-db fallback. The normal
    # path loads dialogue context from SQLite through get_recent_history.
    in_memory_messages: list[BaseMessage] = []
    in_memory_session_state: dict[str, Any] = {}
    last_state: dict[str, Any] | None = None
    context_turn_limit = args.context_turns

    print("Chemistry Tutor Graph Bot")
    print("I0.3.1B.4: plain-text reliability, state-constrained turn relation, semantic safety gate.")
    print("Commands:")
    print("  /exit                 quit")
    print("  /reset                clear no-db history; persisted turns stay; state is marked reset")
    print("  /debug                print last graph state")
    print("  /classification       print last thin label, derived policy, and classifier input")
    print("  /route                print last deterministic routing decision")
    print("  /db                   show SQLite storage status")
    print("  /history [n]          show recent persisted turns")
    print("  /decisions [n]        show recent classification/routing audit rows")
    print("  /state                show persisted session_state")
    print("  /state-reset          replace session_state with a small clean state")
    print("  /state-patch <json>   shallow-merge a JSON object into session_state")
    print("  /tools                show available tutor tools")
    print("  /context-turns <n>    change how many DB turns are sent as context")
    print("  /classifier           show classifier settings")
    print()
    _print_db_status(store)

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if not user_text:
            continue

        command = user_text.lower()
        if command in {"/exit", "exit", "quit", "stop"}:
            print("Bot: Goodbye!")
            break

        if command == "/reset":
            in_memory_messages = []
            in_memory_session_state = {}
            last_state = None
            _call_update_session_state_tool(
                update_session_state_tool,
                session_id=args.session_id,
                patch={
                    "in_memory_history_reset": True,
                    "awaiting_student_answer": False,
                    "last_bot_question": None,
                },
            )
            print("Bot: In-memory fallback history cleared. Persisted history is unchanged.")
            continue

        if command == "/debug":
            if last_state is None:
                print("Bot: No graph state yet.")
            else:
                print(json.dumps(last_state, indent=2, ensure_ascii=False, default=_json_default))
            continue

        if command == "/classification":
            _print_last_classification(last_state)
            continue

        if command == "/route":
            _print_last_route(last_state)
            continue

        if command == "/classifier":
            print("\nQuestion classifier settings:")
            print(json.dumps({
                "enabled": not args.no_classifier,
                "model": args.classifier_model if not args.no_classifier else None,
                "fallback_to_rules": True,
                "confidence_threshold": threshold,
            }, indent=2, ensure_ascii=False))
            print()
            continue

        if command == "/db":
            _print_db_status(store)
            continue

        if command.startswith("/history"):
            raw_limit = user_text[len("/history"):].strip()
            try:
                limit = _parse_positive_int(raw_limit, default=10)
            except ValueError as error:
                print(f"Bot: invalid /history limit: {error}")
                continue
            _print_history_from_tool(
                get_recent_history_tool,
                session_id=args.session_id,
                limit=limit,
            )
            continue

        if command.startswith("/decisions"):
            raw_limit = user_text[len("/decisions"):].strip()
            try:
                limit = _parse_positive_int(raw_limit, default=20)
            except ValueError as error:
                print(f"Bot: invalid /decisions limit: {error}")
                continue
            _print_decisions_from_tool(
                get_recent_decisions_tool,
                session_id=args.session_id,
                limit=limit,
            )
            continue

        if command == "/state":
            if get_session_state_tool is None:
                print("\nIn-memory fallback session_state:")
                print(json.dumps(in_memory_session_state, indent=2, ensure_ascii=False))
                print()
            else:
                _print_session_state_from_tool(
                    get_session_state_tool,
                    session_id=args.session_id,
                )
            continue

        if command == "/state-reset":
            clean_state = {
                "student_id": args.student_id,
                "session_id": args.session_id,
                "model_name": args.model,
                "storage_pr": "PR 8",
                "context_source": "sqlite_get_recent_history_tool",
                "context_turn_limit": context_turn_limit,
                "session_state_source": "sqlite_session_state_tools",
                "classifier_enabled": not args.no_classifier,
                "classifier_model_name": args.classifier_model,
                "classifier_contract": "main_type_only",
                "turn_relation_enabled": bool(args.turn_relation),
                "turn_relation_model_name": (args.turn_relation_model.strip() or args.classifier_model),
                "turn_relation_num_ctx": max(512, int(args.turn_relation_num_ctx)),
                "turn_relation_contract": "one_state_allowed_label_when_reply_awaited",
                "semantic_safety_enabled": bool(args.semantic_safety),
                "safety_embedding_model_name": (args.safety_embedding_model.strip() or "nomic-embed-text"),
                "safety_hazard_threshold": max(-1.0, min(1.0, float(args.safety_hazard_threshold))),
                "safety_procedure_threshold": max(-1.0, min(1.0, float(args.safety_procedure_threshold))),
                "safety_fuzzy_threshold": max(0.0, min(100.0, float(args.safety_fuzzy_threshold))),
                "tutor_output_contract": "plain_text_only",
                "needs_documents_dev": bool(args.needs_documents_dev),
                "followup_enabled": bool(args.followup),
                "followup_model_name": (args.followup_model.strip() or args.classifier_model),
                "classification_threshold_deprecated": threshold,
                "awaiting_student_answer": False,
                "last_bot_question": None,
                "active_concept_id": None,
                "last_tutor_move": None,
            }
            if update_session_state_tool is None:
                in_memory_session_state = clean_state
            else:
                _call_update_session_state_tool(
                    update_session_state_tool,
                    session_id=args.session_id,
                    patch=clean_state,
                    replace=True,
                )
            print("Bot: session_state reset to a clean PR 8 state.")
            continue

        if command.startswith("/state-patch "):
            raw_json = user_text[len("/state-patch "):].strip()
            try:
                patch = json.loads(raw_json)
                if not isinstance(patch, dict):
                    raise ValueError("state patch must be a JSON object")
            except (json.JSONDecodeError, ValueError) as error:
                print(f"Bot: invalid /state-patch JSON: {error}")
                continue
            if update_session_state_tool is None:
                in_memory_session_state.update(patch)
                updated_state = in_memory_session_state
            else:
                result = _call_update_session_state_tool(
                    update_session_state_tool,
                    session_id=args.session_id,
                    patch=patch,
                )
                updated_state = result.get("state", {})
            print("Bot: session_state patched:")
            print(json.dumps(updated_state, indent=2, ensure_ascii=False))
            continue

        if command == "/tools":
            _print_tools(
                save_turn_tool,
                get_recent_history_tool,
                get_session_state_tool,
                update_session_state_tool,
                save_decision_tool,
                get_recent_decisions_tool,
            )
            continue

        if command.startswith("/context-turns "):
            raw_limit = user_text[len("/context-turns "):].strip()
            try:
                context_turn_limit = _parse_positive_int(raw_limit, default=context_turn_limit)
            except ValueError as error:
                print(f"Bot: invalid /context-turns value: {error}")
                continue
            if update_session_state_tool is None:
                in_memory_session_state["context_turn_limit"] = context_turn_limit
            else:
                _call_update_session_state_tool(
                    update_session_state_tool,
                    session_id=args.session_id,
                    patch={"context_turn_limit": context_turn_limit},
                )
            print(f"Bot: DB context turn limit set to {context_turn_limit}.")
            continue

        user_turn_id: int | None = _call_save_turn_tool(
            save_turn_tool,
            student_id=args.student_id,
            session_id=args.session_id,
            role="student",
            text=user_text,
            metadata={"source": "cli", "tool_call": "save_turn"},
        )

        history_result: dict[str, Any] | None = None
        if get_recent_history_tool is not None:
            history_result = _call_get_recent_history_tool(
                get_recent_history_tool,
                session_id=args.session_id,
                limit=context_turn_limit,
                before_turn_id=user_turn_id,
                roles=["student", "tutor"],
            )
            graph_messages = _history_turns_to_messages(history_result.get("turns", []))
            context_source = "sqlite_get_recent_history_tool"
        else:
            graph_messages = list(in_memory_messages)
            context_source = "in_memory_fallback"

        if get_session_state_tool is not None:
            session_state_result = _call_get_session_state_tool(
                get_session_state_tool,
                session_id=args.session_id,
            )
            session_state = dict(session_state_result.get("state", {}))
        else:
            session_state_result = {"ok": False, "tool": "get_session_state", "state": in_memory_session_state}
            session_state = dict(in_memory_session_state)

        state = {
            "student_id": args.student_id,
            "session_id": args.session_id,
            "user_text": user_text,
            "messages": graph_messages,
            "session_state": session_state,
            "classification": None,
            "routing": None,
            "response": None,
            "error": None,
            "debug": {
                "context_source": context_source,
                "context_turn_limit": context_turn_limit,
                "history_turn_count": (
                    history_result.get("turn_count", 0) if history_result else len(graph_messages)
                ),
                "current_user_turn_id": user_turn_id,
                "session_state_source": session_state_result.get("tool"),
                "session_state_keys": sorted(session_state.keys()),
            },
        }

        result = graph.invoke(state)
        last_state = dict(result)
        if get_recent_history_tool is None:
            in_memory_messages = list(result.get("messages", []))

        answer = result.get("response")
        classification = result.get("classification")
        routing = result.get("routing")
        decision_log_result = _persist_graph_decisions(
            save_decision_tool,
            student_id=args.student_id,
            session_id=args.session_id,
            turn_id=user_turn_id,
            classification=(
                classification if isinstance(classification, QuestionClassification) else None
            ),
            routing=routing if isinstance(routing, RoutingDecision) else None,
            debug=result.get("debug", {}),
        )
        result_debug = dict(result.get("debug", {}))
        result_debug["decision_log"] = decision_log_result
        result["debug"] = result_debug
        last_state = dict(result)

        if isinstance(answer, TutorAnswer):
            print("\nBot:")
            print(answer.to_cli_text())
            if args.show_json:
                if isinstance(classification, QuestionClassification):
                    print("\nQuestionClassification:")
                    print(classification.model_dump_json(indent=2))
                if isinstance(routing, RoutingDecision):
                    print("\nRoutingDecision:")
                    print(routing.model_dump_json(indent=2))
                print("\nAssembled TutorAnswer:")
                print(answer.model_dump_json(indent=2))
            if result.get("error"):
                print("\nWarning: the tutor model reported an error; see /debug.")
            print()

            loaded_turn_count = history_result.get("turn_count", 0) if history_result else 0
            tutor_state_patch = _derive_session_state_patch_from_answer(
                answer,
                routing=routing if isinstance(routing, RoutingDecision) else None,
                context_source=context_source,
                context_turn_limit=context_turn_limit,
                loaded_history_turns=loaded_turn_count,
            )
            if isinstance(classification, QuestionClassification):
                tutor_state_patch.update(
                    {
                        "last_question_classification": classification.main_type,
                        "last_classification_safety_level": classification.safety_level,
                        "last_needs_course_rag": classification.needs_course_rag,
                        "last_needs_user_documents": classification.needs_user_documents,
                        "last_needs_wikipedia": classification.needs_wikipedia,
                        "last_needs_diagnostic_question": classification.needs_diagnostic_question,
                    }
                )

            if save_turn_tool is not None:
                tutor_turn_id = _call_save_turn_tool(
                    save_turn_tool,
                    student_id=args.student_id,
                    session_id=args.session_id,
                    role="tutor",
                    text=answer.to_cli_text(),
                    response_json=answer.model_dump(),
                    metadata={
                        "source": "graph",
                        "user_turn_id": user_turn_id,
                        "graph_error": result.get("error"),
                        "debug": result.get("debug", {}),
                        "classification": (
                            classification.model_dump()
                            if isinstance(classification, QuestionClassification)
                            else None
                        ),
                        "routing": (
                            routing.model_dump()
                            if isinstance(routing, RoutingDecision)
                            else None
                        ),
                        "context_source": context_source,
                        "history_result": history_result,
                        "session_state_before": session_state,
                        "session_state_patch": tutor_state_patch,
                        "decision_log": decision_log_result,
                        "tool_call": "save_turn",
                    },
                )
                tutor_state_patch["last_tutor_turn_id"] = tutor_turn_id
                tutor_state_patch["last_student_turn_id"] = user_turn_id
                _call_update_session_state_tool(
                    update_session_state_tool,
                    session_id=args.session_id,
                    patch=tutor_state_patch,
                )
            else:
                in_memory_session_state.update(tutor_state_patch)
        else:
            print("\nBot: I did not get a valid TutorAnswer from the graph. Use /debug for details.\n")
            _call_save_turn_tool(
                save_turn_tool,
                student_id=args.student_id,
                session_id=args.session_id,
                role="system",
                text="Graph returned no valid TutorAnswer.",
                metadata={"result": result, "tool_call": "save_turn"},
            )


if __name__ == "__main__":
    main()
