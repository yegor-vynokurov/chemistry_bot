"""LangChain tools backed by the tutor graph SQLite store.

PR 3 introduced ``save_turn``. PR 4 added ``get_recent_history`` so the
standalone tutor can use SQLite as the conversation-context source.
PR 5 adds explicit ``get_session_state`` and ``update_session_state`` tools.
PR 8 adds ``save_decision`` and ``get_recent_decisions`` for an inspectable
classification/routing audit trail.

The tools are intentionally thin wrappers around ``SQLiteTutorStore``. They do
not import the older RAG bot, Prompt Garden, Chroma, or Wikipedia helpers.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..storage import SQLiteTutorStore


class SaveTurnToolInput(BaseModel):
    """Input schema for the save_turn tool."""

    student_id: str = Field(
        min_length=1,
        description="Stable student identifier, e.g. 'student_001'.",
    )
    session_id: str = Field(
        min_length=1,
        description="Stable session identifier, e.g. 'session_001'.",
    )
    role: Literal["student", "tutor", "system", "tool"] = Field(
        description="Who produced this turn."
    )
    text: str = Field(
        min_length=1,
        description="Human-readable turn text to persist.",
    )
    response_json: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured response payload, usually for tutor turns.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata such as node name, graph errors, or source tags.",
    )


class GetRecentHistoryToolInput(BaseModel):
    """Input schema for the get_recent_history tool."""

    session_id: str | None = Field(
        default=None,
        description=(
            "Session id to load turns from. Usually preferred for dialogue context."
        ),
    )
    student_id: str | None = Field(
        default=None,
        description=(
            "Optional student id. Use this alone to load cross-session history, "
            "or together with session_id as an additional filter."
        ),
    )
    limit: int = Field(
        default=12,
        ge=1,
        le=200,
        description="Maximum number of turns to return, in chronological order.",
    )
    before_turn_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional exclusive upper bound. Use it to load history before the "
            "current just-saved student turn, avoiding duplicate context."
        ),
    )
    roles: list[Literal["student", "tutor", "system", "tool"]] | None = Field(
        default=None,
        description="Optional role filter. Omit to return all roles.",
    )
    include_response_json: bool = Field(
        default=False,
        description="Whether to include stored structured tutor payloads.",
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include stored turn metadata.",
    )


class SaveDecisionToolInput(BaseModel):
    """Input schema for the save_decision tool."""

    student_id: str = Field(min_length=1, description="Stable student identifier.")
    session_id: str = Field(min_length=1, description="Stable session identifier.")
    node_name: str = Field(
        min_length=1,
        description="Graph node that produced the decision, e.g. classify_question.",
    )
    decision_type: str = Field(
        min_length=1,
        description=(
            "Decision category, e.g. classification, routing, safety_override, "
            "or clarification_fallback."
        ),
    )
    decision: dict[str, Any] | None = Field(
        default=None,
        description="Structured decision payload to persist.",
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional decision confidence."
    )
    reason: str | None = Field(
        default=None, description="Optional human-readable decision reason."
    )
    turn_id: int | None = Field(
        default=None,
        ge=1,
        description="Optional related student turn id.",
    )


class GetRecentDecisionsToolInput(BaseModel):
    """Input schema for the get_recent_decisions tool."""

    session_id: str | None = Field(
        default=None, description="Session whose decision rows should be loaded."
    )
    student_id: str | None = Field(
        default=None, description="Optional student filter, including cross-session use."
    )
    limit: int = Field(
        default=20, ge=1, le=200, description="Maximum decision rows to return."
    )
    before_decision_id: int | None = Field(
        default=None, ge=1, description="Optional exclusive decision id upper bound."
    )
    decision_types: list[str] | None = Field(
        default=None,
        description="Optional decision_type filter.",
    )


class GetSessionStateToolInput(BaseModel):
    """Input schema for the get_session_state tool."""

    session_id: str = Field(
        min_length=1,
        description="Session id whose persisted operational state should be loaded.",
    )


class UpdateSessionStateToolInput(BaseModel):
    """Input schema for the update_session_state tool."""

    session_id: str = Field(
        min_length=1,
        description="Session id whose operational state should be updated.",
    )
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON object merged into the existing session state. Use null values "
            "to intentionally clear a key without removing the key yet."
        ),
    )
    replace: bool = Field(
        default=False,
        description=(
            "If true, replace the whole persisted session state with patch. "
            "If false, shallow-merge patch into the current state."
        ),
    )


def build_save_turn_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a LangChain StructuredTool that saves one turn to SQLite.

    The returned tool is bound to the provided ``SQLiteTutorStore`` instance.
    It returns a small JSON-serializable dictionary so graph nodes and CLI code
    can log or inspect the persisted turn id.
    """

    def _save_turn(
        student_id: str,
        session_id: str,
        role: Literal["student", "tutor", "system", "tool"],
        text: str,
        response_json: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn_id = store.save_turn(
            student_id=student_id,
            session_id=session_id,
            role=role,
            text=text,
            response_json=response_json,
            metadata=metadata,
        )
        return {
            "ok": True,
            "tool": "save_turn",
            "turn_id": turn_id,
            "student_id": student_id,
            "session_id": session_id,
            "role": role,
        }

    return StructuredTool.from_function(
        func=_save_turn,
        name="save_turn",
        description=(
            "Persist one student, tutor, system, or tool turn to the tutor "
            "SQLite database and return the stored turn id."
        ),
        args_schema=SaveTurnToolInput,
    )


def build_get_recent_history_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a StructuredTool that reads recent dialogue turns from SQLite.

    This tool is intentionally read-only. It returns JSON-serializable rows so
    graph nodes can build context, CLI commands can preview history, and later
    routing/assessment nodes can request as many turns as they need.
    """

    def _get_recent_history(
        session_id: str | None = None,
        student_id: str | None = None,
        limit: int = 12,
        before_turn_id: int | None = None,
        roles: list[Literal["student", "tutor", "system", "tool"]] | None = None,
        include_response_json: bool = False,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        turns = store.get_recent_turns(
            student_id=student_id,
            session_id=session_id,
            limit=limit,
            before_turn_id=before_turn_id,
            roles=list(roles) if roles else None,
        )
        serialized: list[dict[str, Any]] = []
        for turn in turns:
            item: dict[str, Any] = {
                "id": turn.id,
                "session_id": turn.session_id,
                "student_id": turn.student_id,
                "role": turn.role,
                "text": turn.text,
                "created_at": turn.created_at,
            }
            if include_response_json:
                item["response_json"] = turn.response_json
            if include_metadata:
                item["metadata"] = turn.metadata
            serialized.append(item)
        return {
            "ok": True,
            "tool": "get_recent_history",
            "session_id": session_id,
            "student_id": student_id,
            "limit": limit,
            "before_turn_id": before_turn_id,
            "roles": roles,
            "turn_count": len(serialized),
            "turns": serialized,
        }

    return StructuredTool.from_function(
        func=_get_recent_history,
        name="get_recent_history",
        description=(
            "Load recent persisted tutor conversation turns from SQLite in "
            "chronological order. Use this as the dialogue-context source."
        ),
        args_schema=GetRecentHistoryToolInput,
    )


def build_save_decision_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a StructuredTool that persists one graph decision."""

    def _save_decision(
        student_id: str,
        session_id: str,
        node_name: str,
        decision_type: str,
        decision: dict[str, Any] | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        turn_id: int | None = None,
    ) -> dict[str, Any]:
        decision_id = store.save_decision(
            student_id=student_id,
            session_id=session_id,
            node_name=node_name,
            decision_type=decision_type,
            decision=decision,
            confidence=confidence,
            reason=reason,
            turn_id=turn_id,
        )
        return {
            "ok": True,
            "tool": "save_decision",
            "decision_id": decision_id,
            "student_id": student_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "decision_type": decision_type,
        }

    return StructuredTool.from_function(
        func=_save_decision,
        name="save_decision",
        description=(
            "Persist one classifier, routing, safety, or fallback decision to "
            "the tutor SQLite decision_log table."
        ),
        args_schema=SaveDecisionToolInput,
    )


def build_get_recent_decisions_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a read-only tool for inspecting persisted decision rows."""

    def _get_recent_decisions(
        session_id: str | None = None,
        student_id: str | None = None,
        limit: int = 20,
        before_decision_id: int | None = None,
        decision_types: list[str] | None = None,
    ) -> dict[str, Any]:
        rows = store.get_recent_decisions(
            student_id=student_id,
            session_id=session_id,
            limit=limit,
            before_decision_id=before_decision_id,
            decision_types=decision_types,
        )
        decisions = [
            {
                "id": row.id,
                "session_id": row.session_id,
                "student_id": row.student_id,
                "turn_id": row.turn_id,
                "node_name": row.node_name,
                "decision_type": row.decision_type,
                "decision": row.decision,
                "confidence": row.confidence,
                "reason": row.reason,
                "created_at": row.created_at,
            }
            for row in rows
        ]
        return {
            "ok": True,
            "tool": "get_recent_decisions",
            "session_id": session_id,
            "student_id": student_id,
            "limit": limit,
            "before_decision_id": before_decision_id,
            "decision_types": decision_types,
            "decision_count": len(decisions),
            "decisions": decisions,
        }

    return StructuredTool.from_function(
        func=_get_recent_decisions,
        name="get_recent_decisions",
        description=(
            "Load recent classification and routing audit rows from SQLite in "
            "chronological order."
        ),
        args_schema=GetRecentDecisionsToolInput,
    )


def build_get_session_state_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a StructuredTool that loads persisted session state from SQLite."""

    def _get_session_state(session_id: str) -> dict[str, Any]:
        state = store.get_session_state(session_id)
        return {
            "ok": True,
            "tool": "get_session_state",
            "session_id": session_id,
            "state": state,
        }

    return StructuredTool.from_function(
        func=_get_session_state,
        name="get_session_state",
        description=(
            "Load the current persisted operational state for a tutor session, "
            "for example active_concept_id, awaiting_student_answer, and last_bot_question."
        ),
        args_schema=GetSessionStateToolInput,
    )


def build_update_session_state_tool(store: SQLiteTutorStore) -> StructuredTool:
    """Build a StructuredTool that updates persisted session state in SQLite."""

    def _update_session_state(
        session_id: str,
        patch: dict[str, Any],
        replace: bool = False,
    ) -> dict[str, Any]:
        state = store.update_session_state(
            session_id=session_id,
            patch=patch,
            replace=replace,
        )
        return {
            "ok": True,
            "tool": "update_session_state",
            "session_id": session_id,
            "replace": replace,
            "state": state,
        }

    return StructuredTool.from_function(
        func=_update_session_state,
        name="update_session_state",
        description=(
            "Shallow-merge a JSON patch into the persisted tutor session state, "
            "or replace the full state when replace=true."
        ),
        args_schema=UpdateSessionStateToolInput,
    )
