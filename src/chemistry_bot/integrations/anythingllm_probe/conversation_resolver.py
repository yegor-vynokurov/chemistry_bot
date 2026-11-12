"""Resolve AnythingLLM chat requests to durable tutor sessions.

AnythingLLM Desktop's Generic OpenAI request does not currently expose a
thread identifier. I0.2A therefore resolves a conversation from the ordered
``user``/``assistant`` history that AnythingLLM includes in ``messages``.

The resolver is deliberately deterministic:

1. Ignore system/tool scaffolding for conversation identity.
2. Treat the last user message as the current turn.
3. Match the preceding user/assistant transcript against the suffix of recent
   sessions for the same student.
4. Reuse an exact match, otherwise create a new internal session.

Semantic similarity is intentionally not used here. Similar content may be
useful memory, but it does not prove that two requests belong to the same chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Any, Callable, Iterable, Sequence
from uuid import uuid4

from chemistry_bot.tutor_graph.storage import SQLiteTutorStore, StoredSession, StoredTurn


DEFAULT_STUDENT_ID = "anythingllm-desktop-local"
DEFAULT_MAX_CANDIDATE_SESSIONS = 200
DEFAULT_MAX_MATCH_MESSAGES = 64


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One normalized conversation message used for session identity."""

    role: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}


@dataclass(frozen=True, slots=True)
class ParsedConversation:
    """Conversation-bearing parts extracted from an OpenAI messages array."""

    history: tuple[ConversationMessage, ...]
    current_user_text: str
    ignored_roles: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "history": [item.as_dict() for item in self.history],
            "current_user_text": self.current_user_text,
            "ignored_roles": list(self.ignored_roles),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParsedConversation":
        raw_history = value.get("history")
        history: list[ConversationMessage] = []
        if isinstance(raw_history, list):
            for item in raw_history:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                text = item.get("text")
                if isinstance(role, str) and isinstance(text, str):
                    history.append(ConversationMessage(role=role, text=text))
        current_user_text = value.get("current_user_text")
        if not isinstance(current_user_text, str) or not current_user_text.strip():
            raise ConversationResolutionError(
                "Stored interaction is missing current_user_text."
            )
        raw_ignored = value.get("ignored_roles")
        ignored = tuple(str(item) for item in raw_ignored) if isinstance(raw_ignored, list) else ()
        return cls(
            history=tuple(history),
            current_user_text=current_user_text,
            ignored_roles=ignored,
        )


@dataclass(frozen=True, slots=True)
class ConversationResolution:
    """Auditable result of mapping one request to a tutor session."""

    session_id: str
    student_id: str
    resolution_method: str
    created_new_session: bool
    matched_turn_count: int
    candidate_count: int
    history_message_count: int
    history_fingerprint: str
    current_user_fingerprint: str
    confidence: float
    selected_session_updated_at: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "student_id": self.student_id,
            "resolution_method": self.resolution_method,
            "created_new_session": self.created_new_session,
            "matched_turn_count": self.matched_turn_count,
            "candidate_count": self.candidate_count,
            "history_message_count": self.history_message_count,
            "history_fingerprint": self.history_fingerprint,
            "current_user_fingerprint": self.current_user_fingerprint,
            "confidence": self.confidence,
            "selected_session_updated_at": self.selected_session_updated_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationResolution":
        return cls(
            session_id=str(value["session_id"]),
            student_id=str(value["student_id"]),
            resolution_method=str(value["resolution_method"]),
            created_new_session=bool(value.get("created_new_session", False)),
            matched_turn_count=int(value.get("matched_turn_count", 0)),
            candidate_count=int(value.get("candidate_count", 0)),
            history_message_count=int(value.get("history_message_count", 0)),
            history_fingerprint=str(value.get("history_fingerprint", "")),
            current_user_fingerprint=str(value.get("current_user_fingerprint", "")),
            confidence=float(value.get("confidence", 0.0)),
            selected_session_updated_at=(
                str(value["selected_session_updated_at"])
                if value.get("selected_session_updated_at") is not None
                else None
            ),
            warnings=tuple(str(item) for item in value.get("warnings", [])),
        )


class ConversationResolutionError(ValueError):
    """Raised when a request cannot be interpreted as a user conversation."""


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def _content_to_text(content: Any) -> str:
    """Extract text from common OpenAI string or multipart content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", ""))
            if item_type in {"text", "input_text", "output_text"}:
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def parse_openai_conversation(messages: Sequence[Any]) -> ParsedConversation:
    """Extract the user/assistant transcript and the current user message."""
    transcript: list[ConversationMessage] = []
    ignored_roles: list[str] = []

    for item in messages:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role", "")).strip().lower()
        if raw_role not in {"user", "assistant"}:
            if raw_role:
                ignored_roles.append(raw_role)
            continue
        text = _content_to_text(item.get("content"))
        if not _normalize_text(text):
            continue
        transcript.append(
            ConversationMessage(
                role="student" if raw_role == "user" else "tutor",
                text=text,
            )
        )

    if not transcript:
        raise ConversationResolutionError(
            "No user/assistant conversation messages were found."
        )
    if transcript[-1].role != "student":
        raise ConversationResolutionError(
            "The last conversation message must be a user message."
        )

    return ParsedConversation(
        history=tuple(transcript[:-1]),
        current_user_text=transcript[-1].text,
        ignored_roles=tuple(ignored_roles),
    )


def _normalized_signature(messages: Iterable[ConversationMessage]) -> tuple[tuple[str, str], ...]:
    return tuple((message.role, _normalize_text(message.text)) for message in messages)


def conversation_fingerprint(messages: Iterable[ConversationMessage]) -> str:
    payload = [
        {"role": role, "text": text}
        for role, text in _normalized_signature(messages)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _stored_turns_to_messages(turns: Iterable[StoredTurn]) -> tuple[ConversationMessage, ...]:
    output: list[ConversationMessage] = []
    for turn in turns:
        if turn.role == "student":
            output.append(ConversationMessage(role="student", text=turn.text))
        elif turn.role == "tutor":
            output.append(ConversationMessage(role="tutor", text=turn.text))
    return tuple(output)


def _default_session_id() -> str:
    return f"anythingllm-{uuid4().hex}"


def _title_from_question(text: str, *, max_length: int = 100) -> str:
    compact = _normalize_text(text)
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"


class AnythingLLMConversationResolver:
    """Resolve and persist AnythingLLM conversations in the tutor database."""

    def __init__(
        self,
        store: SQLiteTutorStore,
        *,
        student_id: str = DEFAULT_STUDENT_ID,
        max_candidate_sessions: int = DEFAULT_MAX_CANDIDATE_SESSIONS,
        max_match_messages: int = DEFAULT_MAX_MATCH_MESSAGES,
        session_id_factory: Callable[[], str] = _default_session_id,
    ) -> None:
        if not student_id.strip():
            raise ValueError("student_id must not be empty")
        self.store = store
        self.student_id = student_id
        self.max_candidate_sessions = max(1, int(max_candidate_sessions))
        self.max_match_messages = max(2, int(max_match_messages))
        self.session_id_factory = session_id_factory

    def resolve(self, messages: Sequence[Any]) -> tuple[ConversationResolution, ParsedConversation]:
        parsed = parse_openai_conversation(messages)
        history_for_match = parsed.history[-self.max_match_messages :]
        history_signature = _normalized_signature(history_for_match)
        history_fingerprint = conversation_fingerprint(history_for_match)
        current_fingerprint = text_fingerprint(parsed.current_user_text)

        if not history_for_match:
            resolution = self._create_new_resolution(
                parsed=parsed,
                method="new_conversation_no_history",
                history_fingerprint=history_fingerprint,
                current_fingerprint=current_fingerprint,
                confidence=1.0,
            )
            return resolution, parsed

        sessions = self.store.get_recent_sessions(
            student_id=self.student_id,
            limit=self.max_candidate_sessions,
        )
        matches: list[StoredSession] = []
        for session in sessions:
            stored_turns = self.store.get_recent_turns(
                session_id=session.id,
                limit=len(history_for_match),
                roles=["student", "tutor"],
            )
            stored_messages = _stored_turns_to_messages(stored_turns)
            if len(stored_messages) != len(history_for_match):
                continue
            if _normalized_signature(stored_messages) == history_signature:
                matches.append(session)

        if matches:
            # get_recent_sessions already orders by updated_at DESC. Preserve that
            # deterministic order and use recency only when exact history is
            # genuinely ambiguous.
            selected = matches[0]
            tiebreak = len(matches) > 1
            warnings: tuple[str, ...] = ()
            if tiebreak:
                warnings = (
                    "Multiple sessions had the same exact history suffix; the most recently active session was selected.",
                )
            resolution = ConversationResolution(
                session_id=selected.id,
                student_id=self.student_id,
                resolution_method=(
                    "exact_history_suffix_recent_tiebreak"
                    if tiebreak
                    else "exact_history_suffix"
                ),
                created_new_session=False,
                matched_turn_count=len(history_for_match),
                candidate_count=len(matches),
                history_message_count=len(parsed.history),
                history_fingerprint=history_fingerprint,
                current_user_fingerprint=current_fingerprint,
                confidence=0.75 if tiebreak else 1.0,
                selected_session_updated_at=selected.updated_at,
                warnings=warnings,
            )
            return resolution, parsed

        resolution = self._create_new_resolution(
            parsed=parsed,
            method="new_conversation_no_history_match",
            history_fingerprint=history_fingerprint,
            current_fingerprint=current_fingerprint,
            confidence=0.7,
            warnings=(
                "AnythingLLM supplied prior conversation messages, but no stored session had an exact matching suffix.",
            ),
        )
        return resolution, parsed

    def _create_new_resolution(
        self,
        *,
        parsed: ParsedConversation,
        method: str,
        history_fingerprint: str,
        current_fingerprint: str,
        confidence: float,
        warnings: tuple[str, ...] = (),
    ) -> ConversationResolution:
        session_id = self.session_id_factory()
        self.store.ensure_session(
            session_id=session_id,
            student_id=self.student_id,
            title=_title_from_question(parsed.current_user_text),
        )
        return ConversationResolution(
            session_id=session_id,
            student_id=self.student_id,
            resolution_method=method,
            created_new_session=True,
            matched_turn_count=0,
            candidate_count=0,
            history_message_count=len(parsed.history),
            history_fingerprint=history_fingerprint,
            current_user_fingerprint=current_fingerprint,
            confidence=confidence,
            warnings=warnings,
        )

    def append_exchange(
        self,
        *,
        resolution: ConversationResolution,
        parsed: ParsedConversation,
        response_text: str,
        request_id: str,
        request_fingerprint: str,
        integration_stage: str = "I0.2A",
        extra_metadata: dict[str, Any] | None = None,
        response_json: dict[str, Any] | None = None,
        tutor_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store imported history plus the current user/probe exchange.

        When AnythingLLM supplies history that does not match any known
        session, the resolver creates a session and imports that history once.
        Without this bootstrap, every later request from that chat would remain
        unmatched forever.
        """
        common_metadata = {
            "source": "anythingllm_desktop",
            "integration_stage": integration_stage,
            "request_id": request_id,
            "request_fingerprint": request_fingerprint,
            "conversation_resolution": resolution.as_dict(),
            **(extra_metadata or {}),
        }
        imported_turn_ids: list[int] = []
        if resolution.created_new_session and parsed.history:
            for history_message in parsed.history:
                imported_turn_ids.append(
                    self.store.save_turn(
                        student_id=resolution.student_id,
                        session_id=resolution.session_id,
                        role=history_message.role,
                        text=history_message.text,
                        metadata={
                            **common_metadata,
                            "message_role": (
                                "user" if history_message.role == "student" else "assistant"
                            ),
                            "imported_from_request_history": True,
                        },
                    )
                )

        student_turn_id = self.store.save_turn(
            student_id=resolution.student_id,
            session_id=resolution.session_id,
            role="student",
            text=parsed.current_user_text,
            metadata={**common_metadata, "message_role": "user"},
        )
        tutor_turn_id = self.store.save_turn(
            student_id=resolution.student_id,
            session_id=resolution.session_id,
            role="tutor",
            text=response_text,
            response_json=response_json,
            metadata={
                **common_metadata,
                **(tutor_metadata or {}),
                "message_role": "assistant",
                "reply_to_turn_id": student_turn_id,
            },
        )
        return {
            "imported_history": imported_turn_ids,
            "student": student_turn_id,
            "tutor": tutor_turn_id,
        }
