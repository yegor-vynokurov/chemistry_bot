"""Durable diagnostics for the AnythingLLM native RAG tool-call cycle.

I0.2B intentionally behaves like a model that asks AnythingLLM to execute the
built-in ``rag-memory`` function. AnythingLLM should then send a second
OpenAI-compatible request containing the assistant tool call and the tool
result. This module groups both HTTP requests into one auditable interaction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .request_log import redact_sensitive_json, sha256_json, utc_now_iso


DEFAULT_RAG_TOOL_NAME = "rag-memory"
DEFAULT_TOOL_RESULT_PREVIEW_CHARS = 1200
_TOOL_CALL_PREFIX = "call_i02b_"
_INTERACTION_PREFIX = "i02b-"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _content_to_text(content: Any) -> str:
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
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
        return "\n".join(parts)
    if content is None:
        return ""
    return _canonical_json(content)


def _safe_preview(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def make_interaction_id() -> str:
    return f"{_INTERACTION_PREFIX}{uuid4().hex}"


def make_tool_call_id(interaction_id: str) -> str:
    suffix = interaction_id.removeprefix(_INTERACTION_PREFIX)
    if not re.fullmatch(r"[a-f0-9]{32}", suffix):
        raise ValueError("interaction_id is not an I0.2B interaction identifier")
    return f"{_TOOL_CALL_PREFIX}{suffix}"


def make_retry_tool_call_id(interaction_id: str, retry_number: int) -> str:
    """Return a retry tool-call id that still maps to the same interaction."""
    suffix = interaction_id.removeprefix(_INTERACTION_PREFIX)
    if not re.fullmatch(r"[a-f0-9]{32}", suffix):
        raise ValueError("interaction_id is not an I0.2B interaction identifier")
    retry = max(1, int(retry_number))
    return f"{_TOOL_CALL_PREFIX}{suffix}_r{retry}"


def interaction_id_from_tool_call_id(tool_call_id: str | None) -> str | None:
    if not isinstance(tool_call_id, str):
        return None
    match = re.fullmatch(
        rf"{re.escape(_TOOL_CALL_PREFIX)}([a-f0-9]{{32}})(?:_r\d+)?",
        tool_call_id,
    )
    if match is None:
        return None
    return f"{_INTERACTION_PREFIX}{match.group(1)}"


def available_function_tools(body: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return ()
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return tuple(names)


@dataclass(frozen=True, slots=True)
class ToolCallObservation:
    message_index: int
    tool_call_id: str | None
    tool_name: str | None
    arguments: str | None

    def as_dict(self) -> dict[str, Any]:
        parsed_arguments: Any = None
        if self.arguments:
            try:
                parsed_arguments = json.loads(self.arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
        return {
            "message_index": self.message_index,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "arguments_json": parsed_arguments,
        }


@dataclass(frozen=True, slots=True)
class ToolResultObservation:
    message_index: int
    role: str
    tool_call_id: str | None
    tool_name: str | None
    content: Any
    detection_method: str

    def as_dict(
        self,
        *,
        include_content: bool,
        preview_chars: int = DEFAULT_TOOL_RESULT_PREVIEW_CHARS,
    ) -> dict[str, Any]:
        text = _content_to_text(self.content)
        payload: dict[str, Any] = {
            "message_index": self.message_index,
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "detection_method": self.detection_method,
            "content_character_count": len(text),
            "content_sha256": sha256_json(self.content),
            "content_type": type(self.content).__name__,
        }
        if include_content:
            payload["content"] = redact_sensitive_json(deepcopy(self.content))
            payload["content_preview"] = _safe_preview(text, max_chars=preview_chars)
        return payload


@dataclass(frozen=True, slots=True)
class ToolCycleInspection:
    phase: str
    available_tools: tuple[str, ...]
    assistant_tool_calls: tuple[ToolCallObservation, ...]
    tool_results: tuple[ToolResultObservation, ...]
    interaction_id: str | None
    detected_tool_call_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def has_rag_tool(self) -> bool:
        return DEFAULT_RAG_TOOL_NAME in self.available_tools

    @property
    def has_tool_result(self) -> bool:
        return bool(self.tool_results)

    def as_dict(self, *, include_content: bool) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "available_tools": list(self.available_tools),
            "assistant_tool_calls": [item.as_dict() for item in self.assistant_tool_calls],
            "tool_results": [
                item.as_dict(include_content=include_content)
                for item in self.tool_results
            ],
            "interaction_id": self.interaction_id,
            "detected_tool_call_ids": list(self.detected_tool_call_ids),
            "warnings": list(self.warnings),
        }


def inspect_tool_cycle(
    body: Mapping[str, Any],
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> ToolCycleInspection:
    """Inspect an OpenAI request for tool calls and returned tool output.

    The expected second request uses standard OpenAI messages:

    * assistant message containing ``tool_calls``;
    * tool message containing ``tool_call_id`` and retrieved content.

    The probe also records looser variants because the exact AnythingLLM
    Desktop payload is what this experiment is meant to discover.
    """
    messages = body.get("messages")
    message_list = messages if isinstance(messages, list) else []
    calls: list[ToolCallObservation] = []
    results: list[ToolResultObservation] = []
    call_positions: dict[str, int] = {}

    for index, message in enumerate(message_list):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        raw_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(raw_calls, list):
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                function_dict = function if isinstance(function, dict) else {}
                call_id = raw_call.get("id")
                call_id_text = str(call_id) if call_id is not None else None
                tool_name = function_dict.get("name")
                tool_name_text = str(tool_name) if tool_name is not None else None
                arguments = function_dict.get("arguments")
                arguments_text = (
                    arguments
                    if isinstance(arguments, str)
                    else _canonical_json(arguments)
                    if arguments is not None
                    else None
                )
                calls.append(
                    ToolCallObservation(
                        message_index=index,
                        tool_call_id=call_id_text,
                        tool_name=tool_name_text,
                        arguments=arguments_text,
                    )
                )
                if call_id_text:
                    call_positions[call_id_text] = index

        explicit_call_id = message.get("tool_call_id")
        explicit_name = message.get("name")
        explicit_call_id_text = (
            str(explicit_call_id) if explicit_call_id is not None else None
        )
        explicit_name_text = str(explicit_name) if explicit_name is not None else None
        if role in {"tool", "function"}:
            results.append(
                ToolResultObservation(
                    message_index=index,
                    role=role,
                    tool_call_id=explicit_call_id_text,
                    tool_name=explicit_name_text,
                    content=message.get("content"),
                    detection_method="standard_tool_message",
                )
            )
        elif explicit_call_id_text is not None:
            results.append(
                ToolResultObservation(
                    message_index=index,
                    role=role or "unknown",
                    tool_call_id=explicit_call_id_text,
                    tool_name=explicit_name_text,
                    content=message.get("content"),
                    detection_method="message_with_tool_call_id",
                )
            )

    # If AnythingLLM returns a non-standard message after the assistant tool
    # call, keep it as a *possible* result rather than silently discarding it.
    if calls and not results:
        last_call_index = max(item.message_index for item in calls)
        for index, message in enumerate(message_list):
            if index <= last_call_index or not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role == "assistant":
                continue
            content = message.get("content")
            if not _content_to_text(content).strip():
                continue
            results.append(
                ToolResultObservation(
                    message_index=index,
                    role=role or "unknown",
                    tool_call_id=None,
                    tool_name=None,
                    content=content,
                    detection_method="message_after_assistant_tool_call",
                )
            )

    detected_ids: list[str] = []
    for item in (*calls, *results):
        call_id = item.tool_call_id
        if call_id and call_id not in detected_ids:
            detected_ids.append(call_id)

    interaction_ids = {
        value
        for value in (interaction_id_from_tool_call_id(call_id) for call_id in detected_ids)
        if value is not None
    }
    warnings: list[str] = []
    interaction_id: str | None = None
    if len(interaction_ids) == 1:
        interaction_id = next(iter(interaction_ids))
    elif len(interaction_ids) > 1:
        warnings.append("Multiple I0.2B interaction identifiers were found in one request.")

    if results:
        phase = "tool_result_return"
    elif calls:
        phase = "tool_call_echo_without_result"
    else:
        phase = "initial_request"

    available = available_function_tools(body)
    if rag_tool_name not in available and phase == "initial_request":
        warnings.append(f"The expected tool '{rag_tool_name}' was not present in request.tools.")

    return ToolCycleInspection(
        phase=phase,
        available_tools=available,
        assistant_tool_calls=tuple(calls),
        tool_results=tuple(results),
        interaction_id=interaction_id,
        detected_tool_call_ids=tuple(detected_ids),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class InteractionLogSettings:
    log_dir: Path
    log_content: bool = True


class ToolInteractionLog:
    """One JSON document per two-phase tool interaction plus a JSONL index."""

    def __init__(self, settings: InteractionLogSettings):
        self.settings = settings
        self.root = settings.log_dir / "interactions"
        self.index_path = settings.log_dir / "interactions.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _date_path(timestamp: str) -> tuple[str, str]:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        date_part = parsed.strftime("%Y-%m-%d")
        time_part = parsed.strftime("%H-%M-%S-%f")[:-3] + "Z"
        return date_part, time_part

    def _path_for(self, interaction_id: str, created_at: str) -> Path:
        date_part, time_part = self._date_path(created_at)
        directory = self.root / date_part
        directory.mkdir(parents=True, exist_ok=True)
        suffix = interaction_id.removeprefix(_INTERACTION_PREFIX)[:12]
        return directory / f"{time_part}_{suffix}.json"

    def _find_path(self, interaction_id: str) -> Path | None:
        if not re.fullmatch(r"i02b-[a-f0-9]{32}", interaction_id):
            return None
        for path in self.root.glob("*/*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("interaction_id") == interaction_id:
                return path
        return None

    def start(
        self,
        *,
        interaction_id: str,
        tool_call_id: str,
        created_at: str,
        session_id: str | None,
        student_id: str | None,
        initial_request_id: str,
        initial_request_fingerprint: str,
        query: str,
        resolution: Mapping[str, Any] | None,
        parsed_conversation: Mapping[str, Any] | None,
        routing_preview: Mapping[str, Any] | None = None,
        original_query: str | None = None,
        query_expansion: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": 1,
            "interaction_id": interaction_id,
            "status": "awaiting_tool_result",
            "tool_name": DEFAULT_RAG_TOOL_NAME,
            "tool_call_id": tool_call_id,
            "query": query if self.settings.log_content else None,
            "query_sha256": sha256_json(query),
            "original_query": (original_query or query) if self.settings.log_content else None,
            "original_query_sha256": sha256_json(original_query or query),
            "rag_query_expansion": redact_sensitive_json(deepcopy(dict(query_expansion)))
            if query_expansion is not None
            else None,
            "retry_count": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "session_id": session_id,
            "student_id": student_id,
            "initial_request_id": initial_request_id,
            "initial_request_fingerprint": initial_request_fingerprint,
            "result_request_id": None,
            "resolution": redact_sensitive_json(deepcopy(dict(resolution)))
            if resolution is not None
            else None,
            "parsed_conversation": redact_sensitive_json(
                deepcopy(dict(parsed_conversation))
            )
            if parsed_conversation is not None and self.settings.log_content
            else None,
            "routing_preview": redact_sensitive_json(
                deepcopy(dict(routing_preview))
            )
            if routing_preview is not None
            else None,
            "events": [
                {
                    "event": "tool_call_emitted",
                    "request_id": initial_request_id,
                    "at": created_at,
                }
            ],
        }
        self._write(row, append_index=True)
        return row


    def record_retry(
        self,
        *,
        interaction_id: str,
        request_id: str,
        retry_tool_call_id: str,
        retry_query: str,
        query_expansion: Mapping[str, Any] | None,
        rag_relevance_gate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record that the bridge emitted a second, targeted RAG search."""
        row = self.read(interaction_id)
        if row is None:
            return None
        now = utc_now_iso()
        retry_count = int(row.get("retry_count") or 0) + 1
        row["status"] = "awaiting_tool_retry_result"
        row["updated_at"] = now
        row["retry_count"] = retry_count
        row["retry_tool_call_id"] = retry_tool_call_id
        row["retry_query"] = retry_query if self.settings.log_content else None
        row["retry_query_sha256"] = sha256_json(retry_query)
        row["retry_query_expansion"] = redact_sensitive_json(deepcopy(dict(query_expansion))) if query_expansion is not None else None
        row["retry_after_rag_relevance_gate"] = redact_sensitive_json(deepcopy(dict(rag_relevance_gate))) if rag_relevance_gate is not None else None
        row.setdefault("query_expansion_attempts", [])
        if query_expansion is not None:
            row["query_expansion_attempts"].append(
                redact_sensitive_json(deepcopy(dict(query_expansion)))
            )
        row.setdefault("events", []).append(
            {
                "event": "tool_retry_emitted",
                "request_id": request_id,
                "at": now,
                "retry_count": retry_count,
                "retry_tool_call_id": retry_tool_call_id,
            }
        )
        self._write(row, append_index=True)
        return row

    def complete(
        self,
        *,
        interaction_id: str,
        result_request_id: str,
        completed_at: str,
        inspection: ToolCycleInspection,
        final_response_text: str,
        persisted_turn_ids: Mapping[str, Any] | None,
        completion_error: str | None = None,
        adapted_context: Mapping[str, Any] | None = None,
        rag_relevance_gate: Mapping[str, Any] | None = None,
        external_context_used_as_evidence: bool | None = None,
        graph_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        row = self.read(interaction_id)
        if row is None:
            return None
        row["status"] = "completed" if completion_error is None else "completed_with_error"
        row["updated_at"] = completed_at
        row["completed_at"] = completed_at
        row["result_request_id"] = result_request_id
        row["tool_result"] = inspection.as_dict(
            include_content=self.settings.log_content
        )
        row["final_response_text"] = (
            final_response_text if self.settings.log_content else None
        )
        row["final_response_sha256"] = sha256_json(final_response_text)
        row["persisted_turn_ids"] = dict(persisted_turn_ids or {})
        row["completion_error"] = completion_error
        row["adapted_context"] = redact_sensitive_json(
            deepcopy(dict(adapted_context))
        ) if adapted_context is not None else None
        row["rag_relevance_gate"] = redact_sensitive_json(
            deepcopy(dict(rag_relevance_gate))
        ) if rag_relevance_gate is not None else None
        row["external_context_received"] = adapted_context is not None
        row["external_context_used_as_evidence"] = external_context_used_as_evidence
        row["graph_result"] = redact_sensitive_json(
            deepcopy(dict(graph_result))
        ) if graph_result is not None else None
        row.setdefault("events", []).append(
            {
                "event": "tool_result_received",
                "request_id": result_request_id,
                "at": completed_at,
            }
        )
        self._write(row, append_index=True)
        return row

    def fail(
        self,
        *,
        interaction_id: str,
        request_id: str,
        error: str,
    ) -> dict[str, Any] | None:
        row = self.read(interaction_id)
        if row is None:
            return None
        now = utc_now_iso()
        row["status"] = "failed"
        row["updated_at"] = now
        row["error"] = error
        row.setdefault("events", []).append(
            {"event": "failed", "request_id": request_id, "at": now}
        )
        self._write(row, append_index=True)
        return row

    def _write(self, row: dict[str, Any], *, append_index: bool) -> Path:
        interaction_id = str(row["interaction_id"])
        path = self._find_path(interaction_id)
        if path is None:
            path = self._path_for(interaction_id, str(row.get("created_at") or utc_now_iso()))
        temp = path.with_suffix(".json.tmp")
        pretty = json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        compact = _canonical_json(row)
        with self._lock:
            temp.write_text(pretty + "\n", encoding="utf-8")
            temp.replace(path)
            if append_index:
                with self.index_path.open("a", encoding="utf-8") as handle:
                    handle.write(compact + "\n")
        return path

    def read(self, interaction_id: str) -> dict[str, Any] | None:
        path = self._find_path(interaction_id)
        if path is None:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def recent(self, *, limit: int = 20, include_content: bool = False) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        latest_by_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            interaction_id = row.get("interaction_id")
            if not isinstance(interaction_id, str):
                continue
            if interaction_id not in latest_by_id:
                order.append(interaction_id)
            latest_by_id[interaction_id] = row
        selected = order[-limit:]
        rows: list[dict[str, Any]] = []
        for interaction_id in selected:
            row = deepcopy(latest_by_id[interaction_id])
            if not include_content:
                row.pop("parsed_conversation", None)
                row.pop("final_response_text", None)
                row.pop("query", None)
                adapted_context = row.get("adapted_context")
                if isinstance(adapted_context, dict):
                    for chunk in adapted_context.get("chunks", []):
                        if isinstance(chunk, dict):
                            chunk.pop("text", None)
                graph_result = row.get("graph_result")
                if isinstance(graph_result, dict):
                    graph_result.pop("external_context", None)
                tool_result = row.get("tool_result")
                if isinstance(tool_result, dict):
                    for result in tool_result.get("tool_results", []):
                        if isinstance(result, dict):
                            result.pop("content", None)
                            result.pop("content_preview", None)
            rows.append(row)
        return rows


def parsed_conversation_to_dict(parsed: Any) -> dict[str, Any]:
    history = getattr(parsed, "history", ())
    return {
        "history": [item.as_dict() for item in history],
        "current_user_text": getattr(parsed, "current_user_text", ""),
        "ignored_roles": list(getattr(parsed, "ignored_roles", ())),
    }
