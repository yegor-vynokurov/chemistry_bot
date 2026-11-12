"""Privacy-aware request logging for the AnythingLLM integration probe."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Mapping


_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:authorization|cookie|api[-_]?key|access[-_]?token|refresh[-_]?token|secret|password)",
    re.IGNORECASE,
)

_IDENTIFIER_KEY_ALIASES: dict[str, str] = {
    "threadid": "thread_id",
    "thread_id": "thread_id",
    "sessionid": "session_id",
    "session_id": "session_id",
    "conversationid": "conversation_id",
    "conversation_id": "conversation_id",
    "workspaceid": "workspace_id",
    "workspace_id": "workspace_id",
    "chatid": "chat_id",
    "chat_id": "chat_id",
    "userid": "user_id",
    "user_id": "user_id",
    "requestid": "external_request_id",
    "request_id": "external_request_id",
}


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for logs and API responses."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return headers with credentials and cookies replaced by a marker."""
    output: dict[str, str] = {}
    for key, value in headers.items():
        output[key.lower()] = "[REDACTED]" if _SENSITIVE_KEY_PATTERN.search(key) else value
    return output


def redact_sensitive_json(value: Any) -> Any:
    """Recursively redact values stored under credential-like JSON keys."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _SENSITIVE_KEY_PATTERN.search(str(key))
                else redact_sensitive_json(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_json(item) for item in value]
    return value


def _content_stats(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {
            "content_type": "string",
            "character_count": len(content),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    return {
        "content_type": type(content).__name__,
        "serialized_character_count": len(_canonical_json(content)),
        "sha256": sha256_json(content),
    }


def summarize_body_without_content(body: dict[str, Any]) -> dict[str, Any]:
    """Keep request structure while replacing message content with hashes/stats."""
    safe = redact_sensitive_json(deepcopy(body))
    messages = safe.get("messages")
    if isinstance(messages, list):
        summarized_messages: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                summarized_messages.append({"value_type": type(message).__name__})
                continue
            summarized = {
                key: value for key, value in message.items() if key != "content"
            }
            if "content" in message:
                summarized["content"] = _content_stats(message.get("content"))
            summarized_messages.append(summarized)
        safe["messages"] = summarized_messages
    return safe


def _flatten_candidate_identifiers(value: Any, *, path: str = "body") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9_]", "", str(key).lower())
            canonical = _IDENTIFIER_KEY_ALIASES.get(normalized)
            child_path = f"{path}.{key}"
            if canonical and isinstance(item, (str, int)) and str(item).strip():
                found.append(
                    {
                        "kind": canonical,
                        "path": child_path,
                        "value": str(item),
                    }
                )
            if isinstance(item, (dict, list)):
                found.extend(_flatten_candidate_identifiers(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                found.extend(_flatten_candidate_identifiers(item, path=f"{path}[{index}]"))
    return found


def _header_candidate_identifiers(headers: Mapping[str, str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for key, value in headers.items():
        normalized = re.sub(r"[^a-z0-9_]", "", key.lower().removeprefix("x-"))
        canonical = _IDENTIFIER_KEY_ALIASES.get(normalized)
        if canonical and value.strip():
            found.append(
                {
                    "kind": canonical,
                    "path": f"headers.{key.lower()}",
                    "value": value,
                }
            )
    return found


def build_request_diagnostics(
    body: dict[str, Any],
    *,
    headers: Mapping[str, str],
    include_content_preview: bool = True,
) -> dict[str, Any]:
    """Extract request-shape facts without pretending they identify a session."""
    messages = body.get("messages")
    message_list = messages if isinstance(messages, list) else []
    roles: list[str] = []
    role_counts: dict[str, int] = {}
    for message in message_list:
        role = str(message.get("role", "unknown")) if isinstance(message, dict) else "invalid"
        roles.append(role)
        role_counts[role] = role_counts.get(role, 0) + 1

    history_messages = list(message_list)
    if history_messages and isinstance(history_messages[-1], dict):
        if history_messages[-1].get("role") == "user":
            history_messages = history_messages[:-1]

    last_user_preview: str | None = None
    for message in reversed(message_list):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                last_user_preview = (
                    " ".join(content.split())[:240]
                    if include_content_preview
                    else "[CONTENT NOT LOGGED]"
                )
            else:
                last_user_preview = (
                    f"<{type(content).__name__} content>"
                    if include_content_preview
                    else "[CONTENT NOT LOGGED]"
                )
            break

    body_identifiers = _flatten_candidate_identifiers(body)
    header_identifiers = _header_candidate_identifiers(headers)

    tool_names: list[str] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_names.append(function["name"])

    assistant_tool_call_count = 0
    tool_result_message_count = 0
    tool_call_ids: list[str] = []
    for message in message_list:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            assistant_tool_call_count += len(message["tool_calls"])
            for call in message["tool_calls"]:
                if isinstance(call, dict) and call.get("id") is not None:
                    tool_call_ids.append(str(call["id"]))
        if message.get("role") in {"tool", "function"} or message.get("tool_call_id") is not None:
            tool_result_message_count += 1
            if message.get("tool_call_id") is not None:
                tool_call_ids.append(str(message["tool_call_id"]))

    return {
        "top_level_keys": sorted(body.keys()),
        "model": body.get("model"),
        "stream": bool(body.get("stream", False)),
        "message_count": len(message_list),
        "message_roles": roles,
        "message_role_counts": role_counts,
        "last_user_preview": last_user_preview,
        "messages_sha256": sha256_json(message_list),
        "history_sha256": sha256_json(history_messages),
        "last_four_messages_sha256": sha256_json(message_list[-4:]),
        "candidate_identifiers": body_identifiers + header_identifiers,
        "has_metadata": isinstance(body.get("metadata"), dict),
        "has_user_field": body.get("user") is not None,
        "available_tool_names": tool_names,
        "assistant_tool_call_count": assistant_tool_call_count,
        "tool_result_message_count": tool_result_message_count,
        "tool_call_ids": list(dict.fromkeys(tool_call_ids)),
    }


@dataclass(frozen=True)
class RequestLogSettings:
    log_dir: Path
    log_content: bool = True


class RequestProbeLog:
    """Append-only JSONL index plus one readable JSON file per request."""

    def __init__(self, settings: RequestLogSettings):
        self.settings = settings
        self.log_dir = settings.log_dir
        self.requests_dir = self.log_dir / "requests"
        self.index_path = self.log_dir / "requests.jsonl"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _timestamp_parts(timestamp: str) -> tuple[str, str]:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        date_part = parsed.strftime("%Y-%m-%d")
        time_part = parsed.strftime("%H-%M-%S-%f")[:-3] + "Z"
        return date_part, time_part

    def persist(self, record: dict[str, Any]) -> Path:
        request_id = str(record["request_id"])
        date_part, time_part = self._timestamp_parts(str(record.get("received_at") or utc_now_iso()))
        directory = self.requests_dir / date_part
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{time_part}_{request_id}.json"
        record["log_file"] = str(target.relative_to(self.log_dir)).replace("\\", "/")
        temp = target.with_suffix(".json.tmp")
        payload = _canonical_json(record)
        pretty_payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, default=str)

        with self._lock:
            temp.write_text(pretty_payload + "\n", encoding="utf-8")
            temp.replace(target)
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        return target

    def make_record(
        self,
        *,
        request_id: str,
        received_at: str,
        method: str,
        path: str,
        client: dict[str, Any],
        headers: Mapping[str, str],
        body: dict[str, Any],
        raw_body_bytes: int,
        response_mode: str,
        response_model: str,
    ) -> dict[str, Any]:
        logged_body = (
            redact_sensitive_json(deepcopy(body))
            if self.settings.log_content
            else summarize_body_without_content(body)
        )
        return {
            "schema_version": 1,
            "request_id": request_id,
            "received_at": received_at,
            "http": {
                "method": method,
                "path": path,
                "client": client,
                "headers": redact_headers(headers),
                "raw_body_bytes": raw_body_bytes,
            },
            "request_body": logged_body,
            "diagnostics": build_request_diagnostics(
                body,
                headers=headers,
                include_content_preview=self.settings.log_content,
            ),
            "probe_response": {
                "mode": response_mode,
                "model": response_model,
            },
            "privacy": {
                "message_content_logged": self.settings.log_content,
                "credential_headers_redacted": True,
                "credential_json_fields_redacted": True,
            },
        }

    def recent(self, *, limit: int = 20, include_body: bool = False) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not include_body:
                row.pop("request_body", None)
                http = row.get("http")
                if isinstance(http, dict):
                    http.pop("headers", None)
            rows.append(row)
        return rows

    def read(self, request_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", request_id):
            return None
        candidates = [self.requests_dir / f"{request_id}.json"]
        candidates.extend(self.requests_dir.glob(f"*/*_{request_id}.json"))
        for path in candidates:
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        return None
