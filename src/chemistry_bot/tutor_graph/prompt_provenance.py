"""Capture content-free provenance for each actual model invocation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable

from .prompt_catalog import PromptRevision


@dataclass(frozen=True, slots=True)
class PromptInvocation:
    """One attempted call with exact prompt components and assignment."""

    role: str
    provider: str
    model: str
    status: str
    attempt: int
    prompts: tuple[dict[str, str], ...]
    assignment_kind: str
    experiment_id: str | None = None
    route: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a persistence-safe record without message or prompt text."""
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "attempt": self.attempt,
            "prompts": [dict(prompt) for prompt in self.prompts],
            "assignment_kind": self.assignment_kind,
            "experiment_id": self.experiment_id,
            "route": self.route,
            "error_code": self.error_code,
        }


class PromptInvocationCollector:
    """Collect request-local model call records in execution order."""

    def __init__(self, initial: Iterable[dict[str, Any]] = ()) -> None:
        """Initialize an ordered collector with any already executed support calls."""
        self._records = [dict(record) for record in initial]
        self._lock = Lock()

    def invoke(
        self,
        model: Any,
        messages: list[Any],
        *,
        role: str,
        model_name: str,
        prompts: Iterable[PromptRevision],
        assignment: dict[str, str | None],
        attempt: int = 1,
        route: str | None = None,
    ) -> Any:
        """Invoke a model and record success or failure without retaining content."""
        revisions = tuple(prompts)
        if not revisions:
            raise ValueError("A model invocation must reference at least one published prompt.")
        error_code = None
        status = "succeeded"
        try:
            response = model.invoke(messages)
        except Exception as error:
            status = "failed"
            error_code = type(error).__name__
            self._append(
                PromptInvocation(
                    role=role,
                    provider="ollama",
                    model=model_name,
                    status=status,
                    attempt=attempt,
                    prompts=tuple(item.as_reference() for item in revisions),
                    assignment_kind=str(assignment["kind"]),
                    experiment_id=assignment.get("experiment_id"),
                    route=route,
                    error_code=error_code,
                )
            )
            raise
        self._append(
            PromptInvocation(
                role=role,
                provider="ollama",
                model=model_name,
                status=status,
                attempt=attempt,
                prompts=tuple(item.as_reference() for item in revisions),
                assignment_kind=str(assignment["kind"]),
                experiment_id=assignment.get("experiment_id"),
                route=route,
            )
        )
        return response

    def _append(self, invocation: PromptInvocation) -> None:
        """Append one invocation under a lock for parallel LangGraph nodes."""
        with self._lock:
            self._records.append(invocation.as_dict())

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return an ordered defensive copy of all invocation records."""
        with self._lock:
            return tuple(dict(record) for record in self._records)


def no_model_invocation(reason: str) -> dict[str, Any]:
    """Build the explicit content-free record for a turn with no model calls."""
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("A no-model turn requires a non-empty reason.")
    return {
        "role": "no_model_invocation",
        "provider": None,
        "model": None,
        "status": "no_model_invocation",
        "attempt": 0,
        "prompts": [],
        "assignment_kind": "not_applicable",
        "experiment_id": None,
        "route": None,
        "error_code": None,
        "reason": normalized_reason,
    }


__all__ = ["PromptInvocation", "PromptInvocationCollector", "no_model_invocation"]
