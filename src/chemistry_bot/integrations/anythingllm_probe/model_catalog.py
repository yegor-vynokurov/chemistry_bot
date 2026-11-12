"""Expose the installed local tutor models through the AnythingLLM bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_CONFIGURATION_PATH = Path(__file__).with_name("tutor_models.json")


class OllamaCatalogUnavailable(RuntimeError):
    """Raised when the active Ollama API cannot provide its installed model tags."""


class TutorModelUnavailable(ValueError):
    """Raised when a requested tutor tag is unsupported or not installed."""


def load_tutor_model_configuration(
    path: Path = MODEL_CONFIGURATION_PATH,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the allowlisted chat models and fixed support-model tags.

    The JSON file is shared with the host launcher so both selection surfaces
    use the same initial roster. Invalid or incomplete configuration raises
    ``ValueError`` before the bridge advertises a misleading model list.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read tutor model configuration at {path}.") from error
    tutor_models = payload.get("tutor_models")
    support_roles = payload.get("support_models")
    if not isinstance(tutor_models, list) or not tutor_models:
        raise ValueError("Tutor model configuration must include a non-empty tutor_models list.")
    if not isinstance(support_roles, dict) or not support_roles:
        raise ValueError("Tutor model configuration must include support_models by role.")
    if any(not isinstance(model, str) or not model.strip() for model in tutor_models):
        raise ValueError("Every tutor model must be a non-empty exact Ollama tag.")
    if any(not isinstance(model, str) or not model.strip() for model in support_roles.values()):
        raise ValueError("Every support role must name a non-empty exact Ollama tag.")
    return (
        tuple(dict.fromkeys(model.strip() for model in tutor_models)),
        tuple(dict.fromkeys(model.strip() for model in support_roles.values())),
    )


class OllamaModelCatalog:
    """Resolve AnythingLLM model choices against one active Ollama backend."""

    def __init__(
        self,
        *,
        base_url: str,
        allowed_models: tuple[str, ...],
        timeout_seconds: float = 4.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.allowed_models = tuple(dict.fromkeys(allowed_models))
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    def installed_tutor_models(self) -> tuple[str, ...]:
        """Return allowlisted chat tags installed on the active Ollama server.

        The call reads ``/api/tags`` and never downloads or changes a model.
        Network, HTTP, and malformed-response failures raise
        ``OllamaCatalogUnavailable`` so callers can fail clearly instead of
        showing stale choices.
        """
        request = Request(f"{self.base_url}/api/tags", headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload: Any = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaCatalogUnavailable(
                f"Could not read installed models from the active Ollama server at {self.base_url}."
            ) from error

        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise OllamaCatalogUnavailable("Ollama /api/tags returned no models list.")
        installed: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("model") or entry.get("name")
            if isinstance(name, str) and name.strip():
                installed.add(name.strip())
        return tuple(model for model in self.allowed_models if model in installed)

    def resolve_requested_model(
        self,
        requested_model: str | None,
        *,
        default_model: str,
        legacy_model_id: str | None = None,
    ) -> str:
        """Validate a request model and return its exact installed tutor tag.

        The former fixed bridge ID remains a compatibility alias for the
        startup-selected model. Unknown and uninstalled selections raise
        ``TutorModelUnavailable``; no fallback or model pull is performed.
        """
        candidate = (requested_model or "").strip()
        if not candidate or (legacy_model_id and candidate == legacy_model_id):
            candidate = default_model
        if candidate not in self.allowed_models:
            raise TutorModelUnavailable(
                f"Tutor model {candidate!r} is not in the configured tutor model list."
            )
        available = self.installed_tutor_models()
        if candidate not in available:
            raise TutorModelUnavailable(
                f"Tutor model {candidate!r} is not installed in the active Ollama backend. "
                "Run the startup command with this model to add it."
            )
        return candidate

    def openai_model_list(self) -> dict[str, Any]:
        """Build the OpenAI-compatible model-list response for AnythingLLM."""
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "chemistry-tutor-local",
                }
                for model in self.installed_tutor_models()
            ],
        }


__all__ = [
    "MODEL_CONFIGURATION_PATH",
    "OllamaCatalogUnavailable",
    "OllamaModelCatalog",
    "TutorModelUnavailable",
    "load_tutor_model_configuration",
]
