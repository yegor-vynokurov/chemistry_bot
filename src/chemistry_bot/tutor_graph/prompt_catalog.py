"""Load immutable, hash-verified tutor prompt revisions for runtime use."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompt_garden"
    / "curated"
    / "tutor_runtime"
    / "manifest.json"
)


@dataclass(frozen=True, slots=True)
class PromptRevision:
    """One immutable published prompt component and its verified text."""

    prompt_id: str
    version: str
    role: str
    sha256: str
    text: str

    def as_reference(self) -> dict[str, str]:
        """Return the content-free identity stored with a model invocation."""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "sha256": self.sha256,
            "role": self.role,
        }


class PromptCatalogError(ValueError):
    """Raised when the runtime prompt catalog is invalid or has drifted."""


class PromptCatalog:
    """Resolve role prompts from the packaged immutable runtime export."""

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> None:
        """Load the manifest and reject missing, escaping, or modified assets."""
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PromptCatalogError("The tutor prompt manifest cannot be loaded.") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "tutor-prompt-runtime-v1":
            raise PromptCatalogError("Unsupported tutor prompt manifest schema.")

        self.assignment = self._validate_assignment(manifest.get("assignment"))
        self._prompts: dict[str, PromptRevision] = {}
        raw_prompts = manifest.get("prompts")
        if not isinstance(raw_prompts, dict) or not raw_prompts:
            raise PromptCatalogError("The tutor prompt manifest has no prompt assets.")
        for prompt_id, raw in raw_prompts.items():
            self._prompts[str(prompt_id)] = self._load_revision(str(prompt_id), raw)

        self._components: dict[str, tuple[str, ...]] = {}
        raw_components = manifest.get("components")
        if not isinstance(raw_components, dict):
            raise PromptCatalogError("The tutor prompt manifest has no role-component map.")
        for role, prompt_ids in raw_components.items():
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise PromptCatalogError(f"Prompt role {role!r} must reference at least one asset.")
            ids = tuple(str(prompt_id) for prompt_id in prompt_ids)
            if any(prompt_id not in self._prompts for prompt_id in ids):
                raise PromptCatalogError(f"Prompt role {role!r} references an unknown prompt ID.")
            self._components[str(role)] = ids

    @staticmethod
    def _validate_assignment(value: Any) -> dict[str, str | None]:
        """Validate the active prompt assignment without accepting ambiguous states."""
        if not isinstance(value, dict):
            raise PromptCatalogError("The tutor prompt assignment is missing.")
        kind = value.get("kind")
        experiment_id = value.get("experiment_id")
        if kind == "baseline" and experiment_id is None:
            return {"kind": "baseline", "experiment_id": None}
        if kind == "experiment" and isinstance(experiment_id, str) and experiment_id.strip():
            return {"kind": "experiment", "experiment_id": experiment_id.strip()}
        raise PromptCatalogError("The prompt assignment must be baseline or a named experiment.")

    def _load_revision(self, prompt_id: str, raw: Any) -> PromptRevision:
        """Load one catalog asset and verify its path, role, version, and digest."""
        if not isinstance(raw, dict):
            raise PromptCatalogError(f"Prompt entry {prompt_id!r} is malformed.")
        role = raw.get("role")
        version = raw.get("version")
        relative_path = raw.get("path")
        expected_digest = raw.get("sha256")
        if not all(isinstance(item, str) and item.strip() for item in (role, version, relative_path, expected_digest)):
            raise PromptCatalogError(f"Prompt entry {prompt_id!r} is missing required metadata.")
        asset_path = (self.root / relative_path).resolve()
        if self.root not in asset_path.parents:
            raise PromptCatalogError(f"Prompt entry {prompt_id!r} escapes the runtime asset directory.")
        try:
            text = asset_path.read_text(encoding="utf-8")
        except OSError as error:
            raise PromptCatalogError(f"Prompt asset {prompt_id!r} cannot be read.") from error
        if raw.get("strip_final_newline") is True:
            text = text.removesuffix("\n")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != expected_digest:
            raise PromptCatalogError(f"Prompt asset {prompt_id!r} failed SHA-256 verification.")
        return PromptRevision(
            prompt_id=prompt_id,
            version=version,
            role=role,
            sha256=digest,
            text=text,
        )

    def components(self, role: str, *, runtime_text: str | None = None) -> tuple[PromptRevision, ...]:
        """Return the published components for a role and reject unregistered text."""
        prompt_ids = self._components.get(role)
        if prompt_ids is None:
            raise PromptCatalogError(f"No immutable prompt assets are registered for role {role!r}.")
        revisions = tuple(self._prompts[prompt_id] for prompt_id in prompt_ids)
        if runtime_text is not None and not any(item.text == runtime_text for item in revisions):
            raise PromptCatalogError(f"Runtime prompt text for role {role!r} differs from its published revision.")
        return revisions

    def revision(self, prompt_id: str) -> PromptRevision:
        """Return a verified prompt by ID for recovery and operator readback."""
        try:
            return self._prompts[prompt_id]
        except KeyError as error:
            raise PromptCatalogError(f"Unknown tutor prompt ID {prompt_id!r}.") from error

    def resolve_reference(self, reference: dict[str, Any]) -> PromptRevision:
        """Resolve a stored ID/version/hash tuple and reject historical revision drift."""
        prompt_id = reference.get("prompt_id")
        version = reference.get("version")
        digest = reference.get("sha256")
        if not all(isinstance(value, str) and value.strip() for value in (prompt_id, version, digest)):
            raise PromptCatalogError("A stored prompt reference requires ID, version, and SHA-256.")
        revision = self.revision(prompt_id)
        if revision.version != version or revision.sha256 != digest:
            raise PromptCatalogError(
                f"Stored prompt reference {prompt_id!r}@{version} does not match the published asset."
            )
        return revision


__all__ = ["DEFAULT_MANIFEST_PATH", "PromptCatalog", "PromptCatalogError", "PromptRevision"]
