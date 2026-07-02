"""Shared support helpers for the Prompt Garden Streamlit review app."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import io
import json

from .garden import PromptGarden


REVIEW_NOTES_SCHEMA_VERSION = "prompt-garden-review-notes-v1"
TEXT_COMPARE_FIELDS = (
    "short_answer",
    "explanation",
    "experiment.reason",
    "comparison_text",
    "raw_output",
)


def stable_json(value: Any) -> str:
    """Serialize JSON in a stable pretty format for app downloads."""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


def now_iso() -> str:
    """Return a local timestamp for review-note and export metadata."""

    return datetime.now().isoformat(timespec="seconds")


def default_garden_root(repo_root: str | Path) -> Path:
    """Return the default Prompt Garden workspace root for the repo."""

    return Path(repo_root) / "prompt_garden"


def review_notes_path(
    garden_root: str | Path,
    scope: str,
) -> Path:
    """Return the persisted review-notes path for one scope."""

    garden = PromptGarden(garden_root)
    return garden.ensure_report_scope_dir(scope) / "review_notes.json"


def short_signature(signature: str | None) -> str:
    """Return a compact label for execution signatures."""

    if not signature:
        return "no_sig"
    return signature[:10]


def fewshot_label(value: Any) -> str:
    """Return a display label for the few-shot identifier."""

    if value in (None, "", "none"):
        return "no_fewshot"
    return str(value)


def note_key(row_id: str) -> str:
    """Return the stable storage key for one review-note row."""

    return row_id


def load_review_notes(
    garden_root: str | Path,
    scope: str,
) -> dict[str, Any]:
    """Load review notes for one scope, creating an empty shape in memory."""

    path = review_notes_path(garden_root, scope)
    if not path.exists():
        timestamp = now_iso()
        return {
            "schema_version": REVIEW_NOTES_SCHEMA_VERSION,
            "scope": scope,
            "created_at": timestamp,
            "updated_at": timestamp,
            "entries": {},
        }

    return json.loads(path.read_text(encoding="utf-8"))


def save_review_notes(
    garden_root: str | Path,
    scope: str,
    payload: dict[str, Any],
) -> Path:
    """Persist review notes for one scope."""

    path = review_notes_path(garden_root, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        stable_json(payload) + "\n",
        encoding="utf-8",
    )
    return path


def note_rows(notes_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return saved review-note entries sorted for inspection."""

    rows = list((notes_payload.get("entries") or {}).values())
    rows.sort(
        key=lambda row: (
            not bool(row.get("preferred")),
            row.get("case_id") or "",
            row.get("combo_id") or "",
            row.get("execution_signature") or "",
            row.get("updated_at") or "",
        )
    )
    return rows


def upsert_review_note(
    garden_root: str | Path,
    scope: str,
    row_id: str,
    case_id: str,
    combo_id: str,
    combo_title: str,
    execution_signature: str,
    model: str,
    fewshot_id: str | None,
    note_text: str,
    preferred: bool,
    reviewer: str,
) -> Path:
    """Insert or replace one review-note entry."""

    payload = load_review_notes(garden_root, scope)
    entries = payload.setdefault("entries", {})
    key = note_key(row_id)
    entries[key] = {
        "id": key,
        "row_id": row_id,
        "scope": scope,
        "case_id": case_id,
        "combo_id": combo_id,
        "combo_title": combo_title,
        "execution_signature": execution_signature,
        "model": model,
        "fewshot_id": fewshot_id,
        "note_text": note_text.strip(),
        "preferred": preferred,
        "reviewer": reviewer.strip(),
        "updated_at": now_iso(),
    }
    payload["updated_at"] = now_iso()
    return save_review_notes(garden_root, scope, payload)


def normalize_for_csv(value: Any) -> str:
    """Normalize exported values for CSV output."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Convert row dictionaries into CSV text."""

    if not rows:
        return ""

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: normalize_for_csv(row.get(key))
            for key in fieldnames
        })
    return buffer.getvalue()


def score_bounds(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Return numeric score bounds for slider initialization."""

    values = [
        float(row["score"])
        for row in rows
        if row.get("score") is not None
    ]
    if not values:
        return (0.0, 1.0)
    return (min(values), max(values))


def row_label(row: dict[str, Any]) -> str:
    """Return a readable label for one flattened review row."""

    return (
        f"{row.get('case_id', '-')}"
        f" | {row.get('combo_id', '-')}"
        f" | model={row.get('model', '-')}"
        f" | fewshot={fewshot_label(row.get('fewshot_id'))}"
        f" | score={row.get('score', '-')}"
        f" | sig={short_signature(row.get('execution_signature'))}"
        f" | {row.get('combo_title', '')}"
    )


def artifact_label(artifact: dict[str, Any]) -> str:
    """Return a readable label for one normalized artifact."""

    prompt_snapshot = artifact.get("prompt_snapshot") or {}
    base_combo = prompt_snapshot.get("base_combo") or {}
    combo_title = base_combo.get("title") or artifact.get("combo_id")
    score = (artifact.get("metrics") or {}).get("score")
    execution = artifact.get("execution") or {}
    return (
        f"{artifact.get('case_id', '-')}"
        f" | {artifact.get('combo_id', '-')}"
        f" | model={artifact.get('model', '-')}"
        f" | fewshot={fewshot_label(execution.get('fewshot_id'))}"
        f" | score={score}"
        f" | sig={short_signature(execution.get('signature'))}"
        f" | {combo_title}"
    )


def signature_label(signature: str) -> str:
    """Return a display label for execution signatures in selectors."""

    if not signature:
        return "(missing signature)"
    return signature


__all__ = [
    "REVIEW_NOTES_SCHEMA_VERSION",
    "TEXT_COMPARE_FIELDS",
    "artifact_label",
    "default_garden_root",
    "fewshot_label",
    "load_review_notes",
    "note_key",
    "note_rows",
    "now_iso",
    "review_notes_path",
    "row_label",
    "rows_to_csv",
    "save_review_notes",
    "score_bounds",
    "short_signature",
    "signature_label",
    "stable_json",
    "upsert_review_note",
]
