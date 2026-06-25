"""Review-oriented Prompt Garden artifact normalization and loading."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence
import json
import re

from .garden import PromptGarden


RAW_RUN_ARTIFACT_VERSION = "prompt-garden-raw-run-v1"
NORMALIZED_REVIEW_ARTIFACT_VERSION = "prompt-garden-normalized-review-v1"
SUMMARY_REPORT_ARTIFACT_VERSION = "prompt-garden-summary-report-v1"
TEXT_NORMALIZATION_VERSION = "prompt-review-normalization-v1"
PROMPT_SNAPSHOT_VERSION = "prompt-snapshot-v1"
SUMMARY_REPORT_KIND = "summary"

_FIELD_ORDER = (
    "short_answer",
    "explanation",
    "examples",
    "experiment_reason",
    "experiment_questions",
    "raw_output",
)
_INLINE_WS_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _safe_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def _now_iso() -> str:
    return PromptGarden._now()


def filesystem_timestamp_from_iso(iso_value: str) -> str:
    """Convert an ISO-like timestamp into a Windows-safe filename token."""

    return (
        iso_value.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("+", "_")
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with stable pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text_for_review(text: Any) -> str:
    """
    Produce a stable review-friendly text form.

    Rules:
    - convert missing values to an empty string
    - normalize line endings to LF
    - trim trailing whitespace on each line
    - collapse repeated spaces and tabs inside lines
    - collapse 3+ blank lines into a double line break
    - trim outer whitespace
    """

    if text is None:
        return ""

    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [
        _INLINE_WS_RE.sub(" ", line).rstrip()
        for line in value.split("\n")
    ]
    normalized = "\n".join(normalized_lines).strip()
    normalized = _MULTI_BLANK_RE.sub("\n\n", normalized)
    return normalized


def normalize_inline_text(text: Any) -> str:
    """Normalize text to a single whitespace-collapsed line."""

    return " ".join(normalize_text_for_review(text).split())


def split_paragraphs(text: Any) -> list[str]:
    """Split review text into stable paragraph blocks."""

    normalized = normalize_text_for_review(text)
    if not normalized:
        return []

    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SPLIT_RE.split(normalized)
        if paragraph.strip()
    ]


def text_block_payload(text: Any) -> dict[str, Any]:
    """Return a reusable normalized text block for review UIs."""

    original = "" if text is None else str(text)
    normalized = normalize_text_for_review(original)
    paragraphs = split_paragraphs(normalized)
    words = normalized.split()
    return {
        "text": original,
        "normalized": normalized,
        "inline_normalized": normalize_inline_text(normalized),
        "paragraphs": paragraphs,
        "char_count": len(normalized),
        "word_count": len(words),
        "line_count": len(normalized.splitlines()) if normalized else 0,
        "paragraph_count": len(paragraphs),
    }


def list_block_payload(values: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize a list of text values into stable review items."""

    return [
        {
            "index": index,
            **text_block_payload(value),
        }
        for index, value in enumerate(values, start=1)
    ]


def build_comparison_text(
    normalized_text_blocks: dict[str, Any],
) -> str:
    """Build a stable concatenated text view for diffs and embeddings."""

    sections: list[str] = []

    for field_name in _FIELD_ORDER:
        field_value = normalized_text_blocks.get(field_name)
        if isinstance(field_value, dict):
            normalized = field_value.get("normalized", "")
            if normalized:
                sections.append(
                    f"{field_name.upper()}\n{normalized}"
                )
        elif isinstance(field_value, list):
            normalized_items = [
                item.get("normalized", "")
                for item in field_value
                if isinstance(item, dict)
            ]
            normalized_items = [
                item for item in normalized_items
                if item
            ]
            if normalized_items:
                sections.append(
                    f"{field_name.upper()}\n"
                    + "\n".join(normalized_items)
                )

    return "\n\n".join(sections).strip()


def build_normalized_text_blocks(
    parsed_answer: dict[str, Any] | None,
    raw_output_text: str | None,
) -> dict[str, Any]:
    """Project a parsed answer into stable comparison blocks."""

    experiment_block = {}
    if parsed_answer:
        experiment = parsed_answer.get("experiment")
        if isinstance(experiment, dict):
            experiment_block = experiment

    blocks = {
        "normalization_version": TEXT_NORMALIZATION_VERSION,
        "short_answer": text_block_payload(
            (parsed_answer or {}).get("short_answer")
        ),
        "explanation": text_block_payload(
            (parsed_answer or {}).get("explanation")
        ),
        "examples": list_block_payload(
            (parsed_answer or {}).get("examples", [])
        ),
        "experiment_reason": text_block_payload(
            experiment_block.get("reason")
        ),
        "experiment_questions": list_block_payload(
            experiment_block.get("questions", [])
        ),
        "raw_output": text_block_payload(raw_output_text),
    }
    comparison_text = build_comparison_text(blocks)
    blocks["comparison_text"] = comparison_text
    blocks["comparison_hash"] = PromptGarden._hash_data(
        comparison_text
    )
    return blocks


def build_answer_lengths(
    parsed_answer: dict[str, Any] | None,
    normalized_text_blocks: dict[str, Any],
) -> dict[str, int]:
    """Build simple length metrics for review tables and filters."""

    examples = (parsed_answer or {}).get("examples", [])
    short_answer = normalized_text_blocks.get("short_answer", {})
    explanation = normalized_text_blocks.get("explanation", {})
    experiment_reason = normalized_text_blocks.get("experiment_reason", {})
    raw_output = normalized_text_blocks.get("raw_output", {})

    return {
        "short_answer_chars": int(short_answer.get("char_count", 0)),
        "short_answer_words": int(short_answer.get("word_count", 0)),
        "explanation_chars": int(explanation.get("char_count", 0)),
        "explanation_words": int(explanation.get("word_count", 0)),
        "example_count": len(examples),
        "experiment_reason_chars": int(
            experiment_reason.get("char_count", 0)
        ),
        "raw_output_chars": int(raw_output.get("char_count", 0)),
        "comparison_chars": len(
            normalized_text_blocks.get("comparison_text", "")
        ),
        "paragraph_count_total": (
            int(short_answer.get("paragraph_count", 0))
            + int(explanation.get("paragraph_count", 0))
            + int(experiment_reason.get("paragraph_count", 0))
        ),
    }


def _node_stats_snapshot(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    return {
        "version": stats.get("version"),
        "char_count": stats.get("char_count"),
        "word_count": stats.get("word_count"),
        "line_count": stats.get("line_count"),
        "sentence_count": stats.get("sentence_count"),
        "placeholder_count": stats.get("placeholder_count"),
        "placeholders": list(stats.get("placeholders", [])),
    }


def _node_snapshot(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    return {
        "id": node["id"],
        "type": node["type"],
        "tree_id": node["tree_id"],
        "parent_id": node.get("parent_id"),
        "branch": node["branch"],
        "title": node["title"],
        "path": node["path"],
        "tags": list(node.get("tags", [])),
        "created_at": node.get("created_at"),
        "metadata": {
            "kind": metadata.get("kind"),
            "content_hash": metadata.get("content_hash"),
            "context_hash": metadata.get("context_hash"),
            "prompt_role": metadata.get("prompt_role"),
        },
        "stats": _node_stats_snapshot(node.get("stats")),
    }


def _combo_metadata_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _safe_copy(metadata[key])
        for key in [
            "kind",
            "base_combo_id",
            "context_hash",
            "render_version",
            "generator_version",
            "roles_to_prompt_type",
            "student_context",
            "teacher_context",
            "combo_key",
        ]
        if key in metadata
    }


def _combo_snapshot(combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": combo["id"],
        "title": combo["title"],
        "status": combo.get("status"),
        "test_status": combo.get("test_status"),
        "score": combo.get("score"),
        "notes": combo.get("notes"),
        "tags": list(combo.get("tags", [])),
        "prompt_ids": dict(combo.get("prompt_ids", {})),
        "created_at": combo.get("created_at"),
        "updated_at": combo.get("updated_at"),
        "metadata": _combo_metadata_snapshot(
            combo.get("metadata") or {}
        ),
        "stats": _safe_copy(combo.get("stats") or {}),
    }


def build_prompt_snapshot(
    garden: PromptGarden,
    base_combo_id: str,
    active_combo_id: str,
    fewshot_id: str | None = None,
    selected_fewshot_example_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Capture combo and prompt-lineage state for one executed answer."""

    base_combo = garden.get_combo(base_combo_id)
    active_combo = garden.get_combo(active_combo_id)
    role_map: dict[str, Any] = {}

    for role, prompt_id in active_combo.get("prompt_ids", {}).items():
        node = garden.get_node(prompt_id)
        role_map[role] = {
            "prompt_id": prompt_id,
            "node": _node_snapshot(node),
            "lineage": [
                _node_snapshot(lineage_node)
                for lineage_node in garden.get_lineage(prompt_id)
            ],
        }

    fewshot_payload = None
    if fewshot_id:
        try:
            fewshot_node = garden.get_node(fewshot_id)
        except KeyError:
            fewshot_payload = {
                "prompt_id": fewshot_id,
                "missing": True,
                "selected_example_ids": list(
                    selected_fewshot_example_ids
                ),
            }
        else:
            fewshot_payload = {
                "prompt_id": fewshot_id,
                "missing": False,
                "node": _node_snapshot(fewshot_node),
                "lineage": [
                    _node_snapshot(lineage_node)
                    for lineage_node in garden.get_lineage(fewshot_id)
                ],
                "selected_example_ids": list(
                    selected_fewshot_example_ids
                ),
            }

    snapshot = {
        "version": PROMPT_SNAPSHOT_VERSION,
        "base_combo": _combo_snapshot(base_combo),
        "active_combo": _combo_snapshot(active_combo),
        "combo_relation": {
            "base_combo_id": base_combo_id,
            "active_combo_id": active_combo_id,
            "changed_roles": sorted(
                [
                    role
                    for role, prompt_id in active_combo.get(
                        "prompt_ids",
                        {},
                    ).items()
                    if base_combo.get("prompt_ids", {}).get(role)
                    != prompt_id
                ]
            ),
        },
        "roles": role_map,
        "fewshot": fewshot_payload,
    }
    snapshot["snapshot_hash"] = PromptGarden._hash_data(snapshot)
    return snapshot


def relative_artifact_paths(
    scope: str,
    filename: str,
) -> dict[str, str]:
    """Build stable repo-relative artifact references."""

    return {
        "raw": (
            Path("runs") / "raw" / scope / filename
        ).as_posix(),
        "normalized": (
            Path("runs") / "normalized" / scope / filename
        ).as_posix(),
    }


def load_normalized_scope(
    garden_root: str | Path,
    scope: str,
    execution_signature: str | None = None,
    combo_ids: Sequence[str] = (),
    case_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Load normalized artifacts for one Prompt Garden scope."""

    garden = PromptGarden(garden_root)
    scope_dir = garden.normalized_runs_dir / scope
    if not scope_dir.exists():
        return []

    selected_combo_ids = set(combo_ids)
    selected_case_ids = set(case_ids)
    artifacts: list[dict[str, Any]] = []

    for path in sorted(scope_dir.glob("*.json")):
        try:
            artifact = read_json(path)
        except json.JSONDecodeError:
            continue

        if execution_signature is not None:
            execution = artifact.get("execution") or {}
            if execution.get("signature") != execution_signature:
                continue

        combo_id = artifact.get("combo_id")
        case_id = artifact.get("case_id")

        if selected_combo_ids and combo_id not in selected_combo_ids:
            continue

        if selected_case_ids and case_id not in selected_case_ids:
            continue

        artifact["_path"] = str(path)
        artifact["_relative_path"] = str(
            path.relative_to(garden.root)
        ).replace("\\", "/")
        artifacts.append(artifact)

    return artifacts


def build_review_rows(
    garden: PromptGarden,
    artifacts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten normalized artifacts into review-table rows."""

    rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        prompt_snapshot = artifact.get("prompt_snapshot") or {}
        base_combo = prompt_snapshot.get("base_combo") or {}
        active_combo = prompt_snapshot.get("active_combo") or {}
        combo_title = (
            base_combo.get("title")
            or active_combo.get("title")
            or artifact.get("combo_id")
        )
        combo_metadata = (
            base_combo.get("metadata")
            or active_combo.get("metadata")
            or {}
        )
        role_map = prompt_snapshot.get("roles") or {}
        normalized_text_blocks = artifact.get(
            "normalized_text_blocks"
        ) or {}
        short_answer = (
            normalized_text_blocks.get("short_answer") or {}
        ).get("normalized", "")
        explanation = (
            normalized_text_blocks.get("explanation") or {}
        ).get("normalized", "")
        parsed_answer = artifact.get("parsed_answer") or {}
        metrics = artifact.get("metrics") or {}
        source_ids = list(artifact.get("source_ids", []))
        fewshot = prompt_snapshot.get("fewshot") or {}

        system_prompt_id = None
        if isinstance(role_map.get("system"), dict):
            system_prompt_id = role_map["system"].get("prompt_id")

        user_prompt_id = None
        if isinstance(role_map.get("user"), dict):
            user_prompt_id = role_map["user"].get("prompt_id")

        rows.append({
            "row_id": artifact.get("id"),
            "raw_run_id": artifact.get("raw_run_id"),
            "created_at": artifact.get("created_at"),
            "experiment_id": artifact.get("experiment_id"),
            "execution_signature": (
                artifact.get("execution") or {}
            ).get("signature"),
            "model": artifact.get("model"),
            "combo_id": artifact.get("combo_id"),
            "combo_title": combo_title,
            "combo_kind": combo_metadata.get("kind"),
            "base_combo_id": (
                prompt_snapshot.get("combo_relation") or {}
            ).get("base_combo_id"),
            "active_combo_id": artifact.get("active_combo_id"),
            "system_prompt_id": system_prompt_id,
            "user_prompt_id": user_prompt_id,
            "fewshot_id": fewshot.get("prompt_id"),
            "case_set_id": artifact.get("case_set_id"),
            "case_id": artifact.get("case_id"),
            "question": artifact.get("question"),
            "score": metrics.get("score"),
            "passed": bool(metrics.get("passed")),
            "passed_count": metrics.get("passed_count"),
            "total_count": metrics.get("total_count"),
            "duration_seconds": metrics.get("duration_seconds"),
            "validation_ok": bool(metrics.get("validation_ok")),
            "parse_error": bool(
                (artifact.get("review_flags") or {}).get(
                    "parse_error"
                )
            ),
            "request_type": artifact.get("request_type"),
            "experiment_kind": artifact.get("experiment_kind"),
            "certainty": parsed_answer.get("certainty"),
            "source_count": len(source_ids),
            "source_ids": source_ids,
            "example_count": int(
                (artifact.get("answer_lengths") or {}).get(
                    "example_count",
                    0,
                )
            ),
            "short_answer": short_answer,
            "explanation_preview": explanation[:240],
            "comparison_text": normalized_text_blocks.get(
                "comparison_text",
                "",
            ),
            "comparison_hash": normalized_text_blocks.get(
                "comparison_hash"
            ),
            "tags": list(artifact.get("tags", [])),
            "normalized_artifact_path": (
                artifact.get("artifact_paths") or {}
            ).get("normalized") or artifact.get("_relative_path"),
            "raw_artifact_path": (
                artifact.get("artifact_paths") or {}
            ).get("raw"),
        })

    return sorted(
        rows,
        key=lambda row: (
            row.get("combo_id") or "",
            row.get("case_id") or "",
            row.get("created_at") or "",
        ),
    )


def load_review_rows(
    garden_root: str | Path,
    scope: str,
    execution_signature: str | None = None,
    combo_ids: Sequence[str] = (),
    case_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Load and flatten one normalized scope into review rows."""

    garden = PromptGarden(garden_root)
    artifacts = load_normalized_scope(
        garden_root=garden_root,
        scope=scope,
        execution_signature=execution_signature,
        combo_ids=combo_ids,
        case_ids=case_ids,
    )
    return build_review_rows(garden, artifacts)


def combo_summary_rows(
    review_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate review rows into combo-level summaries."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        grouped[str(row.get("combo_id"))].append(row)

    summary_rows: list[dict[str, Any]] = []

    for combo_id, rows in grouped.items():
        score_values = [
            float(row["score"])
            for row in rows
            if row.get("score") is not None
        ]
        pass_count = sum(1 for row in rows if row.get("passed"))
        parse_error_count = sum(
            1 for row in rows
            if row.get("parse_error")
        )
        validation_fail_count = sum(
            1 for row in rows
            if not row.get("validation_ok")
        )
        durations = [
            float(row["duration_seconds"])
            for row in rows
            if row.get("duration_seconds") is not None
        ]

        summary_rows.append({
            "combo_id": combo_id,
            "combo_title": rows[0].get("combo_title"),
            "combo_kind": rows[0].get("combo_kind"),
            "base_combo_id": rows[0].get("base_combo_id"),
            "active_combo_id": rows[0].get("active_combo_id"),
            "system_prompt_id": rows[0].get("system_prompt_id"),
            "user_prompt_id": rows[0].get("user_prompt_id"),
            "fewshot_id": rows[0].get("fewshot_id"),
            "case_count": len(rows),
            "passed_case_count": pass_count,
            "failed_case_count": len(rows) - pass_count,
            "parse_error_count": parse_error_count,
            "validation_fail_count": validation_fail_count,
            "average_score": (
                round(mean(score_values), 4)
                if score_values else None
            ),
            "pass_rate": (
                round(pass_count / len(rows), 4)
                if rows else None
            ),
            "average_duration_seconds": (
                round(mean(durations), 4)
                if durations else None
            ),
            "request_types": sorted(
                {
                    row["request_type"]
                    for row in rows
                    if row.get("request_type")
                }
            ),
            "experiment_kinds": sorted(
                {
                    row["experiment_kind"]
                    for row in rows
                    if row.get("experiment_kind")
                }
            ),
        })

    return sorted(
        summary_rows,
        key=lambda row: (
            -(row.get("average_score") or 0.0),
            row.get("combo_id") or "",
        ),
    )


def case_summary_rows(
    review_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate review rows into case-level summaries."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        grouped[str(row.get("case_id"))].append(row)

    summary_rows: list[dict[str, Any]] = []

    for case_id, rows in grouped.items():
        score_values = [
            float(row["score"])
            for row in rows
            if row.get("score") is not None
        ]
        pass_count = sum(1 for row in rows if row.get("passed"))
        parse_error_count = sum(
            1 for row in rows
            if row.get("parse_error")
        )
        best_row = max(
            rows,
            key=lambda row: row.get("score") or -1.0,
        )

        summary_rows.append({
            "case_id": case_id,
            "question": rows[0].get("question"),
            "combo_count": len(rows),
            "passed_combo_count": pass_count,
            "failed_combo_count": len(rows) - pass_count,
            "parse_error_count": parse_error_count,
            "average_score": (
                round(mean(score_values), 4)
                if score_values else None
            ),
            "pass_rate": (
                round(pass_count / len(rows), 4)
                if rows else None
            ),
            "best_combo_id": best_row.get("combo_id"),
            "best_score": best_row.get("score"),
            "request_types": sorted(
                {
                    row["request_type"]
                    for row in rows
                    if row.get("request_type")
                }
            ),
            "experiment_kinds": sorted(
                {
                    row["experiment_kind"]
                    for row in rows
                    if row.get("experiment_kind")
                }
            ),
        })

    return sorted(
        summary_rows,
        key=lambda row: row.get("case_id") or "",
    )


def summary_metrics(
    review_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build top-level report metrics from flat review rows."""

    combo_ids = {
        row["combo_id"]
        for row in review_rows
        if row.get("combo_id")
    }
    case_ids = {
        row["case_id"]
        for row in review_rows
        if row.get("case_id")
    }
    score_values = [
        float(row["score"])
        for row in review_rows
        if row.get("score") is not None
    ]
    passed_rows = [
        row for row in review_rows
        if row.get("passed")
    ]
    parse_error_count = sum(
        1 for row in review_rows
        if row.get("parse_error")
    )
    tag_counts = Counter()
    for row in review_rows:
        tag_counts.update(row.get("tags", []))

    return {
        "review_row_count": len(review_rows),
        "combo_count": len(combo_ids),
        "case_count": len(case_ids),
        "average_score": (
            round(mean(score_values), 4)
            if score_values else None
        ),
        "pass_rate": (
            round(len(passed_rows) / len(review_rows), 4)
            if review_rows else None
        ),
        "parse_error_count": parse_error_count,
        "tag_counts": dict(sorted(tag_counts.items())),
    }


def write_summary_report(
    garden: PromptGarden,
    scope: str,
    artifacts: Sequence[dict[str, Any]],
    filters: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> Path:
    """Write a derived summary report for one review scope."""

    review_rows = build_review_rows(garden, artifacts)
    report_scope_dir = garden.ensure_report_scope_dir(scope)
    created_at = _now_iso()
    timestamp = filesystem_timestamp_from_iso(created_at)
    report_path = (
        report_scope_dir
        / f"{SUMMARY_REPORT_KIND}__{timestamp}.json"
    )
    suffix = 2
    while report_path.exists():
        report_path = (
            report_scope_dir
            / f"{SUMMARY_REPORT_KIND}__{timestamp}__{suffix:02d}.json"
        )
        suffix += 1

    normalized_files = sorted(
        {
            (
                artifact.get("artifact_paths") or {}
            ).get("normalized")
            for artifact in artifacts
            if (
                (artifact.get("artifact_paths") or {}).get(
                    "normalized"
                )
            )
        }
    )
    raw_files = sorted(
        {
            (
                artifact.get("artifact_paths") or {}
            ).get("raw")
            for artifact in artifacts
            if (
                (artifact.get("artifact_paths") or {}).get("raw")
            )
        }
    )

    report = {
        "id": (
            f"report_{scope}_{SUMMARY_REPORT_KIND}_{timestamp}"
        ),
        "schema_version": SUMMARY_REPORT_ARTIFACT_VERSION,
        "report_kind": SUMMARY_REPORT_KIND,
        "scope": scope,
        "created_at": created_at,
        "source_run_count": len(raw_files),
        "source_normalized_count": len(normalized_files),
        "artifacts": {
            "raw_files": raw_files,
            "normalized_files": normalized_files,
        },
        "filters": _safe_copy(filters or {}),
        "context": _safe_copy(context or {}),
        "summary_metrics": summary_metrics(review_rows),
        "combo_rows": combo_summary_rows(review_rows),
        "case_rows": case_summary_rows(review_rows),
    }
    write_json(report_path, report)
    return report_path
