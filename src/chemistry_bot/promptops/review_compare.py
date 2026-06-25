"""Reusable Prompt Garden answer-comparison helpers."""

from __future__ import annotations

from difflib import HtmlDiff, SequenceMatcher, unified_diff
from typing import Any, Mapping, Sequence
import json
import re

from .review_store import (
    normalize_inline_text,
    normalize_text_for_review,
    split_paragraphs,
)


DEFAULT_FIELD_PATHS = (
    "request_type",
    "certainty",
    "short_answer",
    "explanation",
    "examples",
    "experiment.kind",
    "experiment.reason",
    "experiment.questions",
    "source_ids",
)
TEXT_FIELD_PATHS = {
    "short_answer",
    "explanation",
    "experiment.reason",
    "comparison_text",
    "question",
    "raw_output",
}
LIST_FIELD_PATHS = {
    "examples",
    "experiment.questions",
    "source_ids",
}
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"'(]))"
)


def _nested_get(
    mapping: Mapping[str, Any],
    path_parts: Sequence[str],
) -> Any:
    current: Any = mapping
    for part in path_parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def record_case_id(record: Mapping[str, Any]) -> str | None:
    """Return the case id from a normalized artifact or review row."""

    value = record.get("case_id")
    return str(value) if value is not None else None


def record_combo_id(record: Mapping[str, Any]) -> str | None:
    """Return the combo id from a normalized artifact or review row."""

    value = record.get("combo_id")
    return str(value) if value is not None else None


def record_created_at(record: Mapping[str, Any]) -> str:
    """Return a sortable timestamp string for a record."""

    value = record.get("created_at")
    return str(value) if value is not None else ""


def record_score(record: Mapping[str, Any]) -> float | None:
    """Return the case-level numeric score when available."""

    direct = record.get("score")
    if direct is not None:
        return float(direct)

    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and metrics.get("score") is not None:
        return float(metrics["score"])

    return None


def record_passed(record: Mapping[str, Any]) -> bool | None:
    """Return the case pass/fail flag when available."""

    direct = record.get("passed")
    if isinstance(direct, bool):
        return direct

    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and isinstance(
        metrics.get("passed"),
        bool,
    ):
        return bool(metrics["passed"])

    return None


def record_parse_error(record: Mapping[str, Any]) -> bool:
    """Return whether the record carries a parse error flag."""

    review_flags = record.get("review_flags")
    if isinstance(review_flags, Mapping):
        return bool(review_flags.get("parse_error"))

    value = record.get("parse_error")
    return bool(value)


def record_combo_title(record: Mapping[str, Any]) -> str | None:
    """Return a human-readable combo title if one is present."""

    direct = record.get("combo_title")
    if direct:
        return str(direct)

    prompt_snapshot = record.get("prompt_snapshot")
    if not isinstance(prompt_snapshot, Mapping):
        return None

    base_combo = prompt_snapshot.get("base_combo")
    if isinstance(base_combo, Mapping) and base_combo.get("title"):
        return str(base_combo["title"])

    active_combo = prompt_snapshot.get("active_combo")
    if isinstance(active_combo, Mapping) and active_combo.get("title"):
        return str(active_combo["title"])

    return None


def _normalized_list_block(
    record: Mapping[str, Any],
    field_name: str,
) -> list[str]:
    blocks = record.get("normalized_text_blocks")
    if isinstance(blocks, Mapping):
        block = blocks.get(field_name)
        if isinstance(block, list):
            values: list[str] = []
            for item in block:
                if isinstance(item, Mapping):
                    values.append(
                        normalize_text_for_review(
                            item.get("normalized", "")
                        )
                    )
                else:
                    values.append(normalize_text_for_review(item))
            return [value for value in values if value]

    direct = record.get(field_name)
    if isinstance(direct, list):
        return [
            normalize_text_for_review(item)
            for item in direct
            if normalize_text_for_review(item)
        ]

    return []


def _normalized_text_block(
    record: Mapping[str, Any],
    field_name: str,
) -> str:
    blocks = record.get("normalized_text_blocks")
    if isinstance(blocks, Mapping):
        block = blocks.get(field_name)
        if isinstance(block, Mapping):
            return normalize_text_for_review(
                block.get("normalized", "")
            )

    direct = record.get(field_name)
    if direct is not None:
        return normalize_text_for_review(direct)

    return ""


def field_value(
    record: Mapping[str, Any],
    field_path: str,
) -> Any:
    """Return a normalized comparison value for a record field path."""

    if field_path == "short_answer":
        return _normalized_text_block(record, "short_answer")

    if field_path == "explanation":
        value = _normalized_text_block(record, "explanation")
        if value:
            return value
        preview = record.get("explanation_preview")
        return normalize_text_for_review(preview)

    if field_path == "examples":
        return _normalized_list_block(record, "examples")

    if field_path == "experiment.reason":
        return _normalized_text_block(record, "experiment_reason")

    if field_path == "experiment.questions":
        return _normalized_list_block(record, "experiment_questions")

    if field_path == "comparison_text":
        value = record.get("comparison_text")
        if value:
            return normalize_text_for_review(value)
        return normalize_text_for_review(
            _nested_get(
                record,
                ["normalized_text_blocks", "comparison_text"],
            )
        )

    if field_path == "raw_output":
        return _normalized_text_block(record, "raw_output")

    if field_path == "request_type":
        return record.get("request_type") or _nested_get(
            record,
            ["parsed_answer", "request_type"],
        )

    if field_path == "certainty":
        return record.get("certainty") or _nested_get(
            record,
            ["parsed_answer", "certainty"],
        )

    if field_path == "experiment.kind":
        return record.get("experiment_kind") or _nested_get(
            record,
            ["parsed_answer", "experiment", "kind"],
        )

    if field_path == "source_ids":
        source_ids = record.get("source_ids")
        if isinstance(source_ids, list):
            return [str(value) for value in source_ids]
        return []

    if field_path == "question":
        return normalize_text_for_review(record.get("question"))

    direct = _nested_get(record, field_path.split("."))
    if direct is not None:
        return direct

    parsed_answer = record.get("parsed_answer")
    if isinstance(parsed_answer, Mapping):
        nested = _nested_get(
            parsed_answer,
            field_path.split("."),
        )
        if nested is not None:
            return nested

    return None


def field_display_text(
    record: Mapping[str, Any],
    field_path: str,
) -> str:
    """Return a readable text form for one record field."""

    value = field_value(record, field_path)
    if value is None:
        return ""

    if isinstance(value, list):
        return "\n".join(
            normalize_text_for_review(item)
            for item in value
            if normalize_text_for_review(item)
        )

    if isinstance(value, str):
        return normalize_text_for_review(value)

    if isinstance(value, Mapping):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

    return str(value)


def text_similarity(
    left_text: Any,
    right_text: Any,
) -> float:
    """Return a SequenceMatcher-based similarity ratio in [0, 1]."""

    left = normalize_text_for_review(left_text)
    right = normalize_text_for_review(right_text)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).ratio()


def unified_text_diff(
    left_text: Any,
    right_text: Any,
    from_label: str = "baseline",
    to_label: str = "challenger",
    context_lines: int = 3,
) -> str:
    """Return a unified diff for two text values."""

    left = normalize_text_for_review(left_text).splitlines()
    right = normalize_text_for_review(right_text).splitlines()
    return "\n".join(
        unified_diff(
            left,
            right,
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
            n=context_lines,
        )
    )


def html_text_diff(
    left_text: Any,
    right_text: Any,
    from_label: str = "baseline",
    to_label: str = "challenger",
    wrapcolumn: int = 100,
) -> str:
    """Return an HTML side-by-side diff for two text values."""

    diff = HtmlDiff(wrapcolumn=wrapcolumn)
    return diff.make_table(
        normalize_text_for_review(left_text).splitlines(),
        normalize_text_for_review(right_text).splitlines(),
        fromdesc=from_label,
        todesc=to_label,
        context=True,
        numlines=3,
    )


def compare_field_values(
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    field_path: str,
) -> dict[str, Any]:
    """Compare one field across two review records."""

    left_value = field_value(left_record, field_path)
    right_value = field_value(right_record, field_path)
    equal = left_value == right_value

    comparison: dict[str, Any] = {
        "field_path": field_path,
        "left_value": left_value,
        "right_value": right_value,
        "equal": equal,
        "changed": not equal,
    }

    if field_path in TEXT_FIELD_PATHS:
        left_text = field_display_text(left_record, field_path)
        right_text = field_display_text(right_record, field_path)
        comparison["similarity"] = round(
            text_similarity(left_text, right_text),
            4,
        )
        comparison["diff_text"] = unified_text_diff(
            left_text,
            right_text,
            from_label="baseline",
            to_label="challenger",
        )
    elif field_path in LIST_FIELD_PATHS:
        left_text = field_display_text(left_record, field_path)
        right_text = field_display_text(right_record, field_path)
        comparison["similarity"] = round(
            text_similarity(left_text, right_text),
            4,
        )
        comparison["left_count"] = len(left_value or [])
        comparison["right_count"] = len(right_value or [])
    else:
        comparison["similarity"] = 1.0 if equal else 0.0

    return comparison


def compare_record_fields(
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    field_paths: Sequence[str] = DEFAULT_FIELD_PATHS,
) -> dict[str, Any]:
    """Compare a set of structured fields between two review records."""

    comparisons = [
        compare_field_values(
            left_record=left_record,
            right_record=right_record,
            field_path=field_path,
        )
        for field_path in field_paths
    ]
    changed_fields = [
        row["field_path"]
        for row in comparisons
        if row["changed"]
    ]

    return {
        "field_count": len(comparisons),
        "equal_field_count": len(comparisons) - len(changed_fields),
        "changed_field_count": len(changed_fields),
        "changed_fields": changed_fields,
        "comparisons": comparisons,
    }


def paragraph_segments(text: Any) -> list[str]:
    """Split review text into stable paragraph segments."""

    return split_paragraphs(text)


def sentence_segments(text: Any) -> list[str]:
    """Split review text into approximate sentence segments."""

    normalized = normalize_text_for_review(text)
    if not normalized:
        return []

    raw_segments = _SENTENCE_BOUNDARY_RE.split(normalized)
    return [
        segment.strip()
        for segment in raw_segments
        if segment.strip()
    ]


def sequence_alignment(
    left_segments: Sequence[str],
    right_segments: Sequence[str],
) -> list[dict[str, Any]]:
    """Align two ordered segment lists with difflib opcodes."""

    matcher = SequenceMatcher(
        None,
        list(left_segments),
        list(right_segments),
        autojunk=False,
    )
    rows: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        rows.append({
            "tag": tag,
            "left_start": i1,
            "left_end": i2,
            "right_start": j1,
            "right_end": j2,
            "left_segments": list(left_segments[i1:i2]),
            "right_segments": list(right_segments[j1:j2]),
        })

    return rows


def field_segment_alignment(
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    field_path: str,
    granularity: str = "paragraph",
) -> dict[str, Any]:
    """Align paragraph or sentence segments for one field."""

    left_text = field_display_text(left_record, field_path)
    right_text = field_display_text(right_record, field_path)

    if granularity == "paragraph":
        left_segments = paragraph_segments(left_text)
        right_segments = paragraph_segments(right_text)
    elif granularity == "sentence":
        left_segments = sentence_segments(left_text)
        right_segments = sentence_segments(right_text)
    else:
        raise ValueError(
            "granularity must be 'paragraph' or 'sentence'"
        )

    return {
        "field_path": field_path,
        "granularity": granularity,
        "left_count": len(left_segments),
        "right_count": len(right_segments),
        "alignment": sequence_alignment(
            left_segments,
            right_segments,
        ),
    }


def full_text_diff(
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
) -> str:
    """Return a unified diff for the canonical comparison text."""

    return unified_text_diff(
        field_display_text(left_record, "comparison_text"),
        field_display_text(right_record, "comparison_text"),
        from_label="baseline",
        to_label="challenger",
    )


def latest_records_by_case_combo(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep only the latest record for each (case_id, combo_id) pair."""

    index: dict[tuple[str | None, str | None], Mapping[str, Any]] = {}

    for record in records:
        key = (
            record_case_id(record),
            record_combo_id(record),
        )
        current = index.get(key)
        if current is None or record_created_at(record) >= record_created_at(
            current
        ):
            index[key] = record

    return sorted(
        index.values(),
        key=lambda record: (
            record_case_id(record) or "",
            record_combo_id(record) or "",
            record_created_at(record),
        ),
    )


def build_baseline_vs_challenger(
    records: Sequence[Mapping[str, Any]],
    baseline_combo_id: str,
    field_paths: Sequence[str] = DEFAULT_FIELD_PATHS,
    case_ids: Sequence[str] = (),
    challenger_combo_ids: Sequence[str] = (),
    min_meaningful_score_delta: float = 0.05,
    max_meaningful_text_similarity: float = 0.985,
) -> list[dict[str, Any]]:
    """Compare all challengers against one baseline combo by case."""

    selected_case_ids = set(case_ids)
    selected_challenger_ids = set(challenger_combo_ids)
    latest_records = latest_records_by_case_combo(records)
    by_case: dict[str, list[Mapping[str, Any]]] = {}

    for record in latest_records:
        case_id = record_case_id(record)
        if not case_id:
            continue
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        by_case.setdefault(case_id, []).append(record)

    comparison_rows: list[dict[str, Any]] = []

    for case_id, case_records in sorted(by_case.items()):
        baseline_record = next(
            (
                record
                for record in case_records
                if record_combo_id(record) == baseline_combo_id
            ),
            None,
        )
        if baseline_record is None:
            continue

        baseline_score = record_score(baseline_record)
        baseline_passed = record_passed(baseline_record)

        for challenger in case_records:
            challenger_combo_id = record_combo_id(challenger)
            if challenger_combo_id == baseline_combo_id:
                continue
            if (
                selected_challenger_ids
                and challenger_combo_id not in selected_challenger_ids
            ):
                continue

            field_summary = compare_record_fields(
                left_record=baseline_record,
                right_record=challenger,
                field_paths=field_paths,
            )
            score_delta = None
            challenger_score = record_score(challenger)
            if baseline_score is not None and challenger_score is not None:
                score_delta = round(
                    challenger_score - baseline_score,
                    4,
                )

            baseline_text = field_display_text(
                baseline_record,
                "comparison_text",
            )
            challenger_text = field_display_text(
                challenger,
                "comparison_text",
            )
            full_similarity = round(
                text_similarity(
                    baseline_text,
                    challenger_text,
                ),
                4,
            )
            pass_delta = None
            challenger_passed = record_passed(challenger)
            if (
                isinstance(baseline_passed, bool)
                and isinstance(challenger_passed, bool)
            ):
                pass_delta = int(challenger_passed) - int(
                    baseline_passed
                )

            meaningful_delta = any([
                field_summary["changed_field_count"] > 0,
                score_delta is not None
                and abs(score_delta) >= min_meaningful_score_delta,
                pass_delta not in (None, 0),
                record_parse_error(baseline_record)
                != record_parse_error(challenger),
                full_similarity <= max_meaningful_text_similarity,
            ])

            comparison_rows.append({
                "case_id": case_id,
                "question": baseline_record.get("question"),
                "baseline_combo_id": baseline_combo_id,
                "baseline_combo_title": record_combo_title(
                    baseline_record
                ),
                "challenger_combo_id": challenger_combo_id,
                "challenger_combo_title": record_combo_title(
                    challenger
                ),
                "baseline_score": baseline_score,
                "challenger_score": challenger_score,
                "score_delta": score_delta,
                "baseline_passed": baseline_passed,
                "challenger_passed": challenger_passed,
                "pass_delta": pass_delta,
                "baseline_parse_error": record_parse_error(
                    baseline_record
                ),
                "challenger_parse_error": record_parse_error(
                    challenger
                ),
                "full_text_similarity": full_similarity,
                "field_summary": field_summary,
                "full_text_diff": full_text_diff(
                    baseline_record,
                    challenger,
                ),
                "paragraph_alignment": field_segment_alignment(
                    baseline_record,
                    challenger,
                    field_path="comparison_text",
                    granularity="paragraph",
                ),
                "sentence_alignment": field_segment_alignment(
                    baseline_record,
                    challenger,
                    field_path="comparison_text",
                    granularity="sentence",
                ),
                "meaningful_delta": meaningful_delta,
            })

    return sorted(
        comparison_rows,
        key=lambda row: (
            row.get("case_id") or "",
            row.get("challenger_combo_id") or "",
        ),
    )


def filter_meaningful_comparisons(
    comparison_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only comparisons flagged as meaningful."""

    return [
        dict(row)
        for row in comparison_rows
        if row.get("meaningful_delta")
    ]
