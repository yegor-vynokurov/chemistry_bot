"""Ranking and summary-metric helpers for Prompt Garden review tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .review_compare import (
    DEFAULT_FIELD_PATHS,
    build_baseline_vs_challenger,
)
from .review_store import case_summary_rows, combo_summary_rows


DEFAULT_COMBO_RANK_WEIGHTS = {
    "average_score": 0.55,
    "pass_rate": 0.25,
    "stability_rate": 0.10,
    "validation_rate": 0.05,
    "speed_score": 0.05,
}


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _range_summary(
    values: Sequence[float],
) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "average": None,
        }

    return {
        "min": min(values),
        "max": max(values),
        "average": round(mean(values), 4),
    }


def _speed_scores(
    combo_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    durations = [
        float(row["average_duration_seconds"])
        for row in combo_rows
        if row.get("average_duration_seconds") is not None
    ]
    if not durations:
        return {}

    fastest = min(durations)
    slowest = max(durations)
    if fastest == slowest:
        return {
            str(row["combo_id"]): 1.0
            for row in combo_rows
        }

    scores: dict[str, float] = {}
    for row in combo_rows:
        duration = row.get("average_duration_seconds")
        if duration is None:
            scores[str(row["combo_id"])] = 0.0
            continue
        scores[str(row["combo_id"])] = round(
            1.0 - (
                (float(duration) - fastest)
                / (slowest - fastest)
            ),
            4,
        )

    return scores


def baseline_challenger_rows(
    records: Sequence[Mapping[str, Any]],
    baseline_combo_id: str,
    field_paths: Sequence[str] = DEFAULT_FIELD_PATHS,
    case_ids: Sequence[str] = (),
    challenger_combo_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Aggregate baseline-vs-challenger comparisons by challenger combo."""

    comparison_rows = build_baseline_vs_challenger(
        records=records,
        baseline_combo_id=baseline_combo_id,
        field_paths=field_paths,
        case_ids=case_ids,
        challenger_combo_ids=challenger_combo_ids,
    )
    by_challenger: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in comparison_rows:
        challenger_combo_id = row.get("challenger_combo_id")
        if challenger_combo_id:
            by_challenger[str(challenger_combo_id)].append(row)

    summary_rows: list[dict[str, Any]] = []

    for challenger_combo_id, rows in by_challenger.items():
        score_deltas = [
            float(row["score_delta"])
            for row in rows
            if row.get("score_delta") is not None
        ]
        similarities = [
            float(row["full_text_similarity"])
            for row in rows
            if row.get("full_text_similarity") is not None
        ]
        changed_field_counts = [
            int(
                (row.get("field_summary") or {}).get(
                    "changed_field_count",
                    0,
                )
            )
            for row in rows
        ]
        meaningful_count = sum(
            1 for row in rows
            if row.get("meaningful_delta")
        )
        improved_score_count = sum(
            1 for row in rows
            if (row.get("score_delta") or 0.0) > 0
        )
        worsened_score_count = sum(
            1 for row in rows
            if (row.get("score_delta") or 0.0) < 0
        )
        pass_gain_count = sum(
            1 for row in rows
            if row.get("pass_delta") == 1
        )
        pass_loss_count = sum(
            1 for row in rows
            if row.get("pass_delta") == -1
        )

        summary_rows.append({
            "baseline_combo_id": baseline_combo_id,
            "baseline_combo_title": rows[0].get(
                "baseline_combo_title"
            ),
            "challenger_combo_id": challenger_combo_id,
            "challenger_combo_title": rows[0].get(
                "challenger_combo_title"
            ),
            "compared_case_count": len(rows),
            "meaningful_case_count": meaningful_count,
            "meaningful_case_rate": round(
                meaningful_count / len(rows),
                4,
            ),
            "improved_score_case_count": improved_score_count,
            "worsened_score_case_count": worsened_score_count,
            "same_score_case_count": (
                len(rows)
                - improved_score_count
                - worsened_score_count
            ),
            "pass_gain_case_count": pass_gain_count,
            "pass_loss_case_count": pass_loss_count,
            "average_score_delta": (
                round(_safe_mean(score_deltas) or 0.0, 4)
                if score_deltas else None
            ),
            "average_text_similarity": (
                round(_safe_mean(similarities) or 0.0, 4)
                if similarities else None
            ),
            "average_changed_field_count": round(
                _safe_mean(changed_field_counts) or 0.0,
                4,
            ),
            "changed_field_rate": round(
                sum(1 for count in changed_field_counts if count > 0)
                / len(rows),
                4,
            ),
        })

    return sorted(
        summary_rows,
        key=lambda row: (
            -(row.get("average_score_delta") or 0.0),
            -(row.get("meaningful_case_rate") or 0.0),
            row.get("challenger_combo_id") or "",
        ),
    )


def combo_ranking_rows(
    review_rows: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]] = (),
    baseline_combo_id: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank combos for review.

    The ranking score is only a reviewer aid. It is not a correctness metric.
    """

    weights_map = {
        **DEFAULT_COMBO_RANK_WEIGHTS,
        **(weights or {}),
    }
    summary_rows = combo_summary_rows(review_rows)
    speed_scores = _speed_scores(summary_rows)
    baseline_index = {}

    if baseline_combo_id and baseline_records:
        baseline_rows = baseline_challenger_rows(
            records=baseline_records,
            baseline_combo_id=baseline_combo_id,
        )
        baseline_index = {
            row["challenger_combo_id"]: row
            for row in baseline_rows
        }

    ranked_rows: list[dict[str, Any]] = []

    for row in summary_rows:
        case_count = max(int(row.get("case_count") or 0), 1)
        stability_rate = 1.0 - (
            int(row.get("parse_error_count") or 0) / case_count
        )
        validation_rate = 1.0 - (
            int(row.get("validation_fail_count") or 0) / case_count
        )
        speed_score = speed_scores.get(str(row["combo_id"]), 0.0)
        ranking_score = (
            (float(row.get("average_score") or 0.0)
             * weights_map["average_score"])
            + (float(row.get("pass_rate") or 0.0)
               * weights_map["pass_rate"])
            + (stability_rate * weights_map["stability_rate"])
            + (validation_rate * weights_map["validation_rate"])
            + (speed_score * weights_map["speed_score"])
        )

        baseline_metrics = baseline_index.get(row["combo_id"])
        ranked_rows.append({
            **row,
            "stability_rate": round(stability_rate, 4),
            "validation_rate": round(validation_rate, 4),
            "speed_score": speed_score,
            "ranking_score": round(ranking_score, 4),
            "average_score_delta_vs_baseline": (
                baseline_metrics.get("average_score_delta")
                if baseline_metrics else None
            ),
            "meaningful_case_rate_vs_baseline": (
                baseline_metrics.get("meaningful_case_rate")
                if baseline_metrics else None
            ),
        })

    ranked_rows.sort(
        key=lambda row: (
            -(row.get("ranking_score") or 0.0),
            -(row.get("average_score") or 0.0),
            row.get("combo_id") or "",
        )
    )

    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index

    return ranked_rows


def case_difficulty_rows(
    review_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank cases by how difficult or disagreement-prone they are."""

    summary_rows = case_summary_rows(review_rows)
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in review_rows:
        case_id = row.get("case_id")
        if case_id:
            by_case[str(case_id)].append(row)

    ranked_rows: list[dict[str, Any]] = []

    for row in summary_rows:
        case_rows = by_case.get(str(row["case_id"]), [])
        score_values = [
            float(case_row["score"])
            for case_row in case_rows
            if case_row.get("score") is not None
        ]
        score_range = (
            round(max(score_values) - min(score_values), 4)
            if score_values else 0.0
        )
        request_type_count = len(
            {
                case_row.get("request_type")
                for case_row in case_rows
                if case_row.get("request_type")
            }
        )
        experiment_kind_count = len(
            {
                case_row.get("experiment_kind")
                for case_row in case_rows
                if case_row.get("experiment_kind")
            }
        )
        parse_error_rate = (
            int(row.get("parse_error_count") or 0)
            / max(int(row.get("combo_count") or 1), 1)
        )
        failure_rate = 1.0 - float(row.get("pass_rate") or 0.0)
        disagreement_rate = min(
            1.0,
            (score_range + max(request_type_count - 1, 0) * 0.2)
            + max(experiment_kind_count - 1, 0) * 0.15,
        )
        difficulty_score = (
            failure_rate * 0.6
            + disagreement_rate * 0.25
            + parse_error_rate * 0.15
        )

        ranked_rows.append({
            **row,
            "score_range": score_range,
            "request_type_count": request_type_count,
            "experiment_kind_count": experiment_kind_count,
            "parse_error_rate": round(parse_error_rate, 4),
            "difficulty_score": round(difficulty_score, 4),
        })

    ranked_rows.sort(
        key=lambda row: (
            -(row.get("difficulty_score") or 0.0),
            row.get("case_id") or "",
        )
    )

    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index

    return ranked_rows


def review_filter_metrics(
    review_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build filter-ready category counts and numeric ranges."""

    request_type_counts = Counter()
    experiment_kind_counts = Counter()
    tag_counts = Counter()
    combo_counts = Counter()
    case_counts = Counter()
    score_values: list[float] = []
    duration_values: list[float] = []
    source_count_values: list[float] = []
    example_count_values: list[float] = []

    for row in review_rows:
        if row.get("request_type"):
            request_type_counts[str(row["request_type"])] += 1
        if row.get("experiment_kind"):
            experiment_kind_counts[str(row["experiment_kind"])] += 1
        for tag in row.get("tags", []):
            tag_counts[str(tag)] += 1
        if row.get("combo_id"):
            combo_counts[str(row["combo_id"])] += 1
        if row.get("case_id"):
            case_counts[str(row["case_id"])] += 1
        if row.get("score") is not None:
            score_values.append(float(row["score"]))
        if row.get("duration_seconds") is not None:
            duration_values.append(float(row["duration_seconds"]))
        if row.get("source_count") is not None:
            source_count_values.append(float(row["source_count"]))
        if row.get("example_count") is not None:
            example_count_values.append(float(row["example_count"]))

    return {
        "row_count": len(review_rows),
        "request_type_counts": dict(sorted(request_type_counts.items())),
        "experiment_kind_counts": dict(
            sorted(experiment_kind_counts.items())
        ),
        "tag_counts": dict(sorted(tag_counts.items())),
        "combo_counts": dict(sorted(combo_counts.items())),
        "case_counts": dict(sorted(case_counts.items())),
        "ranges": {
            "score": _range_summary(score_values),
            "duration_seconds": _range_summary(duration_values),
            "source_count": _range_summary(source_count_values),
            "example_count": _range_summary(example_count_values),
        },
    }
