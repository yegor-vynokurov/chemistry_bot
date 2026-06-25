"""Streamlit review app for Prompt Garden experiment inspection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import io
import json
import sys

import streamlit as st
import streamlit.components.v1 as components


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chemistry_bot.promptops.garden import PromptGarden
from chemistry_bot.promptops.review_compare import (
    DEFAULT_FIELD_PATHS,
    build_baseline_vs_challenger,
    compare_record_fields,
    field_display_text,
    field_segment_alignment,
    html_text_diff,
    latest_records_by_case_combo,
    unified_text_diff,
)
from chemistry_bot.promptops.review_embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_DIMENSIONS,
    load_similarity_cache,
    similarity_bundle,
    write_similarity_cache,
)
from chemistry_bot.promptops.review_metrics import (
    baseline_challenger_rows,
    case_difficulty_rows,
    combo_ranking_rows,
    review_filter_metrics,
)
from chemistry_bot.promptops.review_store import (
    build_review_rows,
    case_summary_rows,
    combo_summary_rows,
    load_normalized_scope,
    summary_metrics,
)


REVIEW_NOTES_SCHEMA_VERSION = "prompt-garden-review-notes-v1"
TEXT_COMPARE_FIELDS = (
    "short_answer",
    "explanation",
    "experiment.reason",
    "comparison_text",
    "raw_output",
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_garden_root() -> Path:
    return REPO_ROOT / "prompt_garden"


def _review_notes_path(
    garden_root: str | Path,
    scope: str,
) -> Path:
    garden = PromptGarden(garden_root)
    return garden.ensure_report_scope_dir(scope) / "review_notes.json"


def _short_signature(signature: str | None) -> str:
    if not signature:
        return "no_sig"
    return signature[:10]


def _fewshot_label(value: Any) -> str:
    if value in (None, "", "none"):
        return "no_fewshot"
    return str(value)


def _note_key(row_id: str) -> str:
    return row_id


def _load_review_notes(
    garden_root: str | Path,
    scope: str,
) -> dict[str, Any]:
    path = _review_notes_path(garden_root, scope)
    if not path.exists():
        return {
            "schema_version": REVIEW_NOTES_SCHEMA_VERSION,
            "scope": scope,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "entries": {},
        }

    return json.loads(path.read_text(encoding="utf-8"))


def _save_review_notes(
    garden_root: str | Path,
    scope: str,
    payload: dict[str, Any],
) -> Path:
    path = _review_notes_path(garden_root, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _stable_json(payload) + "\n",
        encoding="utf-8",
    )
    return path


def _note_rows(notes_payload: dict[str, Any]) -> list[dict[str, Any]]:
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


def _upsert_review_note(
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
    payload = _load_review_notes(garden_root, scope)
    entries = payload.setdefault("entries", {})
    key = _note_key(row_id)
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
        "updated_at": _now_iso(),
    }
    payload["updated_at"] = _now_iso()
    return _save_review_notes(garden_root, scope, payload)


def _normalize_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
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
            key: _normalize_for_csv(row.get(key))
            for key in fieldnames
        })
    return buffer.getvalue()


def _score_bounds(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [
        float(row["score"])
        for row in rows
        if row.get("score") is not None
    ]
    if not values:
        return (0.0, 1.0)
    return (min(values), max(values))


def _row_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('case_id', '-')}"
        f" | {row.get('combo_id', '-')}"
        f" | model={row.get('model', '-')}"
        f" | fewshot={_fewshot_label(row.get('fewshot_id'))}"
        f" | score={row.get('score', '-')}"
        f" | sig={_short_signature(row.get('execution_signature'))}"
        f" | {row.get('combo_title', '')}"
    )


def _artifact_label(artifact: dict[str, Any]) -> str:
    prompt_snapshot = artifact.get("prompt_snapshot") or {}
    base_combo = prompt_snapshot.get("base_combo") or {}
    combo_title = base_combo.get("title") or artifact.get("combo_id")
    score = (artifact.get("metrics") or {}).get("score")
    execution = artifact.get("execution") or {}
    return (
        f"{artifact.get('case_id', '-')}"
        f" | {artifact.get('combo_id', '-')}"
        f" | model={artifact.get('model', '-')}"
        f" | fewshot={_fewshot_label(execution.get('fewshot_id'))}"
        f" | score={score}"
        f" | sig={_short_signature(execution.get('signature'))}"
        f" | {combo_title}"
    )


def _signature_label(signature: str) -> str:
    if not signature:
        return "(missing signature)"
    return signature


def _filter_review_data(
    artifacts: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    selected_signatures: list[str],
    selected_case_sets: list[str],
    selected_combo_ids: list[str],
    selected_case_ids: list[str],
    selected_request_types: list[str],
    selected_experiment_kinds: list[str],
    pass_status: str,
    parse_status: str,
    score_min: float,
    score_max: float,
    search_text: str,
    latest_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_by_id = {
        str(artifact["id"]): artifact
        for artifact in artifacts
    }
    filtered_rows: list[dict[str, Any]] = []
    search_query = search_text.strip().lower()

    for row in review_rows:
        if (
            selected_signatures
            and row.get("execution_signature") not in selected_signatures
        ):
            continue
        if selected_case_sets and row.get("case_set_id") not in selected_case_sets:
            continue
        if selected_combo_ids and row.get("combo_id") not in selected_combo_ids:
            continue
        if selected_case_ids and row.get("case_id") not in selected_case_ids:
            continue
        if selected_request_types and row.get("request_type") not in selected_request_types:
            continue
        if selected_experiment_kinds and row.get("experiment_kind") not in selected_experiment_kinds:
            continue
        if pass_status == "passed" and not row.get("passed"):
            continue
        if pass_status == "failed" and row.get("passed"):
            continue
        if parse_status == "parse_error" and not row.get("parse_error"):
            continue
        if parse_status == "parsed_ok" and row.get("parse_error"):
            continue
        score = row.get("score")
        if score is not None and (float(score) < score_min or float(score) > score_max):
            continue
        if search_query:
            searchable = " ".join(
                [
                    str(row.get("question", "")),
                    str(row.get("short_answer", "")),
                    str(row.get("combo_title", "")),
                    str(row.get("combo_id", "")),
                    str(row.get("case_id", "")),
                ]
            ).lower()
            if search_query not in searchable:
                continue
        filtered_rows.append(row)

    filtered_artifacts = [
        artifact_by_id[str(row["row_id"])]
        for row in filtered_rows
        if str(row["row_id"]) in artifact_by_id
    ]

    if latest_only:
        latest_artifacts = latest_records_by_case_combo(filtered_artifacts)
        latest_ids = {str(artifact["id"]) for artifact in latest_artifacts}
        filtered_rows = [
            row for row in filtered_rows
            if str(row["row_id"]) in latest_ids
        ]
        filtered_artifacts = latest_artifacts

    filtered_rows.sort(
        key=lambda row: (
            row.get("case_id") or "",
            row.get("combo_id") or "",
            row.get("created_at") or "",
        )
    )
    filtered_artifacts.sort(
        key=lambda artifact: (
            artifact.get("case_id") or "",
            artifact.get("combo_id") or "",
            artifact.get("created_at") or "",
        )
    )
    return filtered_artifacts, filtered_rows


@st.cache_data(show_spinner=False)
def discover_review_scopes(garden_root: str) -> list[dict[str, Any]]:
    garden = PromptGarden(garden_root)
    garden.init()

    scopes: list[dict[str, Any]] = []
    if not garden.normalized_runs_dir.exists():
        return scopes

    for scope_dir in sorted(garden.normalized_runs_dir.iterdir()):
        if not scope_dir.is_dir():
            continue
        artifact_files = sorted(scope_dir.glob("*.json"))
        if not artifact_files:
            continue

        scope = scope_dir.name
        experiment = None
        if scope.startswith("exp_"):
            try:
                experiment = garden.get_experiment(scope)
            except KeyError:
                experiment = None

        scopes.append({
            "scope": scope,
            "label": (
                f"{scope}"
                + (
                    f" | {experiment.get('name', '')}"
                    if experiment else ""
                )
            ),
            "artifact_count": len(artifact_files),
            "experiment_name": experiment.get("name") if experiment else None,
            "experiment_status": experiment.get("status") if experiment else None,
            "experiment_tags": list(experiment.get("tags", [])) if experiment else [],
        })

    scopes.sort(
        key=lambda row: row["scope"],
        reverse=True,
    )
    return scopes


@st.cache_data(show_spinner=False)
def load_scope_bundle(
    garden_root: str,
    scope: str,
) -> dict[str, Any]:
    garden = PromptGarden(garden_root)
    garden.init()
    artifacts = load_normalized_scope(
        garden_root=garden_root,
        scope=scope,
    )
    review_rows = build_review_rows(garden, artifacts)
    combo_rows = combo_summary_rows(review_rows)
    case_rows = case_summary_rows(review_rows)
    experiment = None
    if scope.startswith("exp_"):
        try:
            experiment = garden.get_experiment(scope)
        except KeyError:
            experiment = None

    signatures = sorted(
        {
            str((artifact.get("execution") or {}).get("signature"))
            for artifact in artifacts
            if (artifact.get("execution") or {}).get("signature")
        }
    )
    report_scope_dir = garden.ensure_report_scope_dir(scope)
    report_files = [
        path.name
        for path in sorted(report_scope_dir.glob("*.json"))
    ]

    return {
        "artifacts": artifacts,
        "review_rows": review_rows,
        "combo_rows": combo_rows,
        "case_rows": case_rows,
        "summary_metrics": summary_metrics(review_rows),
        "experiment": experiment,
        "signatures": signatures,
        "report_files": report_files,
        "embedding_cache_dir": str(garden.embedding_cache_dir),
    }


def _render_metric_row(metrics: dict[str, Any]) -> None:
    columns = st.columns(5)
    columns[0].metric("Rows", metrics.get("review_row_count", 0))
    columns[1].metric("Combos", metrics.get("combo_count", 0))
    columns[2].metric("Cases", metrics.get("case_count", 0))
    columns[3].metric("Avg Score", metrics.get("average_score"))
    columns[4].metric("Pass Rate", metrics.get("pass_rate"))


def _render_answer_column(
    artifact: dict[str, Any],
    title: str,
) -> None:
    metrics = artifact.get("metrics") or {}
    parsed_answer = artifact.get("parsed_answer") or {}
    prompt_snapshot = artifact.get("prompt_snapshot") or {}
    base_combo = prompt_snapshot.get("base_combo") or {}
    active_combo = prompt_snapshot.get("active_combo") or {}
    execution = artifact.get("execution") or {}

    st.subheader(title)
    st.caption(
        f"combo={artifact.get('combo_id')} | "
        f"active={artifact.get('active_combo_id')} | "
        f"model={artifact.get('model')} | "
        f"fewshot={_fewshot_label(execution.get('fewshot_id'))} | "
        f"sig={_short_signature(execution.get('signature'))} | "
        f"score={metrics.get('score')} | "
        f"passed={metrics.get('passed')}"
    )
    st.markdown("**Short Answer**")
    st.write(field_display_text(artifact, "short_answer") or "(empty)")
    st.markdown("**Explanation**")
    st.write(field_display_text(artifact, "explanation") or "(empty)")
    st.markdown("**Experiment Reason**")
    st.write(field_display_text(artifact, "experiment.reason") or "(empty)")
    st.markdown("**Structured Fields**")
    st.json({
        "request_type": parsed_answer.get("request_type"),
        "certainty": parsed_answer.get("certainty"),
        "experiment_kind": (
            (parsed_answer.get("experiment") or {}).get("kind")
            if isinstance(parsed_answer.get("experiment"), dict)
            else None
        ),
        "source_ids": parsed_answer.get("source_ids", []),
        "examples": parsed_answer.get("examples", []),
    })
    with st.expander("Prompt Snapshot"):
        st.json({
            "base_combo": {
                "id": base_combo.get("id"),
                "title": base_combo.get("title"),
                "prompt_ids": base_combo.get("prompt_ids"),
            },
            "active_combo": {
                "id": active_combo.get("id"),
                "title": active_combo.get("title"),
                "prompt_ids": active_combo.get("prompt_ids"),
            },
            "changed_roles": (
                (prompt_snapshot.get("combo_relation") or {}).get(
                    "changed_roles"
                )
            ),
        })


def _render_alignment_rows(
    alignment_payload: dict[str, Any],
) -> None:
    rows = []
    for item in alignment_payload.get("alignment", []):
        rows.append({
            "tag": item.get("tag"),
            "left_segments": "\n\n".join(item.get("left_segments", [])),
            "right_segments": "\n\n".join(item.get("right_segments", [])),
        })
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def _render_overview_tab(
    filtered_artifacts: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    filtered_metrics: dict[str, Any],
    experiment: dict[str, Any] | None,
    report_files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    st.subheader("Overview")
    _render_metric_row(filtered_metrics)

    if experiment:
        with st.expander("Experiment Metadata"):
            st.json({
                "id": experiment.get("id"),
                "name": experiment.get("name"),
                "status": experiment.get("status"),
                "goal": experiment.get("goal"),
                "hypothesis": experiment.get("hypothesis"),
                "tags": experiment.get("tags", []),
                "combo_count": len(experiment.get("combo_ids", [])),
            })

    if report_files:
        with st.expander("Existing Derived Reports"):
            st.write(report_files)

    ranking_rows = combo_ranking_rows(filtered_rows)
    difficulty_rows = case_difficulty_rows(filtered_rows)
    filter_metrics = review_filter_metrics(filtered_rows)

    ranking_cols = [
        "rank",
        "combo_id",
        "combo_title",
        "average_score",
        "pass_rate",
        "ranking_score",
        "average_duration_seconds",
        "case_count",
        "failed_case_count",
        "parse_error_count",
    ]
    case_cols = [
        "rank",
        "case_id",
        "question",
        "difficulty_score",
        "average_score",
        "pass_rate",
        "score_range",
        "combo_count",
        "parse_error_count",
    ]
    row_cols = [
        "row_id",
        "combo_id",
        "combo_title",
        "case_id",
        "score",
        "passed",
        "request_type",
        "experiment_kind",
        "source_count",
        "duration_seconds",
        "short_answer",
    ]

    table_tab, ranking_tab, difficulty_tab = st.tabs(
        ["Filtered Rows", "Combo Ranking", "Case Difficulty"]
    )
    with table_tab:
        st.dataframe(
            [{key: row.get(key) for key in row_cols} for row in filtered_rows],
            use_container_width=True,
            hide_index=True,
        )
    with ranking_tab:
        st.dataframe(
            [{key: row.get(key) for key in ranking_cols} for row in ranking_rows],
            use_container_width=True,
            hide_index=True,
        )
    with difficulty_tab:
        st.dataframe(
            [{key: row.get(key) for key in case_cols} for row in difficulty_rows],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Filter Diagnostics"):
        st.json(filter_metrics)

    return ranking_rows, difficulty_rows, filter_metrics


def _render_compare_tab(
    filtered_artifacts: list[dict[str, Any]],
) -> None:
    st.subheader("Side-by-Side Comparison")
    if len(filtered_artifacts) < 2:
        st.info("At least two filtered artifacts are needed for comparison.")
        return

    case_ids = sorted(
        {
            artifact.get("case_id")
            for artifact in filtered_artifacts
            if artifact.get("case_id")
        }
    )
    selected_case_id = st.selectbox(
        "Case",
        options=case_ids,
        key="compare_case_id",
    )

    case_artifacts = [
        artifact for artifact in filtered_artifacts
        if artifact.get("case_id") == selected_case_id
    ]
    artifact_labels = {
        _artifact_label(artifact): artifact
        for artifact in case_artifacts
    }
    labels = list(artifact_labels.keys())
    if len(labels) < 2:
        st.info("The selected case has fewer than two answers to compare.")
        return

    left_label = st.selectbox(
        "Left Answer",
        options=labels,
        index=0,
        key="compare_left",
    )
    right_label = st.selectbox(
        "Right Answer",
        options=labels,
        index=min(1, len(labels) - 1),
        key="compare_right",
    )

    left_artifact = artifact_labels[left_label]
    right_artifact = artifact_labels[right_label]

    left_col, right_col = st.columns(2)
    with left_col:
        _render_answer_column(left_artifact, "Left")
    with right_col:
        _render_answer_column(right_artifact, "Right")

    st.markdown("### Structured Field Comparison")
    field_summary = compare_record_fields(
        left_record=left_artifact,
        right_record=right_artifact,
        field_paths=DEFAULT_FIELD_PATHS,
    )
    comparison_rows = []
    for row in field_summary["comparisons"]:
        comparison_rows.append({
            "field_path": row.get("field_path"),
            "changed": row.get("changed"),
            "similarity": row.get("similarity"),
            "left_value": row.get("left_value"),
            "right_value": row.get("right_value"),
        })
    st.dataframe(
        comparison_rows,
        use_container_width=True,
        hide_index=True,
    )

    diff_field = st.selectbox(
        "Text Field For Diff",
        options=TEXT_COMPARE_FIELDS,
        index=TEXT_COMPARE_FIELDS.index("comparison_text"),
        key="compare_diff_field",
    )

    left_text = field_display_text(left_artifact, diff_field)
    right_text = field_display_text(right_artifact, diff_field)
    paragraph_alignment = field_segment_alignment(
        left_artifact,
        right_artifact,
        field_path=diff_field,
        granularity="paragraph",
    )
    sentence_alignment = field_segment_alignment(
        left_artifact,
        right_artifact,
        field_path=diff_field,
        granularity="sentence",
    )

    diff_tab, html_tab, paragraph_tab, sentence_tab = st.tabs(
        ["Unified Diff", "HTML Diff", "Paragraph Alignment", "Sentence Alignment"]
    )
    with diff_tab:
        diff_text = unified_text_diff(
            left_text,
            right_text,
            from_label="left",
            to_label="right",
        )
        st.code(diff_text or "(no diff)", language="diff")
    with html_tab:
        components.html(
            html_text_diff(
                left_text,
                right_text,
                from_label="left",
                to_label="right",
            ),
            height=500,
            scrolling=True,
        )
    with paragraph_tab:
        _render_alignment_rows(paragraph_alignment)
    with sentence_tab:
        _render_alignment_rows(sentence_alignment)


def _render_baseline_tab(
    filtered_artifacts: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    st.subheader("Baseline vs Challenger")
    combo_options = sorted(
        {
            (row.get("combo_id"), row.get("combo_title"))
            for row in filtered_rows
            if row.get("combo_id")
        }
    )
    if len(combo_options) < 2:
        st.info("At least two combos are needed for baseline review.")
        return [], [], None

    label_to_combo_id = {
        f"{combo_id} | {combo_title}": combo_id
        for combo_id, combo_title in combo_options
    }
    baseline_label = st.selectbox(
        "Baseline Combo",
        options=list(label_to_combo_id.keys()),
        key="baseline_combo",
    )
    baseline_combo_id = label_to_combo_id[baseline_label]

    comparison_rows = build_baseline_vs_challenger(
        records=filtered_artifacts,
        baseline_combo_id=baseline_combo_id,
    )
    summary_rows = baseline_challenger_rows(
        records=filtered_artifacts,
        baseline_combo_id=baseline_combo_id,
    )
    meaningful_only = st.checkbox(
        "Show only meaningful deltas",
        value=True,
        key="baseline_meaningful_only",
    )
    shown_rows = [
        row for row in comparison_rows
        if (row.get("meaningful_delta") or not meaningful_only)
    ]

    st.dataframe(
        summary_rows,
        use_container_width=True,
        hide_index=True,
    )

    if not shown_rows:
        st.info("No baseline comparison rows match the current filters.")
        return summary_rows, comparison_rows, baseline_combo_id

    summary_cols = [
        "case_id",
        "challenger_combo_id",
        "challenger_combo_title",
        "score_delta",
        "pass_delta",
        "full_text_similarity",
        "meaningful_delta",
    ]
    st.dataframe(
        [{key: row.get(key) for key in summary_cols} for row in shown_rows],
        use_container_width=True,
        hide_index=True,
    )

    detail_map = {
        (
            f"{row.get('case_id')} | {row.get('challenger_combo_id')} | "
            f"score_delta={row.get('score_delta')}"
        ): row
        for row in shown_rows
    }
    detail_label = st.selectbox(
        "Detailed Baseline Comparison",
        options=list(detail_map.keys()),
        key="baseline_detail_row",
    )
    detail_row = detail_map[detail_label]
    st.markdown("### Changed Fields")
    st.write(detail_row.get("field_summary", {}).get("changed_fields", []))
    st.code(detail_row.get("full_text_diff", ""), language="diff")

    paragraph_tab, sentence_tab = st.tabs(
        ["Paragraph Alignment", "Sentence Alignment"]
    )
    with paragraph_tab:
        _render_alignment_rows(detail_row.get("paragraph_alignment", {}))
    with sentence_tab:
        _render_alignment_rows(detail_row.get("sentence_alignment", {}))

    return summary_rows, comparison_rows, baseline_combo_id


def _render_similarity_tab(
    garden_root: str,
    scope: str,
    filtered_artifacts: list[dict[str, Any]],
    embedding_cache_dir: str,
) -> dict[str, Any] | None:
    st.subheader("Similarity And Outlier Review")
    if len(filtered_artifacts) < 2:
        st.info("At least two filtered artifacts are needed for similarity review.")
        return None

    field_path = st.selectbox(
        "Similarity Field",
        options=["comparison_text", "short_answer", "explanation", "experiment.reason"],
        index=0,
        key="similarity_field_path",
    )
    same_case_only = st.checkbox(
        "Limit to same-case comparisons",
        value=True,
        key="similarity_same_case_only",
    )
    duplicate_threshold = st.slider(
        "Near-Duplicate Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.92,
        step=0.01,
        key="similarity_threshold",
    )

    cache_dir = Path(embedding_cache_dir)
    cached_payload = load_similarity_cache(
        cache_dir=cache_dir,
        scope=scope,
        field_path=field_path,
        backend=DEFAULT_EMBEDDING_BACKEND,
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        same_case_only=same_case_only,
        duplicate_threshold=duplicate_threshold,
    )
    use_cache = cached_payload is not None and st.checkbox(
        "Use cached similarity bundle when available",
        value=True,
        key="similarity_use_cache",
    )

    if use_cache and cached_payload is not None:
        bundle = cached_payload.get("bundle") or {}
        cache_origin = "cache"
    else:
        bundle = similarity_bundle(
            records=filtered_artifacts,
            field_path=field_path,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            same_case_only=same_case_only,
            latest_only=False,
            duplicate_threshold=duplicate_threshold,
        )
        cache_origin = "computed"

    if st.button("Save Similarity Bundle To Cache", key="save_similarity_cache"):
        cache_path = write_similarity_cache(
            cache_dir=cache_dir,
            scope=scope,
            bundle=bundle,
            field_path=field_path,
            backend=DEFAULT_EMBEDDING_BACKEND,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            same_case_only=same_case_only,
            duplicate_threshold=duplicate_threshold,
        )
        st.success(f"Saved similarity cache to {cache_path}")

    st.caption(
        f"backend={bundle.get('embedding_backend', DEFAULT_EMBEDDING_BACKEND)} | "
        f"field={field_path} | source={cache_origin}"
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Record Count", bundle.get("record_count"))
    metric_cols[1].metric("Pair Count", bundle.get("pair_count"))
    metric_cols[2].metric("Near-Duplicate Clusters", len(bundle.get("near_duplicate_clusters", [])))

    pair_tab, cluster_tab, outlier_tab = st.tabs(
        ["Pairwise Similarity", "Near-Duplicate Clusters", "Outliers"]
    )
    with pair_tab:
        st.dataframe(
            bundle.get("pairwise_rows", []),
            use_container_width=True,
            hide_index=True,
        )
    with cluster_tab:
        st.dataframe(
            bundle.get("near_duplicate_clusters", []),
            use_container_width=True,
            hide_index=True,
        )
    with outlier_tab:
        st.dataframe(
            bundle.get("outlier_rows", []),
            use_container_width=True,
            hide_index=True,
        )

    return bundle


def _render_notes_and_export_tab(
    garden_root: str,
    scope: str,
    filtered_rows: list[dict[str, Any]],
    combo_rankings: list[dict[str, Any]],
    baseline_summary_rows: list[dict[str, Any]],
    baseline_comparison_rows: list[dict[str, Any]],
    similarity_data: dict[str, Any] | None,
) -> None:
    st.subheader("Reviewer Notes And Export")
    notes_payload = _load_review_notes(garden_root, scope)
    notes_rows = _note_rows(notes_payload)

    if filtered_rows:
        note_case_id = st.selectbox(
            "Case For Note",
            options=sorted(
                {
                    row.get("case_id")
                    for row in filtered_rows
                    if row.get("case_id")
                }
            ),
            key="note_case_id",
        )
        case_rows = [
            row for row in filtered_rows
            if row.get("case_id") == note_case_id
        ]
        note_row_map = {
            _row_label(row): row
            for row in case_rows
        }
        note_row_label = st.selectbox(
            "Combo For Note",
            options=list(note_row_map.keys()),
            key="note_row_label",
        )
        selected_row = note_row_map[note_row_label]
        note_key = _note_key(selected_row["row_id"])
        current_entry = (notes_payload.get("entries") or {}).get(note_key, {})

        reviewer_name = st.text_input(
            "Reviewer Name",
            value=current_entry.get("reviewer", ""),
            key="note_reviewer",
        )
        preferred = st.checkbox(
            "Mark this answer as preferred",
            value=bool(current_entry.get("preferred")),
            key="note_preferred",
        )
        note_text = st.text_area(
            "Review Note",
            value=current_entry.get("note_text", ""),
            height=180,
            key="note_text",
        )
        if st.button("Save Review Note", key="save_review_note"):
            saved_path = _upsert_review_note(
                garden_root=garden_root,
                scope=scope,
                row_id=selected_row["row_id"],
                case_id=selected_row["case_id"],
                combo_id=selected_row["combo_id"],
                combo_title=selected_row.get("combo_title", ""),
                execution_signature=selected_row.get("execution_signature", ""),
                model=selected_row.get("model", ""),
                fewshot_id=selected_row.get("fewshot_id"),
                note_text=note_text,
                preferred=preferred,
                reviewer=reviewer_name,
            )
            st.success(f"Saved note to {saved_path}")
            st.rerun()

    st.markdown("### Saved Notes")
    st.dataframe(
        notes_rows,
        use_container_width=True,
        hide_index=True,
    )

    export_bundle = {
        "scope": scope,
        "created_at": _now_iso(),
        "filtered_review_rows": filtered_rows,
        "combo_rankings": combo_rankings,
        "baseline_summary_rows": baseline_summary_rows,
        "baseline_comparison_rows": baseline_comparison_rows,
        "review_notes": notes_rows,
        "similarity_data": similarity_data,
    }

    st.markdown("### Export")
    export_cols = st.columns(4)
    export_cols[0].download_button(
        "Review Rows JSON",
        data=_stable_json(filtered_rows),
        file_name=f"{scope}__filtered_review_rows.json",
        mime="application/json",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Review Rows CSV",
        data=_rows_to_csv(filtered_rows),
        file_name=f"{scope}__filtered_review_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[2].download_button(
        "Combo Ranking CSV",
        data=_rows_to_csv(combo_rankings),
        file_name=f"{scope}__combo_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[3].download_button(
        "Compact Review Bundle",
        data=_stable_json(export_bundle),
        file_name=f"{scope}__review_bundle.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Prompt Garden Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Prompt Garden Review")
    st.caption(
        "Review prompt-combo experiments with filters, side-by-side comparison, "
        "baseline challenger analysis, similarity grouping, and lightweight notes."
    )

    with st.sidebar:
        st.header("Workspace")
        garden_root = st.text_input(
            "Prompt Garden Root",
            value=str(_default_garden_root()),
            help="Path to the Prompt Garden workspace root.",
        )
        if st.button("Reload Cached Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    scopes = discover_review_scopes(garden_root)
    if not scopes:
        st.warning(
            "No normalized Prompt Garden run scopes were found. "
            "Run `scripts/run_prompt_experiment.py` first."
        )
        return

    with st.sidebar:
        st.header("Scope")
        scope_label_map = {
            row["label"]: row["scope"]
            for row in scopes
        }
        selected_scope_label = st.selectbox(
            "Experiment / Scope",
            options=list(scope_label_map.keys()),
        )
        scope = scope_label_map[selected_scope_label]

    bundle = load_scope_bundle(garden_root, scope)
    artifacts = bundle["artifacts"]
    review_rows = bundle["review_rows"]
    if not review_rows:
        st.warning("The selected scope has no review rows.")
        return

    all_signatures = bundle["signatures"]
    default_signatures = all_signatures[-1:] if all_signatures else []
    case_set_options = sorted(
        {
            row.get("case_set_id")
            for row in review_rows
            if row.get("case_set_id")
        }
    )
    combo_options = sorted(
        {
            row.get("combo_id")
            for row in review_rows
            if row.get("combo_id")
        }
    )
    case_options = sorted(
        {
            row.get("case_id")
            for row in review_rows
            if row.get("case_id")
        }
    )
    request_type_options = sorted(
        {
            row.get("request_type")
            for row in review_rows
            if row.get("request_type")
        }
    )
    experiment_kind_options = sorted(
        {
            row.get("experiment_kind")
            for row in review_rows
            if row.get("experiment_kind")
        }
    )
    score_min_bound, score_max_bound = _score_bounds(review_rows)

    with st.sidebar:
        st.header("Filters")
        selected_signatures = st.multiselect(
            "Execution Signatures",
            options=all_signatures,
            default=default_signatures,
            format_func=_signature_label,
        )
        selected_case_sets = st.multiselect(
            "Case Sets",
            options=case_set_options,
        )
        selected_combo_ids = st.multiselect(
            "Combo IDs",
            options=combo_options,
        )
        selected_case_ids = st.multiselect(
            "Case IDs",
            options=case_options,
        )
        selected_request_types = st.multiselect(
            "Request Types",
            options=request_type_options,
        )
        selected_experiment_kinds = st.multiselect(
            "Experiment Kinds",
            options=experiment_kind_options,
        )
        pass_status = st.selectbox(
            "Pass Status",
            options=["all", "passed", "failed"],
            index=0,
        )
        parse_status = st.selectbox(
            "Parse Status",
            options=["all", "parsed_ok", "parse_error"],
            index=0,
        )
        if score_min_bound < score_max_bound:
            selected_score_range = st.slider(
                "Score Range",
                min_value=float(score_min_bound),
                max_value=float(score_max_bound),
                value=(float(score_min_bound), float(score_max_bound)),
                step=0.01,
            )
        else:
            st.caption(f"Score Range: {score_min_bound}")
            selected_score_range = (
                float(score_min_bound),
                float(score_max_bound),
            )
        latest_only = st.checkbox(
            "Keep only latest artifact per (case, combo)",
            value=True,
        )
        search_text = st.text_input(
            "Search",
            help="Search in question, short answer, combo title, combo id, or case id.",
        )

    filtered_artifacts, filtered_rows = _filter_review_data(
        artifacts=artifacts,
        review_rows=review_rows,
        selected_signatures=selected_signatures,
        selected_case_sets=selected_case_sets,
        selected_combo_ids=selected_combo_ids,
        selected_case_ids=selected_case_ids,
        selected_request_types=selected_request_types,
        selected_experiment_kinds=selected_experiment_kinds,
        pass_status=pass_status,
        parse_status=parse_status,
        score_min=selected_score_range[0],
        score_max=selected_score_range[1],
        search_text=search_text,
        latest_only=latest_only,
    )

    if not filtered_rows:
        st.warning("No review rows match the current filters.")
        return

    filtered_metrics = summary_metrics(filtered_rows)
    st.markdown(
        f"**Scope:** `{scope}`  "
        f"| **Rows:** `{len(filtered_rows)}`  "
        f"| **Combos:** `{filtered_metrics.get('combo_count')}`  "
        f"| **Cases:** `{filtered_metrics.get('case_count')}`"
    )

    overview_tab, compare_tab, baseline_tab, similarity_tab, notes_tab = st.tabs(
        ["Overview", "Compare", "Baseline", "Similarity", "Notes & Export"]
    )

    with overview_tab:
        combo_rankings, _case_difficulty, _filter_metrics = _render_overview_tab(
            filtered_artifacts=filtered_artifacts,
            filtered_rows=filtered_rows,
            filtered_metrics=filtered_metrics,
            experiment=bundle["experiment"],
            report_files=bundle["report_files"],
        )

    with compare_tab:
        _render_compare_tab(filtered_artifacts)

    with baseline_tab:
        baseline_summary_rows_data, baseline_comparison_rows_data, _baseline_combo_id = _render_baseline_tab(
            filtered_artifacts=filtered_artifacts,
            filtered_rows=filtered_rows,
        )

    with similarity_tab:
        similarity_data = _render_similarity_tab(
            garden_root=garden_root,
            scope=scope,
            filtered_artifacts=filtered_artifacts,
            embedding_cache_dir=bundle["embedding_cache_dir"],
        )

    with notes_tab:
        _render_notes_and_export_tab(
            garden_root=garden_root,
            scope=scope,
            filtered_rows=filtered_rows,
            combo_rankings=combo_rankings,
            baseline_summary_rows=baseline_summary_rows_data,
            baseline_comparison_rows=baseline_comparison_rows_data,
            similarity_data=similarity_data,
        )


if __name__ == "__main__":
    main()
