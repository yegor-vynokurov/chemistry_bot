"""Analysis-surface rendering for the Prompt Garden Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from .garden import PromptGarden
from .review_app_data import (
    discover_review_scopes,
    filter_review_data,
    load_experiment_composition_bundle,
    load_prompt_similarity_items,
    load_scope_bundle,
)
from .review_app_support import (
    TEXT_COMPARE_FIELDS,
    artifact_label,
    fewshot_label,
    load_review_notes,
    note_key,
    note_rows,
    now_iso,
    row_label,
    rows_to_csv,
    score_bounds,
    short_signature,
    signature_label,
    stable_json,
    upsert_review_note,
)
from .review_compare import (
    DEFAULT_FIELD_PATHS,
    build_baseline_vs_challenger,
    compare_record_fields,
    field_display_text,
    field_segment_alignment,
    html_text_diff,
    unified_text_diff,
)
from .review_embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_DIMENSIONS,
    load_similarity_cache,
    nearest_neighbor_rows_for_text_items,
    similarity_bundle,
    text_item_similarity_bundle,
    write_similarity_cache,
)
from .review_metrics import (
    baseline_challenger_rows,
    case_difficulty_rows,
    combo_ranking_rows,
    review_filter_metrics,
)
from .review_store import summary_metrics


_ANALYSIS_FLASH_KEY = "prompt_garden_analysis_flash_message"


def _queue_analysis_refresh(message: str) -> None:
    st.session_state[_ANALYSIS_FLASH_KEY] = message
    st.cache_data.clear()
    st.rerun()


def _consume_analysis_flash_message() -> None:
    message = st.session_state.pop(_ANALYSIS_FLASH_KEY, None)
    if message:
        st.success(message)


def build_scope_selector_maps(
    scopes: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build stable scope selectors for id-first and name-first browsing."""

    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}

    for row in scopes:
        scope = str(row.get("scope") or "")
        status = row.get("experiment_status") or "-"
        artifact_count = int(row.get("artifact_count", 0))
        experiment_name = row.get("experiment_name")

        by_id[
            f"{scope} | status={status} | artifacts={artifact_count}"
        ] = scope

        if experiment_name:
            by_name[
                f"{experiment_name} | {scope} | status={status} | artifacts={artifact_count}"
            ] = scope
        else:
            by_name[
                f"{scope} | ad hoc scope | artifacts={artifact_count}"
            ] = scope

    return {
        "by_id": by_id,
        "by_name": by_name,
    }


def build_review_score_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize score range and row counts for the current analysis selection."""

    scores = [
        float(row["score"])
        for row in rows
        if row.get("score") is not None
    ]
    return {
        "row_count": len(rows),
        "scored_row_count": len(scores),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "average_score": (
            sum(scores) / len(scores)
            if scores else None
        ),
    }


def render_metric_row(metrics: dict[str, Any]) -> None:
    """Render top-level summary metrics for filtered review rows."""

    columns = st.columns(5)
    columns[0].metric("Rows", metrics.get("review_row_count", 0))
    columns[1].metric("Combos", metrics.get("combo_count", 0))
    columns[2].metric("Cases", metrics.get("case_count", 0))
    columns[3].metric("Avg Score", metrics.get("average_score"))
    columns[4].metric("Pass Rate", metrics.get("pass_rate"))


def render_answer_column(
    artifact: dict[str, Any],
    title: str,
    *,
    show_artifact_paths: bool = False,
) -> None:
    """Render one answer detail column for side-by-side inspection."""

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
        f"fewshot={fewshot_label(execution.get('fewshot_id'))} | "
        f"sig={short_signature(execution.get('signature'))} | "
        f"score={metrics.get('score')} | "
        f"passed={metrics.get('passed')}"
    )
    st.markdown("**Question**")
    st.write(artifact.get("question") or "(empty)")
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
    if show_artifact_paths:
        with st.expander("Artifact Files"):
            st.json(artifact.get("artifact_paths") or {})


def render_alignment_rows(
    alignment_payload: dict[str, Any],
) -> None:
    """Render paragraph or sentence alignment rows in a table."""

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


def render_overview_tab(
    filtered_artifacts: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    filtered_metrics: dict[str, Any],
    experiment: dict[str, Any] | None,
    report_files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Render overview tables and aggregate metrics."""

    st.subheader("Overview")
    render_metric_row(filtered_metrics)

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


def render_experiment_summary_panel(
    scope: str,
    experiment: dict[str, Any] | None,
    filtered_rows: list[dict[str, Any]],
    filtered_metrics: dict[str, Any],
    composition_bundle: dict[str, Any] | None,
    report_files: list[str],
) -> None:
    """Render the main experiment-oriented summary block above analysis tabs."""

    score_summary = build_review_score_summary(filtered_rows)
    scope_line = f"**Scope:** `{scope}`"
    if experiment:
        scope_line += (
            f"  | **Experiment:** `{experiment.get('id')}`"
            f"  | **Name:** `{experiment.get('name')}`"
            f"  | **Status:** `{experiment.get('status')}`"
        )
    st.markdown(scope_line)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rows", filtered_metrics.get("review_row_count", 0))
    metric_cols[1].metric("Combos", filtered_metrics.get("combo_count", 0))
    metric_cols[2].metric("Cases", filtered_metrics.get("case_count", 0))
    metric_cols[3].metric("Avg Score", filtered_metrics.get("average_score"))
    score_range_value = (
        "-"
        if score_summary["min_score"] is None
        else (
            f"{score_summary['min_score']:.3f} - "
            f"{score_summary['max_score']:.3f}"
        )
    )
    metric_cols[4].metric("Score Range", score_range_value)
    metric_cols[5].metric("Pass Rate", filtered_metrics.get("pass_rate"))

    if experiment:
        summary = (
            (composition_bundle or {}).get("summary")
            or {}
        )
        overview_cols = st.columns(5)
        overview_cols[0].metric(
            "Attached Combos",
            summary.get("combo_count", len(experiment.get("combo_ids", []))),
        )
        overview_cols[1].metric(
            "Tested Combos",
            summary.get("tested_combo_count", 0),
        )
        overview_cols[2].metric(
            "Untested Combos",
            summary.get("untested_combo_count", 0),
        )
        overview_cols[3].metric(
            "Missing Combos",
            summary.get("missing_combo_count", 0),
        )
        overview_cols[4].metric(
            "Reports",
            len(report_files),
        )

        with st.expander("Experiment Summary", expanded=False):
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("**Goal**")
                st.write(experiment.get("goal") or "(empty)")
                st.markdown("**Hypothesis**")
                st.write(experiment.get("hypothesis") or "(empty)")
            with right_col:
                st.markdown("**Notes**")
                st.write(experiment.get("notes") or "(empty)")
                st.markdown("**Tags**")
                st.write(", ".join(experiment.get("tags", [])) or "(none)")
            st.markdown("**Final Result Text**")
            st.write(experiment.get("final_result_text") or "(not finalized)")
            st.markdown("**Final Subject Score**")
            st.write(experiment.get("final_subject_score"))

        if composition_bundle:
            combo_rows = composition_bundle.get("combo_rows") or []
            with st.expander("Attached Combo Overview", expanded=False):
                st.dataframe(
                    [
                        {
                            "position": row.get("position"),
                            "combo_id": row.get("combo_id"),
                            "combo_title": row.get("combo_title"),
                            "combo_status": row.get("combo_status"),
                            "test_status": row.get("combo_test_status"),
                            "result_score": row.get("result_score"),
                            "prompt_count": row.get("prompt_count"),
                            "missing_prompt_ids": ", ".join(
                                row.get("missing_prompt_ids", [])
                            ),
                        }
                        for row in combo_rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    if report_files:
        with st.expander("Existing Derived Reports", expanded=False):
            st.write(report_files)


def render_answer_browser_tab(
    filtered_artifacts: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
) -> None:
    """Render a manual answer-by-answer browser for one filtered scope."""

    st.subheader("Answer Browser")
    if not filtered_rows:
        st.info("No answers match the current filters.")
        return

    artifact_by_row_id = {
        str(artifact.get("id")): artifact
        for artifact in filtered_artifacts
    }
    browser_col, detail_col = st.columns([1, 2], gap="large")

    with browser_col:
        browse_mode = st.radio(
            "Browse By",
            options=["Case", "Combo"],
            horizontal=True,
            key="analysis_browse_mode",
        )
        if browse_mode == "Case":
            case_options = sorted(
                {
                    row.get("case_id")
                    for row in filtered_rows
                    if row.get("case_id")
                }
            )
            selected_case_id = st.selectbox(
                "Case",
                options=case_options,
                key="analysis_browser_case_id",
            )
            candidate_rows = [
                row for row in filtered_rows
                if row.get("case_id") == selected_case_id
            ]
        else:
            combo_options = sorted(
                {
                    (row.get("combo_id"), row.get("combo_title"))
                    for row in filtered_rows
                    if row.get("combo_id")
                }
            )
            combo_label_map = {
                f"{combo_id} | {combo_title}": combo_id
                for combo_id, combo_title in combo_options
            }
            selected_combo_label = st.selectbox(
                "Combo",
                options=list(combo_label_map.keys()),
                key="analysis_browser_combo_id",
            )
            selected_combo_id = combo_label_map[selected_combo_label]
            candidate_rows = [
                row for row in filtered_rows
                if row.get("combo_id") == selected_combo_id
            ]

        st.caption(f"Candidate answers: {len(candidate_rows)}")
        st.dataframe(
            [
                {
                    "case_id": row.get("case_id"),
                    "combo_id": row.get("combo_id"),
                    "model": row.get("model"),
                    "fewshot_id": row.get("fewshot_id"),
                    "score": row.get("score"),
                    "passed": row.get("passed"),
                    "request_type": row.get("request_type"),
                }
                for row in candidate_rows
            ],
            use_container_width=True,
            hide_index=True,
        )

        answer_label_map = {
            row_label(row): row
            for row in candidate_rows
        }
        selected_answer_label = st.selectbox(
            "Answer Detail",
            options=list(answer_label_map.keys()),
            key="analysis_browser_answer_row",
        )
        selected_row = answer_label_map[selected_answer_label]

    with detail_col:
        selected_artifact = artifact_by_row_id.get(
            str(selected_row.get("row_id"))
        )
        if selected_artifact is None:
            st.warning("The selected answer artifact could not be loaded.")
            return

        render_answer_column(
            selected_artifact,
            "Selected Answer",
            show_artifact_paths=True,
        )

        meta_left, meta_right = st.columns(2)
        with meta_left:
            st.markdown("**Answer Metadata**")
            st.json({
                "row_id": selected_row.get("row_id"),
                "raw_run_id": selected_row.get("raw_run_id"),
                "combo_id": selected_row.get("combo_id"),
                "combo_title": selected_row.get("combo_title"),
                "model": selected_row.get("model"),
                "fewshot_id": selected_row.get("fewshot_id"),
                "score": selected_row.get("score"),
                "passed": selected_row.get("passed"),
                "duration_seconds": selected_row.get("duration_seconds"),
            })
        with meta_right:
            st.markdown("**Prompt References**")
            st.json({
                "system_prompt_id": selected_row.get("system_prompt_id"),
                "user_prompt_id": selected_row.get("user_prompt_id"),
                "fewshot_id": selected_row.get("fewshot_id"),
                "source_ids": selected_row.get("source_ids", []),
                "example_count": selected_row.get("example_count"),
            })

        sibling_rows = [
            row for row in filtered_rows
            if row.get("case_id") == selected_row.get("case_id")
            and row.get("row_id") != selected_row.get("row_id")
        ]
        st.markdown("### Other Answers For This Case")
        if sibling_rows:
            st.dataframe(
                [
                    {
                        "combo_id": row.get("combo_id"),
                        "combo_title": row.get("combo_title"),
                        "model": row.get("model"),
                        "fewshot_id": row.get("fewshot_id"),
                        "score": row.get("score"),
                        "passed": row.get("passed"),
                        "short_answer": row.get("short_answer"),
                    }
                    for row in sibling_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No other filtered answers exist for this case.")


def render_compare_tab(
    filtered_artifacts: list[dict[str, Any]],
) -> None:
    """Render the side-by-side answer comparison surface."""

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
        artifact_label(artifact): artifact
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
        render_answer_column(left_artifact, "Left")
    with right_col:
        render_answer_column(right_artifact, "Right")

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
        render_alignment_rows(paragraph_alignment)
    with sentence_tab:
        render_alignment_rows(sentence_alignment)


def render_baseline_tab(
    filtered_artifacts: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Render baseline-vs-challenger aggregate comparisons."""

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
        render_alignment_rows(detail_row.get("paragraph_alignment", {}))
    with sentence_tab:
        render_alignment_rows(detail_row.get("sentence_alignment", {}))

    return summary_rows, comparison_rows, baseline_combo_id


def render_similarity_tab(
    garden_root: str,
    scope: str,
    filtered_artifacts: list[dict[str, Any]],
    embedding_cache_dir: str,
) -> dict[str, Any] | None:
    """Render similarity tables and optional cache actions."""

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
        item_kind="review_records",
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
            item_kind="review_records",
        )
        st.success(f"Saved similarity cache to {cache_path}")

    st.caption(
        f"backend={bundle.get('embedding_backend', DEFAULT_EMBEDDING_BACKEND)} | "
        f"field={field_path} | source={cache_origin}"
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Record Count", bundle.get("record_count"))
    metric_cols[1].metric("Pair Count", bundle.get("pair_count"))
    metric_cols[2].metric(
        "Near-Duplicate Clusters",
        len(bundle.get("near_duplicate_clusters", [])),
    )

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


def _matches_prompt_similarity_search(
    prompt_item: dict[str, Any],
    query: str,
) -> bool:
    if not query:
        return True
    haystack = " ".join([
        str(prompt_item.get("id", "")),
        str(prompt_item.get("title", "")),
        str(prompt_item.get("type", "")),
        str(prompt_item.get("tree_id", "")),
        str(prompt_item.get("branch", "")),
        str(prompt_item.get("path", "")),
        " ".join(str(tag) for tag in prompt_item.get("tags", [])),
        str(prompt_item.get("text_preview", "")),
    ]).lower()
    return query in haystack


def _prompt_similarity_text(
    prompt_item: dict[str, Any],
    mode: str,
) -> str:
    prompt_text = prompt_item.get("text") or ""
    if mode == "Prompt Text":
        return prompt_text
    if mode == "Prompt Text + Title":
        return (
            f"TITLE\n{prompt_item.get('title') or ''}\n\n"
            f"PROMPT\n{prompt_text}"
        ).strip()
    if mode == "Prompt Text + Metadata":
        metadata_lines = [
            f"id={prompt_item.get('id')}",
            f"title={prompt_item.get('title') or ''}",
            f"type={prompt_item.get('type') or ''}",
            f"tree_id={prompt_item.get('tree_id') or ''}",
            f"branch={prompt_item.get('branch') or ''}",
            "tags=" + ", ".join(prompt_item.get("tags", [])),
        ]
        return (
            "METADATA\n"
            + "\n".join(metadata_lines)
            + "\n\nPROMPT\n"
            + prompt_text
        ).strip()
    raise ValueError(f"Unsupported prompt similarity mode: {mode}")


def render_prompt_similarity_tab(
    garden_root: str,
    embedding_cache_dir: str,
) -> dict[str, Any] | None:
    """Render prompt-to-prompt similarity review for one workspace."""

    st.subheader("Prompt Similarity")
    prompt_items = load_prompt_similarity_items(garden_root)
    if len(prompt_items) < 2:
        st.info("At least two prompts are needed for prompt similarity review.")
        return None

    similarity_modes = [
        "Prompt Text",
        "Prompt Text + Title",
        "Prompt Text + Metadata",
    ]
    type_options = sorted(
        {
            item.get("type")
            for item in prompt_items
            if item.get("type")
        }
    )
    tree_id_options = sorted(
        {
            item.get("tree_id")
            for item in prompt_items
            if item.get("tree_id")
        }
    )
    branch_options = sorted(
        {
            item.get("branch")
            for item in prompt_items
            if item.get("branch")
        }
    )
    tag_options = sorted(
        {
            tag
            for item in prompt_items
            for tag in item.get("tags", [])
        }
    )

    controls_col, detail_col = st.columns([1, 2], gap="large")

    with controls_col:
        search_query = st.text_input(
            "Search Prompt",
            help="Search by id, title, type, tree, branch, tags, or prompt preview.",
            key="prompt_similarity_search",
        ).strip().lower()
        selected_types = st.multiselect(
            "Prompt Types",
            options=type_options,
            key="prompt_similarity_types",
        )
        selected_tree_ids = st.multiselect(
            "Tree IDs",
            options=tree_id_options,
            key="prompt_similarity_tree_ids",
        )
        selected_branches = st.multiselect(
            "Branches",
            options=branch_options,
            key="prompt_similarity_branches",
        )
        selected_tags = st.multiselect(
            "Tags",
            options=tag_options,
            key="prompt_similarity_tags",
        )
        include_archived = st.checkbox(
            "Include Archived Prompts",
            value=False,
            key="prompt_similarity_include_archived",
        )
        similarity_mode = st.selectbox(
            "Similarity Text",
            options=similarity_modes,
            key="prompt_similarity_mode",
        )
        same_tree_only = st.checkbox(
            "Limit To Same Tree",
            value=False,
            key="prompt_similarity_same_tree_only",
        )
        duplicate_threshold = st.slider(
            "Near-Duplicate Threshold",
            min_value=0.5,
            max_value=0.99,
            value=0.92,
            step=0.01,
            key="prompt_similarity_threshold",
        )

        filtered_prompt_items = []
        for item in prompt_items:
            if selected_types and item.get("type") not in selected_types:
                continue
            if selected_tree_ids and item.get("tree_id") not in selected_tree_ids:
                continue
            if selected_branches and item.get("branch") not in selected_branches:
                continue
            if not include_archived and item.get("is_archived"):
                continue
            if selected_tags and not set(selected_tags).issubset(
                set(item.get("tags", []))
            ):
                continue
            if not _matches_prompt_similarity_search(item, search_query):
                continue
            filtered_prompt_items.append(item)

        st.caption(f"Prompt candidates: {len(filtered_prompt_items)}")
        st.dataframe(
            [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "type": item.get("type"),
                    "tree_id": item.get("tree_id"),
                    "branch": item.get("branch"),
                    "combo_count": item.get("combo_count"),
                    "experiment_count": item.get("experiment_count"),
                    "is_archived": item.get("is_archived"),
                }
                for item in filtered_prompt_items[:20]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered_prompt_items:
            selected_prompt_id = None
        else:
            prompt_label_map = {
                item.get("label") or item.get("id"): str(item.get("item_id"))
                for item in filtered_prompt_items
            }
            selected_label = st.selectbox(
                "Selected Prompt",
                options=list(prompt_label_map.keys()),
                key="prompt_similarity_selected_prompt",
            )
            selected_prompt_id = prompt_label_map[selected_label]

    with detail_col:
        if len(filtered_prompt_items) < 2:
            st.info(
                "At least two prompts are needed after filtering to compute prompt similarity."
            )
            return None

        similarity_field = (
            similarity_mode.lower()
            .replace(" + ", "_plus_")
            .replace(" ", "_")
        )
        similarity_items = [
            {
                **item,
                "text": _prompt_similarity_text(item, similarity_mode),
            }
            for item in filtered_prompt_items
        ]
        selection_scope_hash = PromptGarden._hash_data(
            sorted(item["item_id"] for item in similarity_items)
        )[:12]
        similarity_scope = (
            f"prompt_similarity__{Path(garden_root).name}__{selection_scope_hash}"
        )
        cache_dir = Path(embedding_cache_dir)
        cached_payload = load_similarity_cache(
            cache_dir=cache_dir,
            scope=similarity_scope,
            field_path=similarity_field,
            backend=DEFAULT_EMBEDDING_BACKEND,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            same_case_only=same_tree_only,
            duplicate_threshold=duplicate_threshold,
            item_kind="prompt_items",
        )
        use_cache = cached_payload is not None and st.checkbox(
            "Use cached prompt similarity bundle when available",
            value=True,
            key="prompt_similarity_use_cache",
        )

        if use_cache and cached_payload is not None:
            bundle = cached_payload.get("bundle") or {}
            cache_origin = "cache"
        else:
            bundle = text_item_similarity_bundle(
                similarity_items,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                same_group_only=same_tree_only,
                duplicate_threshold=duplicate_threshold,
                item_kind="prompt_items",
            )
            cache_origin = "computed"

        if st.button(
            "Save Prompt Similarity Bundle To Cache",
            key="save_prompt_similarity_cache",
        ):
            cache_path = write_similarity_cache(
                cache_dir=cache_dir,
                scope=similarity_scope,
                bundle=bundle,
                field_path=similarity_field,
                backend=DEFAULT_EMBEDDING_BACKEND,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                same_case_only=same_tree_only,
                duplicate_threshold=duplicate_threshold,
                item_kind="prompt_items",
            )
            st.success(f"Saved prompt similarity cache to {cache_path}")

        st.caption(
            f"backend={bundle.get('embedding_backend', DEFAULT_EMBEDDING_BACKEND)} | "
            f"mode={similarity_mode} | source={cache_origin} | "
            f"same_tree_only={same_tree_only}"
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Prompt Count", bundle.get("item_count"))
        metric_cols[1].metric("Pair Count", bundle.get("pair_count"))
        metric_cols[2].metric(
            "Near-Duplicate Clusters",
            len(bundle.get("near_duplicate_clusters", [])),
        )
        metric_cols[3].metric(
            "Trees Represented",
            len(
                {
                    item.get("tree_id")
                    for item in filtered_prompt_items
                    if item.get("tree_id")
                }
            ),
        )

        selected_prompt_item = next(
            (
                item for item in similarity_items
                if item["item_id"] == selected_prompt_id
            ),
            similarity_items[0],
        )
        st.markdown("### Selected Prompt")
        st.json({
            "id": selected_prompt_item.get("id"),
            "title": selected_prompt_item.get("title"),
            "type": selected_prompt_item.get("type"),
            "tree_id": selected_prompt_item.get("tree_id"),
            "branch": selected_prompt_item.get("branch"),
            "tags": selected_prompt_item.get("tags", []),
            "combo_count": selected_prompt_item.get("combo_count"),
            "experiment_count": selected_prompt_item.get("experiment_count"),
            "child_count": selected_prompt_item.get("child_count"),
            "is_archived": selected_prompt_item.get("is_archived"),
        })
        with st.expander("Selected Prompt Text", expanded=False):
            st.code(
                selected_prompt_item.get("text") or "",
                language="markdown",
            )

        st.markdown("### Nearest Prompt Neighbors")
        nearest_rows = nearest_neighbor_rows_for_text_items(
            similarity_items,
            selected_item_id=str(selected_prompt_item["item_id"]),
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            same_group_only=same_tree_only,
            top_k=10,
            item_kind="prompt_items",
        )
        if nearest_rows:
            neighbor_lookup = {
                str(item["item_id"]): item
                for item in similarity_items
            }
            st.dataframe(
                [
                    {
                        "neighbor_item_id": row.get("neighbor_item_id"),
                        "neighbor_label": row.get("neighbor_label"),
                        "neighbor_tree_id": (
                            neighbor_lookup.get(
                                str(row.get("neighbor_item_id")),
                                {},
                            ).get("tree_id")
                        ),
                        "neighbor_branch": (
                            neighbor_lookup.get(
                                str(row.get("neighbor_item_id")),
                                {},
                            ).get("branch")
                        ),
                        "similarity": row.get("similarity"),
                    }
                    for row in nearest_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No prompt neighbors are available under the current filters.")

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


def render_experiment_notes_tab(
    garden_root: str,
    experiment: dict[str, Any] | None,
) -> None:
    """Render experiment-level analysis notes and finalization controls."""

    st.subheader("Experiment Notes And Finalization")
    if experiment is None:
        st.info(
            "The current scope is not linked to a saved experiment object, "
            "so experiment-level notes and finalization are unavailable."
        )
        return

    experiment_id = str(experiment.get("id") or "")
    summary = experiment.get("summary") or {}
    subjective_summary = experiment.get("subjective_summary") or {}
    result_rows = list(experiment.get("results", []))

    st.caption(
        f"{experiment_id} | status={experiment.get('status')} | "
        f"tested={summary.get('tested_combo_count', 0)} / "
        f"attached={summary.get('attached_combo_count', len(experiment.get('combo_ids', [])))}"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Tested Combos",
        summary.get("tested_combo_count", 0),
    )
    metric_cols[1].metric(
        "Average Score",
        summary.get("average_score"),
    )
    metric_cols[2].metric(
        "Avg Subject Score",
        subjective_summary.get("average_subject_score"),
    )
    metric_cols[3].metric(
        "Final Subject Score",
        experiment.get("final_subject_score"),
    )

    notes_col, finalize_col = st.columns(2, gap="large")

    with notes_col:
        st.markdown("**Experiment Notes**")
        with st.form(f"analysis_experiment_notes_form_{experiment_id}"):
            experiment_notes = st.text_area(
                "Notes",
                value=experiment.get("notes") or "",
                height=260,
                help="Use this for experiment-level interpretation, follow-up ideas, or rerun decisions.",
            )
            notes_submitted = st.form_submit_button(
                "Save Experiment Notes",
                use_container_width=True,
            )

        if notes_submitted:
            garden = PromptGarden(garden_root)
            garden.init()
            try:
                garden.update_experiment_metadata(
                    experiment_id,
                    notes=experiment_notes,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                _queue_analysis_refresh(
                    f"Experiment notes saved for `{experiment_id}`."
                )

    with finalize_col:
        st.markdown("**Final Summary**")
        existing_final_subject_score = experiment.get("final_subject_score")
        with st.form(f"analysis_experiment_finalize_form_{experiment_id}"):
            final_result_text = st.text_area(
                "Final Result Text",
                value=experiment.get("final_result_text") or "",
                height=260,
                help="Write the concluding interpretation that should stay with the experiment.",
            )
            use_subject_score = st.checkbox(
                "Set Final Subject Score",
                value=existing_final_subject_score is not None,
            )
            final_subject_score = st.number_input(
                "Final Subject Score",
                min_value=0.0,
                max_value=1.0,
                value=float(existing_final_subject_score or 0.0),
                step=0.01,
                disabled=not use_subject_score,
                help="Optional human judgement score on a 0-1 scale.",
            )
            finalize_label = (
                "Update Final Summary"
                if experiment.get("status") == "completed"
                else "Finalize Experiment"
            )
            finalize_submitted = st.form_submit_button(
                finalize_label,
                use_container_width=True,
            )

        if finalize_submitted:
            if not final_result_text.strip():
                st.error("Final result text is required before finalizing.")
            else:
                garden = PromptGarden(garden_root)
                garden.init()
                try:
                    garden.finalize_experiment(
                        experiment_id,
                        result_text=final_result_text,
                        subject_score=(
                            float(final_subject_score)
                            if use_subject_score else None
                        ),
                        status="completed",
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_analysis_refresh(
                        f"Experiment `{experiment_id}` finalized in the analysis workspace."
                    )

    with st.expander("Current Finalization State", expanded=False):
        st.json({
            "status": experiment.get("status"),
            "updated_at": experiment.get("updated_at"),
            "final_result_text": experiment.get("final_result_text"),
            "final_subject_score": experiment.get("final_subject_score"),
            "subjective_summary": subjective_summary,
        })

    st.markdown("### Stored Combo Result Notes")
    if result_rows:
        st.dataframe(
            [
                {
                    "combo_id": row.get("combo_id"),
                    "score": row.get("score"),
                    "subject_score": row.get("subject_score"),
                    "status": row.get("status"),
                    "created_at": row.get("created_at"),
                    "result_text": row.get("result_text"),
                    "subjective_notes": row.get("subjective_notes"),
                }
                for row in result_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("This experiment does not yet have stored combo-level result notes.")


def render_notes_and_export_tab(
    garden_root: str,
    scope: str,
    filtered_rows: list[dict[str, Any]],
    combo_rankings: list[dict[str, Any]],
    baseline_summary_rows: list[dict[str, Any]],
    baseline_comparison_rows: list[dict[str, Any]],
    similarity_data: dict[str, Any] | None,
    experiment: dict[str, Any] | None,
) -> None:
    """Render review-note editing and export downloads."""

    st.subheader("Reviewer Notes And Export")
    notes_payload = load_review_notes(garden_root, scope)
    notes_rows = note_rows(notes_payload)

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
            row_label(row): row
            for row in case_rows
        }
        note_row_label = st.selectbox(
            "Combo For Note",
            options=list(note_row_map.keys()),
            key="note_row_label",
        )
        selected_row = note_row_map[note_row_label]
        current_entry = (notes_payload.get("entries") or {}).get(
            note_key(selected_row["row_id"]),
            {},
        )

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
            saved_path = upsert_review_note(
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
        "created_at": now_iso(),
        "experiment": (
            {
                "id": experiment.get("id"),
                "name": experiment.get("name"),
                "status": experiment.get("status"),
                "notes": experiment.get("notes"),
                "final_result_text": experiment.get("final_result_text"),
                "final_subject_score": experiment.get("final_subject_score"),
                "summary": experiment.get("summary"),
                "subjective_summary": experiment.get("subjective_summary"),
            }
            if experiment is not None else None
        ),
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
        data=stable_json(filtered_rows),
        file_name=f"{scope}__filtered_review_rows.json",
        mime="application/json",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "Review Rows CSV",
        data=rows_to_csv(filtered_rows),
        file_name=f"{scope}__filtered_review_rows.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[2].download_button(
        "Combo Ranking CSV",
        data=rows_to_csv(combo_rankings),
        file_name=f"{scope}__combo_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[3].download_button(
        "Compact Review Bundle",
        data=stable_json(export_bundle),
        file_name=f"{scope}__review_bundle.json",
        mime="application/json",
        use_container_width=True,
    )


def render_review_analysis_surface(
    garden_root: str,
    scopes: list[dict[str, Any]] | None = None,
) -> None:
    """Render the current analysis-oriented Prompt Garden review surface."""

    if scopes is None:
        scopes = discover_review_scopes(garden_root)
    if not scopes:
        st.warning(
            "No normalized Prompt Garden run scopes were found. "
            "Run `scripts/run_prompt_experiment.py` first."
        )
        return

    st.subheader("Analysis")
    st.caption(
        "Review normalized experiment outputs with experiment-centric scope selection, "
        "manual answer browsing, experiment finalization, comparison tools, similarity, "
        "and lightweight notes."
    )

    controls_col, content_col = st.columns([1, 3], gap="large")

    with controls_col:
        st.markdown("### Scope")
        selector_maps = build_scope_selector_maps(scopes)
        selection_mode = st.radio(
            "Load Scope",
            options=["By Name", "By ID"],
            horizontal=True,
            key="analysis_scope_selection_mode",
        )
        if selection_mode == "By ID":
            scope_label_map = selector_maps["by_id"]
        else:
            scope_label_map = selector_maps["by_name"]
        selected_scope_label = st.selectbox(
            "Experiment / Scope",
            options=list(scope_label_map.keys()),
            key="analysis_selected_scope",
        )
        scope = scope_label_map[selected_scope_label]

    bundle = load_scope_bundle(garden_root, scope)
    artifacts = bundle["artifacts"]
    review_rows = bundle["review_rows"]
    if not review_rows:
        st.warning("The selected scope has no review rows.")
        return
    composition_bundle = None
    experiment = bundle["experiment"]
    if experiment is not None:
        composition_bundle = load_experiment_composition_bundle(
            garden_root,
            experiment["id"],
        )

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
    score_min_bound, score_max_bound = score_bounds(review_rows)

    with controls_col:
        st.markdown("### Filters")
        selected_signatures = st.multiselect(
            "Execution Signatures",
            options=all_signatures,
            default=default_signatures,
            format_func=signature_label,
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

    filtered_artifacts, filtered_rows = filter_review_data(
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
    with content_col:
        render_experiment_summary_panel(
            scope=scope,
            experiment=experiment,
            filtered_rows=filtered_rows,
            filtered_metrics=filtered_metrics,
            composition_bundle=composition_bundle,
            report_files=bundle["report_files"],
        )
        _consume_analysis_flash_message()

        answer_tab, overview_tab, compare_tab, baseline_tab, similarity_tab, prompt_similarity_tab, experiment_notes_tab, notes_tab = st.tabs(
            [
                "Answer Browser",
                "Overview",
                "Compare",
                "Baseline",
                "Similarity",
                "Prompt Similarity",
                "Experiment Notes",
                "Review Notes & Export",
            ]
        )

        with answer_tab:
            render_answer_browser_tab(
                filtered_artifacts=filtered_artifacts,
                filtered_rows=filtered_rows,
            )

        with overview_tab:
            combo_rankings, _case_difficulty, _filter_metrics = render_overview_tab(
                filtered_artifacts=filtered_artifacts,
                filtered_rows=filtered_rows,
                filtered_metrics=filtered_metrics,
                experiment=experiment,
                report_files=bundle["report_files"],
            )

        with compare_tab:
            render_compare_tab(filtered_artifacts)

        with baseline_tab:
            baseline_summary_rows_data, baseline_comparison_rows_data, _baseline_combo_id = render_baseline_tab(
                filtered_artifacts=filtered_artifacts,
                filtered_rows=filtered_rows,
            )

        with similarity_tab:
            similarity_data = render_similarity_tab(
                garden_root=garden_root,
                scope=scope,
                filtered_artifacts=filtered_artifacts,
                embedding_cache_dir=bundle["embedding_cache_dir"],
            )

        with prompt_similarity_tab:
            _prompt_similarity_data = render_prompt_similarity_tab(
                garden_root=garden_root,
                embedding_cache_dir=bundle["embedding_cache_dir"],
            )

        with experiment_notes_tab:
            render_experiment_notes_tab(
                garden_root=garden_root,
                experiment=experiment,
            )

        with notes_tab:
            render_notes_and_export_tab(
                garden_root=garden_root,
                scope=scope,
                filtered_rows=filtered_rows,
                combo_rankings=combo_rankings,
                baseline_summary_rows=baseline_summary_rows_data,
                baseline_comparison_rows=baseline_comparison_rows_data,
                similarity_data=similarity_data,
                experiment=experiment,
            )


__all__ = [
    "render_review_analysis_surface",
]
