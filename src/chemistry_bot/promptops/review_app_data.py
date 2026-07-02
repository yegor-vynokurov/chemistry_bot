"""Cached loaders and filtering helpers for the Prompt Garden review app."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .garden import PromptGarden
from .garden_index import (
    combo_detail_bundle,
    combo_usage_rows,
    experiment_composition,
    experiment_summary_rows,
    prompt_detail_bundle,
    prompt_similarity_items,
    prompt_usage_rows,
)
from .review_compare import latest_records_by_case_combo
from .review_store import (
    build_review_rows,
    case_summary_rows,
    combo_summary_rows,
    load_normalized_scope,
    prompt_combo_performance_rows,
    prompt_model_performance_rows,
    prompt_review_rows,
    summary_metrics,
)


def filter_review_data(
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
    """Apply sidebar filters to artifacts and flattened review rows."""

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
    """Discover normalized review scopes under a Prompt Garden root."""

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
    """Load one review scope together with derived summary tables."""

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


@st.cache_data(show_spinner=False)
def load_all_review_rows_bundle(
    garden_root: str,
) -> dict[str, Any]:
    """Load review rows from every normalized scope under one workspace."""

    scope_rows = discover_review_scopes(garden_root)
    all_review_rows: list[dict[str, Any]] = []

    for scope_row in scope_rows:
        scope = str(scope_row.get("scope") or "")
        if not scope:
            continue
        bundle = load_scope_bundle(garden_root, scope)
        for row in bundle.get("review_rows", []):
            all_review_rows.append({
                **row,
                "scope": scope,
            })

    return {
        "scope_rows": scope_rows,
        "review_rows": all_review_rows,
    }


@st.cache_data(show_spinner=False)
def load_prompt_workspace_bundle(
    garden_root: str,
    prompt_id: str,
) -> dict[str, Any]:
    """Load the unified backend bundle for one prompt workspace screen."""

    garden = PromptGarden(garden_root)
    garden.init()

    detail_bundle = prompt_detail_bundle(
        garden,
        prompt_id=prompt_id,
    )
    dependency_summary = garden.describe_prompt_dependencies(prompt_id)
    all_review_bundle = load_all_review_rows_bundle(garden_root)
    all_review_rows = all_review_bundle.get("review_rows", [])
    dependent_combo_ids = [
        row.get("id")
        for row in detail_bundle.get("dependent_combo_rows", [])
        if row.get("id")
    ]
    matched_review_rows = prompt_review_rows(
        all_review_rows,
        prompt_id=prompt_id,
        combo_ids=dependent_combo_ids,
    )
    combo_performance_rows = prompt_combo_performance_rows(
        matched_review_rows
    )
    model_performance_rows = prompt_model_performance_rows(
        matched_review_rows
    )

    scored_run_count = sum(
        1 for row in matched_review_rows
        if row.get("score") is not None
    )
    latest_run_at = max(
        (
            str(row.get("created_at") or "")
            for row in matched_review_rows
        ),
        default="",
    ) or None
    execution_signatures = sorted(
        {
            str(row.get("execution_signature"))
            for row in matched_review_rows
            if row.get("execution_signature")
        }
    )
    models = sorted(
        {
            str(row.get("model"))
            for row in matched_review_rows
            if row.get("model")
        }
    )

    return {
        **detail_bundle,
        "usage_summary": {
            "child_prompt_count": len(detail_bundle.get("child_rows", [])),
            "dependent_combo_count": len(
                detail_bundle.get("dependent_combo_rows", [])
            ),
            "dependent_experiment_count": len(
                detail_bundle.get("dependent_experiment_rows", [])
            ),
            "recorded_run_count": len(matched_review_rows),
            "scored_run_count": scored_run_count,
            "model_count": len(models),
            "models": models,
            "latest_run_at": latest_run_at,
            "execution_signature_count": len(execution_signatures),
            "execution_signatures": execution_signatures,
            "has_review_history": bool(matched_review_rows),
        },
        "review_rows": matched_review_rows,
        "combo_performance_rows": combo_performance_rows,
        "model_performance_rows": model_performance_rows,
        "top_combo_rows": combo_performance_rows[:3],
        "top_model_rows": model_performance_rows[:3],
        "scope_rows": all_review_bundle.get("scope_rows", []),
        "dependency_summary": dependency_summary,
    }


@st.cache_data(show_spinner=False)
def load_garden_index_bundle(
    garden_root: str,
) -> dict[str, Any]:
    """Load prompt, combo, and experiment index rows for one workspace."""

    garden = PromptGarden(garden_root)
    garden.init()
    prompt_rows = prompt_usage_rows(garden)
    combo_rows = combo_usage_rows(garden)
    experiment_rows = experiment_summary_rows(garden)

    return {
        "prompt_rows": prompt_rows,
        "combo_rows": combo_rows,
        "experiment_rows": experiment_rows,
        "summary": {
            "prompt_count": len(prompt_rows),
            "combo_count": len(combo_rows),
            "experiment_count": len(experiment_rows),
            "archived_prompt_count": sum(
                1 for row in prompt_rows
                if row.get("is_archived")
            ),
            "archived_combo_count": sum(
                1 for row in combo_rows
                if row.get("is_archived")
            ),
            "archived_experiment_count": sum(
                1 for row in experiment_rows
                if row.get("is_archived")
            ),
            "unreadable_prompt_count": sum(
                1 for row in prompt_rows
                if row.get("file_error")
            ),
            "prompt_path_fallback_count": sum(
                1 for row in prompt_rows
                if row.get("used_path_fallback")
            ),
        },
    }


@st.cache_data(show_spinner=False)
def load_experiment_composition_bundle(
    garden_root: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Load one experiment composition bundle for the control panel."""

    garden = PromptGarden(garden_root)
    garden.init()
    return experiment_composition(
        garden,
        experiment_id=experiment_id,
    )


@st.cache_data(show_spinner=False)
def load_prompt_explorer_bundle(
    garden_root: str,
    prompt_id: str,
) -> dict[str, Any]:
    """Load one prompt inspection bundle for the Prompt Workspace."""

    garden = PromptGarden(garden_root)
    garden.init()
    return prompt_detail_bundle(
        garden,
        prompt_id=prompt_id,
    )


@st.cache_data(show_spinner=False)
def load_combo_explorer_bundle(
    garden_root: str,
    combo_id: str,
) -> dict[str, Any]:
    """Load one combo inspection bundle for the Combo Explorer."""

    garden = PromptGarden(garden_root)
    garden.init()
    return combo_detail_bundle(
        garden,
        combo_id=combo_id,
    )


@st.cache_data(show_spinner=False)
def load_prompt_similarity_items(
    garden_root: str,
) -> list[dict[str, Any]]:
    """Load full-text prompt items for prompt similarity workflows."""

    garden = PromptGarden(garden_root)
    garden.init()
    return prompt_similarity_items(garden)


__all__ = [
    "load_all_review_rows_bundle",
    "load_combo_explorer_bundle",
    "discover_review_scopes",
    "filter_review_data",
    "load_experiment_composition_bundle",
    "load_garden_index_bundle",
    "load_prompt_explorer_bundle",
    "load_prompt_workspace_bundle",
    "load_prompt_similarity_items",
    "load_scope_bundle",
]
