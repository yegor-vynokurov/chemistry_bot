"""Control-surface rendering for the Prompt Garden Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import streamlit as st

from .garden import PromptGarden
from .runner import (
    ExperimentRunConfig,
    build_runner_command,
    list_case_set_rows,
    load_case_set_payload,
    plan_prompt_experiment,
    resolve_case_set_path,
)
from .review_app_data import (
    load_combo_explorer_bundle,
    load_experiment_composition_bundle,
    load_garden_index_bundle,
    load_prompt_explorer_bundle,
    load_prompt_workspace_bundle,
)


_DEFAULT_EXPERIMENT_STATUSES = [
    "planned",
    "running",
    "completed",
    "archived",
]
_EXPERIMENT_FLASH_KEY = "prompt_garden_experiment_flash_message"
_EXPERIMENT_SELECTION_KEY = "prompt_garden_selected_experiment_id"
_CLEANUP_FLASH_KEY = "prompt_garden_cleanup_flash_message"
_AUTHORING_MODE_KEY = "prompt_garden_authoring_mode"
_AUTHORING_SOURCE_TAB_KEY = "prompt_garden_authoring_source_tab"
_AUTHORING_PARENT_PROMPT_KEY = "prompt_garden_authoring_parent_prompt_id"
_AUTHORING_DRAFT_KEY = "prompt_garden_authoring_draft"
_AUTHORING_FLASH_KEY = "prompt_garden_authoring_flash_message"
_PROMPT_SELECTION_KEY = "prompt_garden_selected_prompt_id"
_COMBO_SELECTION_KEY = "prompt_garden_selected_combo_id"
_CONTROL_SECTION_KEY = "prompt_garden_control_section"
_CONTROL_SECTION_REDIRECT_KEY = "prompt_garden_control_section_redirect"

_AUTHORING_CREATE_ROOT_PROMPT = "create_root_prompt"
_AUTHORING_BRANCH_PROMPT = "branch_prompt"
_AUTHORING_CREATE_COMBO = "create_combo"

_CONTROL_SECTION_OPTIONS = (
    "Prompt Workspace",
    "Combo Explorer",
    "Experiments",
    "Cleanup",
    "Review Scopes",
)

_PROMPT_EXPLORER_STATE_KEYS = (
    "prompt_explorer_search",
    "prompt_explorer_types",
    "prompt_explorer_trees",
    "prompt_explorer_branches",
    "prompt_explorer_tags",
    "prompt_explorer_keywords",
    "prompt_explorer_include_archived",
    "prompt_explorer_selected_prompt",
)

_COMBO_EXPLORER_STATE_KEYS = (
    "combo_explorer_search",
    "combo_explorer_statuses",
    "combo_explorer_test_statuses",
    "combo_explorer_kinds",
    "combo_explorer_tags",
    "combo_explorer_include_archived",
    "combo_explorer_selected_combo",
)

_AUTHORING_WIDGET_KEYS = (
    "authoring_root_prompt_type",
    "authoring_root_tree_id",
    "authoring_root_title",
    "authoring_root_description",
    "authoring_root_keywords",
    "authoring_root_branch",
    "authoring_root_tags",
    "authoring_root_text",
    "authoring_branch_title",
    "authoring_branch_description",
    "authoring_branch_keywords",
    "authoring_branch_branch",
    "authoring_branch_tags",
    "authoring_branch_text",
    "authoring_combo_title",
    "authoring_combo_notes",
    "authoring_combo_tags",
    "authoring_combo_system_prompt",
    "authoring_combo_user_prompt",
    "authoring_combo_fewshot_prompt",
    "authoring_combo_include_archived",
)

def build_workspace_status_summary(
    garden_root: str,
    index_bundle: dict[str, Any],
    scopes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build workspace-level counts for the new app shell header."""

    summary = index_bundle.get("summary") or {}
    experiment_rows = index_bundle.get("experiment_rows") or []
    active_experiment_count = sum(
        1 for row in experiment_rows
        if row.get("status") not in {None, "archived", "completed"}
    )
    archived_total = (
        int(summary.get("archived_prompt_count", 0))
        + int(summary.get("archived_combo_count", 0))
        + int(summary.get("archived_experiment_count", 0))
    )
    return {
        "garden_root": garden_root,
        "prompt_count": int(summary.get("prompt_count", 0)),
        "combo_count": int(summary.get("combo_count", 0)),
        "experiment_count": int(summary.get("experiment_count", 0)),
        "review_scope_count": len(scopes),
        "active_experiment_count": active_experiment_count,
        "archived_total": archived_total,
        "unreadable_prompt_count": int(summary.get("unreadable_prompt_count", 0)),
        "prompt_path_fallback_count": int(summary.get("prompt_path_fallback_count", 0)),
    }


def render_workspace_status_header(
    garden_root: str,
    index_bundle: dict[str, Any],
    scopes: Sequence[dict[str, Any]],
) -> None:
    """Render the global shell header shared by Control and Analysis."""

    summary = build_workspace_status_summary(
        garden_root=garden_root,
        index_bundle=index_bundle,
        scopes=scopes,
    )

    st.markdown(
        f"**Workspace:** `{summary['garden_root']}`  "
        f"| **Prompts:** `{summary['prompt_count']}`  "
        f"| **Combos:** `{summary['combo_count']}`  "
        f"| **Experiments:** `{summary['experiment_count']}`  "
        f"| **Review Scopes:** `{summary['review_scope_count']}`"
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Prompts", summary["prompt_count"])
    metric_cols[1].metric("Combos", summary["combo_count"])
    metric_cols[2].metric("Experiments", summary["experiment_count"])
    metric_cols[3].metric("Review Scopes", summary["review_scope_count"])
    metric_cols[4].metric("Active Experiments", summary["active_experiment_count"])
    metric_cols[5].metric("Archived Items", summary["archived_total"])

    st.caption(
        "Control is the workspace overview and experiment-entry surface. "
        "Analysis is the answer-review surface."
    )
    if summary["unreadable_prompt_count"] > 0:
        st.warning(
            "Some prompt files could not be loaded. "
            "The panel stays available, but those prompts need inspection."
        )
    elif summary["prompt_path_fallback_count"] > 0:
        st.info(
            "Some prompt rows were loaded through canonical path fallback. "
            "Their registry paths may need cleanup."
        )


def _experiment_option_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('id')} | {row.get('name', '')}"
        f" | status={row.get('status', '-')}"
        f" | combos={row.get('combo_count', 0)}"
    )


def _prompt_option_label(row: dict[str, Any]) -> str:
    prompt_title = str(
        row.get("display_title")
        or row.get("title")
        or ""
    ).strip()
    prompt_description = str(row.get("description") or "").strip()
    if len(prompt_description) > 60:
        prompt_description = prompt_description[:57].rstrip() + "..."
    description_suffix = (
        f" | {prompt_description}"
        if prompt_description else ""
    )
    return (
        f"{row.get('id')} | {prompt_title}"
        f" | type={row.get('type', '-')}"
        f" | branch={row.get('branch', '-')}"
        f" | combos={row.get('combo_count', 0)}"
        f" | experiments={row.get('experiment_count', 0)}"
        f"{description_suffix}"
    )


def _combo_option_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('id')} | {row.get('title', '')}"
        f" | status={row.get('status', '-')}"
        f" | test={row.get('test_status', '-')}"
        f" | prompts={len(row.get('prompt_roles', []))}"
        f" | experiments={row.get('experiment_count', 0)}"
    )


def _dependency_id_preview(ids: Sequence[str], *, limit: int = 5) -> str:
    visible_ids = [str(item) for item in ids[:limit] if item]
    hidden_count = max(len(ids) - len(visible_ids), 0)
    if not visible_ids:
        return "-"
    suffix = f" +{hidden_count} more" if hidden_count else ""
    return ", ".join(visible_ids) + suffix


def _format_score_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric_value:.3f}".rstrip("0").rstrip(".")


def _render_prompt_usage_results_block(
    garden_root: str,
    prompt_bundle: dict[str, Any],
    selected_prompt_id: str,
) -> None:
    usage_summary = prompt_bundle.get("usage_summary") or {}
    combo_performance_rows = list(
        prompt_bundle.get("combo_performance_rows") or []
    )
    model_performance_rows = list(
        prompt_bundle.get("model_performance_rows") or []
    )
    top_combo_rows = list(prompt_bundle.get("top_combo_rows") or [])
    top_model_rows = list(prompt_bundle.get("top_model_rows") or [])
    dependent_combo_rows = list(
        prompt_bundle.get("dependent_combo_rows") or []
    )
    dependent_experiment_rows = list(
        prompt_bundle.get("dependent_experiment_rows") or []
    )

    st.markdown("### Usage & Results")
    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "Child Prompts",
        usage_summary.get("child_prompt_count", 0),
    )
    metric_cols[1].metric(
        "Combos",
        usage_summary.get("dependent_combo_count", 0),
    )
    metric_cols[2].metric(
        "Experiments",
        usage_summary.get("dependent_experiment_count", 0),
    )
    metric_cols[3].metric(
        "Recorded Runs",
        usage_summary.get("recorded_run_count", 0),
    )
    metric_cols[4].metric(
        "Models Seen",
        usage_summary.get("model_count", 0),
    )

    latest_run_at = usage_summary.get("latest_run_at")
    if latest_run_at:
        st.caption(f"Latest observed run: {latest_run_at}")

    if not usage_summary.get("has_review_history"):
        st.info(
            "No normalized run history exists for this prompt yet. "
            "You can still inspect where it is attached and decide whether to branch or archive it."
        )
    else:
        top_col_left, top_col_right = st.columns(2, gap="large")
        with top_col_left:
            st.markdown("**Top Combos**")
            if top_combo_rows:
                for row in top_combo_rows:
                    combo_title = row.get("combo_title") or row.get("combo_id")
                    st.write(
                        f"{row.get('combo_id')} | {combo_title} | "
                        f"avg={_format_score_value(row.get('average_score'))} | "
                        f"pass={_format_score_value(row.get('pass_rate'))} | "
                        f"best model={row.get('best_model') or '-'}"
                    )
            else:
                st.caption("No combo-level scores are available yet.")

        with top_col_right:
            st.markdown("**Top Models**")
            if top_model_rows:
                for row in top_model_rows:
                    st.write(
                        f"{row.get('model')} | "
                        f"avg={_format_score_value(row.get('average_score'))} | "
                        f"pass={_format_score_value(row.get('pass_rate'))} | "
                        f"best combo={row.get('best_combo_id') or '-'}"
                    )
            else:
                st.caption("No model-level score summary is available yet.")

    st.markdown("**Combo Usage**")
    if combo_performance_rows:
        st.dataframe(
            [
                {
                    "combo_id": row.get("combo_id"),
                    "combo_title": row.get("combo_title"),
                    "prompt_roles": ", ".join(row.get("matched_prompt_roles", [])),
                    "experiment_count": len(row.get("experiment_ids", [])),
                    "run_count": row.get("run_count"),
                    "average_score": row.get("average_score"),
                    "best_score": row.get("best_score"),
                    "pass_rate": row.get("pass_rate"),
                    "best_model": row.get("best_model"),
                    "latest_run_at": row.get("latest_run_at"),
                }
                for row in combo_performance_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif dependent_combo_rows:
        st.caption(
            "These combos currently use the prompt, but no normalized run results were found yet."
        )
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "test_status": row.get("test_status"),
                    "experiment_count": row.get("experiment_count"),
                    "result_experiment_count": row.get("result_experiment_count"),
                    "prompt_roles": ", ".join(row.get("prompt_roles", [])),
                }
                for row in dependent_combo_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("This prompt is not currently used by any combo.")

    if dependent_combo_rows:
        combo_option_map = {
            _combo_option_label(row): row["id"]
            for row in dependent_combo_rows
        }
        selected_combo_label = st.selectbox(
            "Inspect Combo",
            options=list(combo_option_map.keys()),
            key=f"prompt_workspace_combo_preview_{selected_prompt_id}",
        )
        st.markdown("**Selected Combo Preview**")
        _render_combo_inline_preview(
            garden_root=garden_root,
            combo_id=combo_option_map[selected_combo_label],
        )

    st.markdown("**Model Summary**")
    if model_performance_rows:
        st.dataframe(
            [
                {
                    "model": row.get("model"),
                    "run_count": row.get("run_count"),
                    "combo_count": row.get("combo_count"),
                    "experiment_count": row.get("experiment_count"),
                    "average_score": row.get("average_score"),
                    "best_score": row.get("best_score"),
                    "pass_rate": row.get("pass_rate"),
                    "best_combo_id": row.get("best_combo_id"),
                    "latest_run_at": row.get("latest_run_at"),
                }
                for row in model_performance_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No model summary is available yet for this prompt.")

    st.markdown("**Experiment Coverage**")
    if dependent_experiment_rows:
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "combo_count": row.get("combo_count"),
                    "tested_combo_count": row.get("tested_combo_count"),
                    "average_score": row.get("average_score"),
                    "normalized_artifact_count": row.get("normalized_artifact_count"),
                }
                for row in dependent_experiment_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No experiments currently reference this prompt.")


def _parse_tag_text(raw_text: str) -> list[str]:
    normalized_tags: list[str] = []
    seen_tags: set[str] = set()

    for chunk in raw_text.split(","):
        tag = chunk.strip()
        if not tag:
            continue
        normalized_key = tag.casefold()
        if normalized_key in seen_tags:
            continue
        seen_tags.add(normalized_key)
        normalized_tags.append(tag)

    return normalized_tags


def _normalize_control_section(value: Any) -> str:
    section = str(value or "").strip()
    if section in {"Workspace", "Prompt Explorer"}:
        return "Prompt Workspace"
    if section in _CONTROL_SECTION_OPTIONS:
        return section
    return "Prompt Workspace"


def _set_control_section(value: Any) -> None:
    st.session_state[_CONTROL_SECTION_KEY] = _normalize_control_section(
        value
    )


def _queue_control_section_redirect(value: Any) -> None:
    st.session_state[_CONTROL_SECTION_REDIRECT_KEY] = (
        _normalize_control_section(value)
    )


def _clear_session_state_keys(keys: Sequence[str]) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _reset_prompt_explorer_state() -> None:
    _clear_session_state_keys(_PROMPT_EXPLORER_STATE_KEYS)


def _reset_combo_explorer_state() -> None:
    _clear_session_state_keys(_COMBO_EXPLORER_STATE_KEYS)


def _authoring_prompt_selector_label(row: dict[str, Any]) -> str:
    preview = str(row.get("text_preview") or "").strip()
    if len(preview) > 80:
        preview = preview[:80] + "..."
    prompt_title = str(
        row.get("display_title")
        or row.get("title")
        or ""
    ).strip()
    prompt_description = str(row.get("description") or "").strip()
    if len(prompt_description) > 60:
        prompt_description = prompt_description[:57].rstrip() + "..."

    status_suffixes: list[str] = []
    if row.get("is_archived"):
        status_suffixes.append("archived")
    if row.get("file_error"):
        status_suffixes.append("file issue")
    status_text = (
        f" | {' / '.join(status_suffixes)}"
        if status_suffixes else ""
    )
    description_suffix = (
        f" | {prompt_description}"
        if prompt_description else ""
    )

    return (
        f"{row.get('id')} | {prompt_title}"
        f" | branch={row.get('branch', '-')}"
        f"{description_suffix}"
        f" | {preview or '[no preview]'}"
        f"{status_text}"
    )


def _prompt_display_title_text(payload: dict[str, Any]) -> str:
    display_title = str(
        payload.get("display_title")
        or payload.get("title")
        or ""
    ).strip()
    return display_title or "Untitled Prompt"


def _prompt_description_text(payload: dict[str, Any]) -> str:
    return str(payload.get("description") or "").strip()


def _experiment_status_options(
    experiment_rows: Sequence[dict[str, Any]],
) -> list[str]:
    seen_statuses = {
        row.get("status")
        for row in experiment_rows
        if row.get("status")
    }
    ordered_statuses = [
        status
        for status in _DEFAULT_EXPERIMENT_STATUSES
        if status in seen_statuses or status == "planned"
    ]
    extra_statuses = sorted(
        status
        for status in seen_statuses
        if status not in ordered_statuses
    )
    return ordered_statuses + extra_statuses


def _case_option_label(case_row: dict[str, Any]) -> str:
    question = str(case_row.get("question") or "").strip().replace("\n", " ")
    preview = question[:80] + ("..." if len(question) > 80 else "")
    return f"{case_row.get('id')} | {preview}"


def _build_runner_config(
    garden_root: str,
    experiment_id: str,
    *,
    model: str,
    bot_variant: str,
    fewshot_id: str | None,
    use_rag: bool,
    rag_k: int,
    candidate_k: int,
    max_context_chars: int,
    max_history_messages: int,
    case_set: str | None,
    only_case_ids: Sequence[str],
    skip_case_ids: Sequence[str],
    only_combo_ids: Sequence[str],
    skip_combo_ids: Sequence[str],
    run_mode: str,
    dry_run: bool,
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        garden_root=Path(garden_root).resolve(),
        experiment_id=experiment_id,
        model=model,
        bot_variant=bot_variant,  # type: ignore[arg-type]
        fewshot_id=fewshot_id,
        use_rag=use_rag,
        rag_k=rag_k,
        candidate_k=candidate_k,
        max_context_chars=max_context_chars,
        max_history_messages=max_history_messages,
        case_set=case_set,
        only_case_ids=tuple(only_case_ids),
        skip_case_ids=tuple(skip_case_ids),
        only_combo_ids=tuple(only_combo_ids),
        skip_combo_ids=tuple(skip_combo_ids),
        run_mode=run_mode,  # type: ignore[arg-type]
        dry_run=dry_run,
    )


def _queue_experiment_refresh(
    message: str,
    *,
    selected_experiment_id: str | None = None,
) -> None:
    st.session_state[_EXPERIMENT_FLASH_KEY] = message
    if selected_experiment_id is not None:
        st.session_state[_EXPERIMENT_SELECTION_KEY] = selected_experiment_id
    st.cache_data.clear()
    st.rerun()


def _consume_experiment_flash_message() -> None:
    message = st.session_state.pop(_EXPERIMENT_FLASH_KEY, None)
    if message:
        st.success(message)


def _queue_cleanup_refresh(message: str) -> None:
    st.session_state[_CLEANUP_FLASH_KEY] = message
    st.cache_data.clear()
    st.rerun()


def _queue_prompt_workspace_refresh(
    message: str,
    *,
    selected_prompt_id: str | None = None,
    ensure_archived_visible: bool = False,
    clear_selected_prompt: bool = False,
) -> None:
    st.session_state[_CLEANUP_FLASH_KEY] = message
    _queue_control_section_redirect("Prompt Workspace")
    if clear_selected_prompt:
        st.session_state.pop(_PROMPT_SELECTION_KEY, None)
        st.session_state.pop("prompt_explorer_selected_prompt", None)
    elif selected_prompt_id is not None:
        st.session_state[_PROMPT_SELECTION_KEY] = selected_prompt_id
    if ensure_archived_visible:
        st.session_state["prompt_explorer_include_archived"] = True
    st.cache_data.clear()
    st.rerun()


def _consume_cleanup_flash_message() -> None:
    message = st.session_state.pop(_CLEANUP_FLASH_KEY, None)
    if message:
        st.success(message)


def _consume_authoring_flash_message() -> None:
    message = st.session_state.pop(_AUTHORING_FLASH_KEY, None)
    if message:
        st.success(message)


def _authoring_mode_title(mode: str | None) -> str:
    if mode == _AUTHORING_CREATE_ROOT_PROMPT:
        return "Create Root Prompt"
    if mode == _AUTHORING_BRANCH_PROMPT:
        return "Branch Prompt"
    if mode == _AUTHORING_CREATE_COMBO:
        return "Create Combo"
    return "Authoring Workspace"


def _authoring_primary_action_label(mode: str | None) -> str:
    if mode == _AUTHORING_CREATE_COMBO:
        return "Register Combo"
    return "Register Prompt"


def _build_root_prompt_draft_defaults() -> dict[str, Any]:
    return {
        "prompt_type": "system",
        "tree_id": "",
        "title": "",
        "description": "",
        "keywords_text": "",
        "branch": "main",
        "tags_text": "",
        "text": "",
    }


def _build_branch_prompt_draft_defaults(
    prompt_payload: dict[str, Any],
) -> dict[str, Any]:
    source_title = str(prompt_payload.get("title") or "").strip()
    source_branch = str(prompt_payload.get("branch") or "").strip() or "main"
    source_tags = [
        str(tag).strip()
        for tag in prompt_payload.get("tags", [])
        if str(tag).strip()
    ]
    source_keywords = [
        str(keyword).strip()
        for keyword in prompt_payload.get("keywords", [])
        if str(keyword).strip()
    ]

    return {
        "parent_prompt_id": prompt_payload.get("id"),
        "prompt_type": prompt_payload.get("type"),
        "tree_id": prompt_payload.get("tree_id"),
        "title": (
            f"{source_title} branch"
            if source_title else "Prompt branch"
        ),
        "branch": (
            f"{source_branch}_v2"
            if source_branch else "branch_v2"
        ),
        "description": prompt_payload.get("description") or "",
        "keywords_text": ", ".join(source_keywords),
        "tags_text": ", ".join(source_tags),
        "text": prompt_payload.get("text") or "",
    }


def _build_combo_draft_defaults() -> dict[str, Any]:
    return {
        "title": "",
        "notes": "",
        "tags_text": "",
        "system_prompt_id": "",
        "user_prompt_id": "",
        "fewshot_prompt_id": "",
        "include_archived_prompts": False,
    }


def _apply_authoring_widget_defaults(
    mode: str,
    draft: dict[str, Any],
) -> None:
    for widget_key in _AUTHORING_WIDGET_KEYS:
        st.session_state.pop(widget_key, None)

    if mode == _AUTHORING_CREATE_ROOT_PROMPT:
        st.session_state["authoring_root_prompt_type"] = str(
            draft.get("prompt_type") or "system"
        )
        st.session_state["authoring_root_tree_id"] = str(
            draft.get("tree_id") or ""
        )
        st.session_state["authoring_root_title"] = str(
            draft.get("title") or ""
        )
        st.session_state["authoring_root_description"] = str(
            draft.get("description") or ""
        )
        st.session_state["authoring_root_keywords"] = str(
            draft.get("keywords_text") or ""
        )
        st.session_state["authoring_root_branch"] = str(
            draft.get("branch") or "main"
        )
        st.session_state["authoring_root_tags"] = str(
            draft.get("tags_text") or ""
        )
        st.session_state["authoring_root_text"] = str(
            draft.get("text") or ""
        )
    elif mode == _AUTHORING_BRANCH_PROMPT:
        st.session_state["authoring_branch_title"] = str(
            draft.get("title") or ""
        )
        st.session_state["authoring_branch_description"] = str(
            draft.get("description") or ""
        )
        st.session_state["authoring_branch_keywords"] = str(
            draft.get("keywords_text") or ""
        )
        st.session_state["authoring_branch_branch"] = str(
            draft.get("branch") or "branch_v2"
        )
        st.session_state["authoring_branch_tags"] = str(
            draft.get("tags_text") or ""
        )
        st.session_state["authoring_branch_text"] = str(
            draft.get("text") or ""
        )
    elif mode == _AUTHORING_CREATE_COMBO:
        st.session_state["authoring_combo_title"] = str(
            draft.get("title") or ""
        )
        st.session_state["authoring_combo_notes"] = str(
            draft.get("notes") or ""
        )
        st.session_state["authoring_combo_tags"] = str(
            draft.get("tags_text") or ""
        )
        st.session_state["authoring_combo_system_prompt"] = str(
            draft.get("system_prompt_id") or ""
        )
        st.session_state["authoring_combo_user_prompt"] = str(
            draft.get("user_prompt_id") or ""
        )
        st.session_state["authoring_combo_fewshot_prompt"] = str(
            draft.get("fewshot_prompt_id") or ""
        )
        st.session_state["authoring_combo_include_archived"] = bool(
            draft.get("include_archived_prompts") or False
        )


def _clear_authoring_mode() -> None:
    for key in (
        _AUTHORING_MODE_KEY,
        _AUTHORING_SOURCE_TAB_KEY,
        _AUTHORING_PARENT_PROMPT_KEY,
        _AUTHORING_DRAFT_KEY,
    ):
        st.session_state.pop(key, None)
    for widget_key in _AUTHORING_WIDGET_KEYS:
        st.session_state.pop(widget_key, None)
    st.rerun()


def _open_authoring_mode(
    mode: str,
    *,
    source_tab: str,
    prompt_payload: dict[str, Any] | None = None,
) -> None:
    parent_prompt_id: str | None = None
    if mode == _AUTHORING_CREATE_ROOT_PROMPT:
        draft = _build_root_prompt_draft_defaults()
    elif mode == _AUTHORING_BRANCH_PROMPT:
        if prompt_payload is None:
            raise ValueError("Branch Prompt mode requires prompt payload.")
        parent_prompt_id = str(prompt_payload.get("id") or "").strip() or None
        draft = _build_branch_prompt_draft_defaults(prompt_payload)
    elif mode == _AUTHORING_CREATE_COMBO:
        draft = _build_combo_draft_defaults()
    else:
        raise ValueError(f"Unsupported authoring mode: {mode}")

    st.session_state[_AUTHORING_MODE_KEY] = mode
    st.session_state[_AUTHORING_SOURCE_TAB_KEY] = source_tab
    st.session_state[_AUTHORING_PARENT_PROMPT_KEY] = parent_prompt_id
    st.session_state[_AUTHORING_DRAFT_KEY] = draft
    _queue_control_section_redirect(source_tab)
    _apply_authoring_widget_defaults(mode, draft)
    st.rerun()


def _queue_authoring_refresh(
    message: str,
    *,
    selected_prompt_id: str | None = None,
    selected_combo_id: str | None = None,
) -> None:
    st.session_state[_AUTHORING_FLASH_KEY] = message
    target_control_section = st.session_state.get(_AUTHORING_SOURCE_TAB_KEY)
    if selected_prompt_id is not None:
        st.session_state[_PROMPT_SELECTION_KEY] = selected_prompt_id
        _reset_prompt_explorer_state()
        target_control_section = "Prompt Workspace"
    if selected_combo_id is not None:
        st.session_state[_COMBO_SELECTION_KEY] = selected_combo_id
        _reset_combo_explorer_state()
        target_control_section = "Combo Explorer"
    _queue_control_section_redirect(target_control_section)
    for key in (
        _AUTHORING_MODE_KEY,
        _AUTHORING_SOURCE_TAB_KEY,
        _AUTHORING_PARENT_PROMPT_KEY,
        _AUTHORING_DRAFT_KEY,
    ):
        st.session_state.pop(key, None)
    for widget_key in _AUTHORING_WIDGET_KEYS:
        st.session_state.pop(widget_key, None)
    st.cache_data.clear()
    st.rerun()


def _render_shared_authoring_block(
    garden_root: str,
    prompt_rows: Sequence[dict[str, Any]],
) -> None:
    mode = st.session_state.get(_AUTHORING_MODE_KEY)
    source_tab = st.session_state.get(_AUTHORING_SOURCE_TAB_KEY)
    parent_prompt_id = st.session_state.get(_AUTHORING_PARENT_PROMPT_KEY)
    draft = st.session_state.get(_AUTHORING_DRAFT_KEY) or {}

    st.markdown("### Authoring Workspace")
    if not mode:
        st.caption(
            "Prompt and combo creation drafts will open here after you use the action buttons above."
        )
        return

    garden = PromptGarden(garden_root)
    garden.init()

    st.info(
        f"Active mode: {_authoring_mode_title(mode)}"
        + (
            f" | opened from {source_tab}"
            if source_tab else ""
        )
    )
    st.markdown(f"**{_authoring_mode_title(mode)}**")

    if mode == _AUTHORING_CREATE_ROOT_PROMPT:
        prompt_type_options = sorted(
            {
                str(row.get("type"))
                for row in prompt_rows
                if row.get("type")
            }
            | set(garden.prompt_types.keys())
        )
        if not prompt_type_options:
            prompt_type_options = sorted(garden.prompt_types.keys())

        with st.form("authoring_create_root_prompt_form"):
            form_left, form_right = st.columns([1, 2], gap="large")
            with form_left:
                st.selectbox(
                    "Prompt Type",
                    options=prompt_type_options,
                    key="authoring_root_prompt_type",
                )
                st.text_input(
                    "Tree ID",
                    key="authoring_root_tree_id",
                    help="For example: `system_chemistry_main` or `fewshot_safety_branch`.",
                )
                st.text_input(
                    "Prompt Title",
                    key="authoring_root_title",
                    help="Short human-readable prompt name shown in Prompt Workspace.",
                )
                st.text_area(
                    "Description",
                    key="authoring_root_description",
                    height=80,
                    help="Optional short explanation of when this prompt is useful.",
                )
                st.text_input(
                    "Keywords",
                    key="authoring_root_keywords",
                    help="Optional comma-separated discovery keywords for filtering and search.",
                )
                st.text_input(
                    "Branch",
                    key="authoring_root_branch",
                    help="Default is `main`.",
                )
                st.text_input(
                    "Tags",
                    key="authoring_root_tags",
                    help="Comma-separated tags.",
                )
            with form_right:
                st.text_area(
                    "Prompt Text",
                    key="authoring_root_text",
                    height=320,
                )

            submit_col, cancel_col = st.columns(2)
            with submit_col:
                save_submitted = st.form_submit_button(
                    "Register Prompt",
                    use_container_width=True,
                )
            with cancel_col:
                cancel_submitted = st.form_submit_button(
                    "Cancel",
                    use_container_width=True,
                )

        if cancel_submitted:
            _clear_authoring_mode()

        if save_submitted:
            prompt_type = str(
                st.session_state.get("authoring_root_prompt_type") or ""
            ).strip()
            tree_id = str(
                st.session_state.get("authoring_root_tree_id") or ""
            ).strip()
            title = str(
                st.session_state.get("authoring_root_title") or ""
            ).strip()
            description = str(
                st.session_state.get("authoring_root_description") or ""
            ).strip()
            keywords = _parse_tag_text(
                str(st.session_state.get("authoring_root_keywords") or "")
            )
            branch = str(
                st.session_state.get("authoring_root_branch") or ""
            ).strip() or "main"
            tags = _parse_tag_text(
                str(st.session_state.get("authoring_root_tags") or "")
            )
            text = str(
                st.session_state.get("authoring_root_text") or ""
            )

            validation_errors: list[str] = []
            if not prompt_type:
                validation_errors.append("Prompt type is required.")
            if not tree_id:
                validation_errors.append("Tree ID is required.")
            if not title:
                validation_errors.append("Title is required.")
            if not text.strip():
                validation_errors.append("Prompt text is required.")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                try:
                    created_prompt = garden.create_root(
                        prompt_type=prompt_type,
                        tree_id=tree_id,
                        title=title,
                        text=text,
                        tags=tags,
                        branch=branch,
                        description=description,
                        keywords=keywords,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_authoring_refresh(
                        f"Prompt `{created_prompt['id']}` created successfully.",
                        selected_prompt_id=str(created_prompt["id"]),
                    )
        return

    if mode == _AUTHORING_BRANCH_PROMPT:
        parent_prompt_payload: dict[str, Any] = {}
        parent_prompt_file_error: str | None = None
        parent_prompt_is_archived = False
        parent_prompt_used_path_fallback = False
        if parent_prompt_id:
            try:
                parent_prompt_bundle = load_prompt_explorer_bundle(
                    garden_root,
                    str(parent_prompt_id),
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                parent_prompt_payload = parent_prompt_bundle.get("prompt") or {}
                parent_prompt_file_error = (
                    str(parent_prompt_payload.get("file_error"))
                    if parent_prompt_payload.get("file_error")
                    else None
                )
                parent_prompt_is_archived = bool(
                    parent_prompt_payload.get("is_archived")
                )
                parent_prompt_used_path_fallback = bool(
                    parent_prompt_payload.get("used_path_fallback")
                )

        if parent_prompt_is_archived:
            st.warning(
                "You are branching from an archived prompt. "
                "This is allowed, but check that you really want to revive this line of work."
            )
        if parent_prompt_file_error:
            st.warning(
                "The parent prompt has file-read issues. "
                "Branching is still available because metadata and current draft text are present."
            )
            st.code(parent_prompt_file_error)
        elif parent_prompt_used_path_fallback:
            st.info(
                "The parent prompt was loaded through a canonical fallback path because its stored path was stale."
            )

        with st.form("authoring_branch_prompt_form"):
            form_left, form_right = st.columns([1, 2], gap="large")
            with form_left:
                st.text_input(
                    "Parent Prompt ID",
                    value=str(parent_prompt_id or ""),
                    disabled=True,
                )
                st.text_input(
                    "Prompt Type",
                    value=str(draft.get("prompt_type") or ""),
                    disabled=True,
                )
                st.text_input(
                    "Tree ID",
                    value=str(draft.get("tree_id") or ""),
                    disabled=True,
                )
                st.text_input(
                    "Prompt Title",
                    key="authoring_branch_title",
                    help="Short human-readable prompt name shown in Prompt Workspace.",
                )
                st.text_area(
                    "Description",
                    key="authoring_branch_description",
                    height=80,
                    help="Optional short explanation of what changed in this branch.",
                )
                st.text_input(
                    "Keywords",
                    key="authoring_branch_keywords",
                    help="Optional comma-separated discovery keywords for filtering and search.",
                )
                st.text_input(
                    "Branch",
                    key="authoring_branch_branch",
                )
                st.text_input(
                    "Tags",
                    key="authoring_branch_tags",
                    help="Comma-separated tags.",
                )
            with form_right:
                st.text_area(
                    "Prompt Text",
                    key="authoring_branch_text",
                    height=320,
                )

            submit_col, cancel_col = st.columns(2)
            with submit_col:
                save_submitted = st.form_submit_button(
                    "Register Prompt",
                    use_container_width=True,
                )
            with cancel_col:
                cancel_submitted = st.form_submit_button(
                    "Cancel",
                    use_container_width=True,
                )

        if parent_prompt_id:
            st.caption("Parent prompt summary")
            st.json({
                "id": parent_prompt_payload.get("id"),
                "title": (
                    parent_prompt_payload.get("display_title")
                    or parent_prompt_payload.get("title")
                ),
                "description": parent_prompt_payload.get("description"),
                "type": parent_prompt_payload.get("type"),
                "tree_id": parent_prompt_payload.get("tree_id"),
                "branch": parent_prompt_payload.get("branch"),
                "tags": parent_prompt_payload.get("tags", []),
                "keywords": parent_prompt_payload.get("keywords", []),
            })
            with st.expander("Parent Prompt Preview", expanded=False):
                _render_prompt_inline_preview(
                    garden_root=garden_root,
                    prompt_id=str(parent_prompt_id),
                )

        if cancel_submitted:
            _clear_authoring_mode()

        if save_submitted:
            title = str(
                st.session_state.get("authoring_branch_title") or ""
            ).strip()
            description = str(
                st.session_state.get("authoring_branch_description") or ""
            ).strip()
            keywords = _parse_tag_text(
                str(st.session_state.get("authoring_branch_keywords") or "")
            )
            branch = str(
                st.session_state.get("authoring_branch_branch") or ""
            ).strip() or "branch_v2"
            tags = _parse_tag_text(
                str(st.session_state.get("authoring_branch_tags") or "")
            )
            text = str(
                st.session_state.get("authoring_branch_text") or ""
            )

            validation_errors: list[str] = []
            if not parent_prompt_id:
                validation_errors.append("Parent prompt is required for branching.")
            if not title:
                validation_errors.append("Title is required.")
            if not text.strip():
                validation_errors.append("Prompt text is required.")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                try:
                    created_prompt = garden.create_child(
                        parent_id=str(parent_prompt_id),
                        title=title,
                        text=text,
                        tags=tags,
                        branch=branch,
                        description=description,
                        keywords=keywords,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_authoring_refresh(
                        f"Prompt branch `{created_prompt['id']}` created successfully.",
                        selected_prompt_id=str(created_prompt["id"]),
                    )
        return

    if mode == _AUTHORING_CREATE_COMBO:
        include_archived_prompts = bool(
            st.session_state.get("authoring_combo_include_archived") or False
        )
        prompt_row_by_id = {
            str(row["id"]): row
            for row in prompt_rows
            if row.get("id")
        }
        selectable_prompt_rows = sorted(
            [
                row for row in prompt_rows
                if include_archived_prompts or not row.get("is_archived")
            ],
            key=lambda row: (
                str(row.get("type") or ""),
                bool(row.get("is_archived")),
                str(row.get("title") or ""),
                str(row.get("id") or ""),
            ),
        )
        system_prompt_rows = [
            row for row in selectable_prompt_rows
            if row.get("type") == "system"
        ]
        user_prompt_rows = [
            row for row in selectable_prompt_rows
            if row.get("type") == "user"
        ]
        fewshot_prompt_rows = [
            row for row in selectable_prompt_rows
            if row.get("type") == "fewshot"
        ]

        prompt_label_map = {
            str(row["id"]): _authoring_prompt_selector_label(row)
            for row in selectable_prompt_rows
            if row.get("id")
        }

        def _sync_combo_selector(
            widget_key: str,
            available_rows: Sequence[dict[str, Any]],
        ) -> None:
            option_ids = {
                str(row["id"])
                for row in available_rows
                if row.get("id")
            }
            current_value = str(
                st.session_state.get(widget_key) or ""
            ).strip()
            if current_value and current_value not in option_ids:
                st.session_state[widget_key] = ""

        _sync_combo_selector(
            "authoring_combo_system_prompt",
            system_prompt_rows,
        )
        _sync_combo_selector(
            "authoring_combo_user_prompt",
            user_prompt_rows,
        )
        _sync_combo_selector(
            "authoring_combo_fewshot_prompt",
            fewshot_prompt_rows,
        )

        st.checkbox(
            "Show Archived Prompts",
            key="authoring_combo_include_archived",
            help="Archived prompts are hidden from the combo selectors by default.",
        )

        form_left, form_right = st.columns([1, 2], gap="large")
        with form_left:
            st.text_input(
                "Title",
                key="authoring_combo_title",
            )
            st.text_input(
                "Tags",
                key="authoring_combo_tags",
                help="Comma-separated tags.",
            )
            st.selectbox(
                "System Prompt",
                options=[""] + [
                    str(row["id"])
                    for row in system_prompt_rows
                ],
                format_func=lambda prompt_id: (
                    "Select a system prompt"
                    if not prompt_id else prompt_label_map[prompt_id]
                ),
                key="authoring_combo_system_prompt",
            )
            st.selectbox(
                "User Prompt",
                options=[""] + [
                    str(row["id"])
                    for row in user_prompt_rows
                ],
                format_func=lambda prompt_id: (
                    "Select a user prompt"
                    if not prompt_id else prompt_label_map[prompt_id]
                ),
                key="authoring_combo_user_prompt",
            )
            st.selectbox(
                "Few-Shot Prompt",
                options=[""] + [
                    str(row["id"])
                    for row in fewshot_prompt_rows
                ],
                format_func=lambda prompt_id: (
                    "No few-shot prompt"
                    if not prompt_id else prompt_label_map[prompt_id]
                ),
                key="authoring_combo_fewshot_prompt",
            )
        with form_right:
            st.text_area(
                "Notes",
                key="authoring_combo_notes",
                height=220,
                help="Optional operator notes about why this combo exists.",
            )

        availability_notes: list[str] = []
        if not system_prompt_rows:
            availability_notes.append(
                "No selectable system prompts are available right now."
            )
        if not user_prompt_rows:
            availability_notes.append(
                "No selectable user prompts are available right now."
            )
        if not fewshot_prompt_rows:
            availability_notes.append(
                "No few-shot prompts are registered yet, so this combo will be two-part unless you add one later."
            )
        hidden_archived_prompt_count = sum(
            1 for row in prompt_rows
            if row.get("is_archived")
        )
        if hidden_archived_prompt_count and not include_archived_prompts:
            availability_notes.append(
                f"{hidden_archived_prompt_count} archived prompt(s) are hidden from these selectors by default."
            )
        for note in availability_notes:
            st.caption(note)

        title = str(
            st.session_state.get("authoring_combo_title") or ""
        ).strip()
        notes = str(
            st.session_state.get("authoring_combo_notes") or ""
        ).strip()
        tags = _parse_tag_text(
            str(st.session_state.get("authoring_combo_tags") or "")
        )
        system_prompt_id = str(
            st.session_state.get("authoring_combo_system_prompt") or ""
        ).strip()
        user_prompt_id = str(
            st.session_state.get("authoring_combo_user_prompt") or ""
        ).strip()
        fewshot_prompt_id = str(
            st.session_state.get("authoring_combo_fewshot_prompt") or ""
        ).strip()

        candidate_prompt_ids: dict[str, str] = {}
        if system_prompt_id:
            candidate_prompt_ids["system"] = system_prompt_id
        if user_prompt_id:
            candidate_prompt_ids["user"] = user_prompt_id
        if fewshot_prompt_id:
            candidate_prompt_ids["fewshot"] = fewshot_prompt_id

        preview_rows: list[dict[str, Any]] = []
        for role, prompt_id in candidate_prompt_ids.items():
            row = prompt_row_by_id.get(prompt_id)
            if row is None:
                continue
            file_state = "ok"
            if row.get("file_error"):
                file_state = "file issue"
            elif row.get("used_path_fallback"):
                file_state = "path fallback"

            preview_rows.append({
                "role": role,
                "prompt_id": prompt_id,
                "title": row.get("title"),
                "type": row.get("type"),
                "branch": row.get("branch"),
                "is_archived": row.get("is_archived"),
                "file_state": file_state,
                "tags": ", ".join(row.get("tags", [])),
                "text_preview": row.get("text_preview"),
            })

        existing_combo = None
        if "system" in candidate_prompt_ids and "user" in candidate_prompt_ids:
            existing_combo = garden.find_combo_by_prompt_ids(
                candidate_prompt_ids
            )

        st.markdown("**Prompt Set Preview**")
        if preview_rows:
            st.dataframe(
                preview_rows,
                use_container_width=True,
                hide_index=True,
            )
            for row in preview_rows:
                with st.expander(
                    f"{row['role']} | {row['prompt_id']} | {row.get('title') or ''}",
                    expanded=False,
                ):
                    _render_prompt_inline_preview(
                        garden_root=garden_root,
                        prompt_id=str(row["prompt_id"]),
                    )
        else:
            st.caption(
                "Select prompts to preview the exact combo composition before registration."
            )

        if existing_combo is not None:
            st.warning(
                "This exact prompt-role combination already exists as "
                f"`{existing_combo['id']}`. Reuse it or change the selected prompts."
            )
            st.json({
                "id": existing_combo.get("id"),
                "title": existing_combo.get("title"),
                "status": existing_combo.get("status"),
                "test_status": existing_combo.get("test_status"),
                "is_archived": existing_combo.get("is_archived"),
                "prompt_ids": existing_combo.get("prompt_ids", {}),
            })
        elif "system" in candidate_prompt_ids and "user" in candidate_prompt_ids:
            st.info(
                "No existing combo matches this exact prompt-role set. "
                "You can register it when the title and notes look right."
            )
        else:
            st.caption(
                "Pick at least a system prompt and a user prompt to validate uniqueness."
            )

        submit_col, cancel_col = st.columns(2)
        with submit_col:
            save_submitted = st.button(
                "Register Combo",
                use_container_width=True,
                disabled=existing_combo is not None,
                help=(
                    "Saving is blocked while the selected prompt-role combination already exists."
                    if existing_combo is not None else None
                ),
                key="authoring_combo_save",
            )
        with cancel_col:
            cancel_submitted = st.button(
                "Cancel",
                use_container_width=True,
                key="authoring_combo_cancel",
            )

        if cancel_submitted:
            _clear_authoring_mode()

        if save_submitted:
            validation_errors: list[str] = []
            if not title:
                validation_errors.append("Title is required.")
            if not system_prompt_id:
                validation_errors.append("System prompt selection is required.")
            if not user_prompt_id:
                validation_errors.append("User prompt selection is required.")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                prompt_ids = {
                    "system": system_prompt_id,
                    "user": user_prompt_id,
                }
                if fewshot_prompt_id:
                    prompt_ids["fewshot"] = fewshot_prompt_id

                duplicate_combo = garden.find_combo_by_prompt_ids(prompt_ids)
                if duplicate_combo is not None:
                    st.error(
                        "This exact prompt-role combination already exists as "
                        f"`{duplicate_combo['id']}`."
                    )
                else:
                    try:
                        created_combo = garden.create_combo(
                            title=title,
                            prompt_ids=prompt_ids,
                            notes=notes,
                            tags=tags,
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _queue_authoring_refresh(
                            f"Combo `{created_combo['id']}` created successfully.",
                            selected_combo_id=str(created_combo["id"]),
                        )
        return

    summary_col, action_col = st.columns([2, 1], gap="large")
    with summary_col:
        st.markdown(f"**{_authoring_mode_title(mode)} Draft**")
        if parent_prompt_id:
            st.caption(f"Parent prompt: `{parent_prompt_id}`")
            with st.expander("Parent Prompt Preview", expanded=False):
                _render_prompt_inline_preview(
                    garden_root=garden_root,
                    prompt_id=str(parent_prompt_id),
                )
        st.json(draft)

    with action_col:
        st.markdown("**Draft Actions**")
        st.button(
            _authoring_primary_action_label(mode),
            use_container_width=True,
            disabled=True,
            help=(
                "This save flow is introduced in a later authoring step."
            ),
            key=f"authoring_primary_placeholder_{mode}",
        )
        if st.button(
            "Cancel",
            use_container_width=True,
            key=f"authoring_cancel_{mode}",
        ):
            _clear_authoring_mode()


def _matches_experiment_search(
    row: dict[str, Any],
    query: str,
) -> bool:
    if not query:
        return True
    haystack = " ".join([
        str(row.get("id", "")),
        str(row.get("name", "")),
        str(row.get("status", "")),
        str(row.get("goal", "")),
        str(row.get("hypothesis", "")),
        str(row.get("notes", "")),
        " ".join(str(tag) for tag in row.get("tags", [])),
        " ".join(str(combo_id) for combo_id in row.get("combo_ids", [])),
    ]).lower()
    return query in haystack


def _matches_search(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = " ".join([
        str(row.get("id", "")),
        str(row.get("display_title", "")),
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("type", "")),
        str(row.get("tree_id", "")),
        str(row.get("branch", "")),
        str(row.get("path", "")),
        " ".join(str(tag) for tag in row.get("tags", [])),
        " ".join(str(keyword) for keyword in row.get("keywords", [])),
        str(row.get("text_preview", "")),
    ]).lower()
    return query in haystack


def _matches_combo_search(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = " ".join([
        str(row.get("id", "")),
        str(row.get("title", "")),
        str(row.get("status", "")),
        str(row.get("test_status", "")),
        str(row.get("kind", "")),
        str(row.get("notes", "")),
        " ".join(str(tag) for tag in row.get("tags", [])),
        " ".join(
            f"{role}:{prompt_id}"
            for role, prompt_id in row.get("prompt_ids", {}).items()
        ),
    ]).lower()
    return query in haystack


def _render_prompt_inline_preview(
    garden_root: str,
    prompt_id: str,
) -> None:
    prompt_bundle = load_prompt_explorer_bundle(
        garden_root,
        prompt_id,
    )
    prompt_payload = prompt_bundle.get("prompt") or {}
    prompt_summary = prompt_bundle.get("summary") or {}

    st.caption(
        f"{prompt_payload.get('id')} | "
        f"type={prompt_payload.get('type')} | "
        f"tree={prompt_payload.get('tree_id')} | "
        f"branch={prompt_payload.get('branch')}"
    )
    st.json({
        "title": prompt_payload.get("display_title") or prompt_payload.get("title"),
        "description": prompt_payload.get("description"),
        "keywords": prompt_payload.get("keywords", []),
        "path": prompt_payload.get("path"),
        "resolved_path": prompt_payload.get("resolved_path"),
        "tags": prompt_payload.get("tags", []),
        "combo_count": prompt_summary.get("combo_count"),
        "experiment_count": prompt_summary.get("experiment_count"),
        "child_count": prompt_summary.get("child_count"),
    })
    if prompt_payload.get("file_error"):
        st.warning(str(prompt_payload.get("file_error")))
    elif prompt_payload.get("used_path_fallback"):
        st.info(
            "Loaded this prompt through a canonical fallback path because the stored path was stale."
        )
    with st.expander("Full Prompt Text", expanded=False):
        st.code(
            prompt_payload.get("text") or "",
            language="markdown",
        )


def _render_combo_inline_preview(
    garden_root: str,
    combo_id: str,
) -> None:
    combo_bundle = load_combo_explorer_bundle(
        garden_root,
        combo_id,
    )
    combo_payload = combo_bundle.get("combo") or {}
    combo_summary = combo_bundle.get("summary") or {}
    prompt_role_rows = combo_bundle.get("prompt_role_rows") or []
    dependent_experiment_rows = (
        combo_bundle.get("dependent_experiment_rows") or []
    )

    st.caption(
        f"{combo_payload.get('id')} | "
        f"status={combo_payload.get('status')} | "
        f"test={combo_payload.get('test_status')} | "
        f"kind={combo_summary.get('kind')}"
    )
    st.json({
        "title": combo_payload.get("title"),
        "tags": combo_payload.get("tags", []),
        "score": combo_payload.get("score"),
        "prompt_ids": combo_payload.get("prompt_ids", {}),
        "experiment_count": combo_summary.get("experiment_count"),
        "result_experiment_count": combo_summary.get("result_experiment_count"),
    })
    st.dataframe(
        [
            {
                "role": row.get("role"),
                "prompt_id": row.get("prompt_id"),
                "prompt_title": row.get("prompt_title"),
                "prompt_type": row.get("prompt_type"),
                "branch": row.get("branch"),
            }
            for row in prompt_role_rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    if dependent_experiment_rows:
        st.caption(
            "Experiments: "
            + ", ".join(
                row.get("id") or ""
                for row in dependent_experiment_rows
            )
        )


def _render_prompt_explorer(
    garden_root: str,
    prompt_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Prompt Workspace")
    _consume_cleanup_flash_message()
    if not prompt_rows:
        st.info("No prompt nodes were found in this Prompt Garden workspace.")
        st.markdown("### Workspace Actions")
        if st.button(
            "Create Root Prompt",
            use_container_width=True,
            key="prompt_authoring_create_root_empty_workspace",
        ):
            _open_authoring_mode(
                _AUTHORING_CREATE_ROOT_PROMPT,
                source_tab="Prompt Workspace",
            )
        return

    explorer_col, detail_col = st.columns([1, 2], gap="large")
    selected_prompt_id: str | None = None
    prompt_bundle: dict[str, Any] = {}
    prompt_payload: dict[str, Any] = {}

    prompt_type_options = sorted(
        {
            row.get("type")
            for row in prompt_rows
            if row.get("type")
        }
    )
    tree_id_options = sorted(
        {
            row.get("tree_id")
            for row in prompt_rows
            if row.get("tree_id")
        }
    )
    branch_options = sorted(
        {
            row.get("branch")
            for row in prompt_rows
            if row.get("branch")
        }
    )
    tag_options = sorted(
        {
            tag
            for row in prompt_rows
            for tag in row.get("tags", [])
        }
    )
    keyword_options = sorted(
        {
            keyword
            for row in prompt_rows
            for keyword in row.get("keywords", [])
        }
    )

    with explorer_col:
        search_query = st.text_input(
            "Search Prompt",
            help=(
                "Search by id, prompt title, description, type, branch, tree id, "
                "tags, keywords, or text preview."
            ),
            key="prompt_explorer_search",
        ).strip().lower()
        selected_prompt_types = st.multiselect(
            "Prompt Types",
            options=prompt_type_options,
            key="prompt_explorer_types",
        )
        selected_tree_ids = st.multiselect(
            "Tree IDs",
            options=tree_id_options,
            key="prompt_explorer_trees",
        )
        selected_branches = st.multiselect(
            "Branches",
            options=branch_options,
            key="prompt_explorer_branches",
        )
        selected_tags = st.multiselect(
            "Tags",
            options=tag_options,
            key="prompt_explorer_tags",
        )
        selected_keywords = st.multiselect(
            "Keywords",
            options=keyword_options,
            key="prompt_explorer_keywords",
        )
        include_archived = st.checkbox(
            "Include Archived Prompts",
            value=True,
            key="prompt_explorer_include_archived",
        )

        filtered_prompt_rows = []
        for row in prompt_rows:
            if (
                selected_prompt_types
                and row.get("type") not in selected_prompt_types
            ):
                continue
            if (
                selected_tree_ids
                and row.get("tree_id") not in selected_tree_ids
            ):
                continue
            if (
                selected_branches
                and row.get("branch") not in selected_branches
            ):
                continue
            if not include_archived and row.get("is_archived"):
                continue
            if selected_tags and not set(selected_tags).issubset(
                set(row.get("tags", []))
            ):
                continue
            if selected_keywords and not set(selected_keywords).issubset(
                set(row.get("keywords", []))
            ):
                continue
            if not _matches_search(row, search_query):
                continue
            filtered_prompt_rows.append(row)

        st.caption(f"Filtered prompts: {len(filtered_prompt_rows)}")
        if not filtered_prompt_rows:
            selected_prompt_id = None
        else:
            option_map = {
                _prompt_option_label(row): row["id"]
                for row in filtered_prompt_rows
            }
            prompt_option_labels = list(option_map.keys())
            selected_prompt_from_state = st.session_state.get(
                _PROMPT_SELECTION_KEY
            )
            default_index = 0
            if selected_prompt_from_state:
                for index, label in enumerate(prompt_option_labels):
                    if option_map[label] == selected_prompt_from_state:
                        default_index = index
                        break
            selected_prompt_label = st.selectbox(
                "Prompt Detail",
                options=prompt_option_labels,
                index=default_index,
                key="prompt_explorer_selected_prompt",
            )
            selected_prompt_id = option_map[selected_prompt_label]
            st.session_state[_PROMPT_SELECTION_KEY] = selected_prompt_id

    with detail_col:
        if not filtered_prompt_rows:
            st.info("No prompts match the current explorer filters.")
        else:
            prompt_bundle = load_prompt_workspace_bundle(
                garden_root,
                selected_prompt_id,
            )
            prompt_payload = prompt_bundle.get("prompt") or {}
            prompt_summary = prompt_bundle.get("summary") or {}
            parsed_fewshot_examples = prompt_bundle.get("parsed_fewshot_examples")
            parsed_fewshot_error = prompt_bundle.get("parsed_fewshot_error")
            prompt_file_error = prompt_payload.get("file_error")
            used_path_fallback = prompt_payload.get("used_path_fallback")
            prompt_title = _prompt_display_title_text(prompt_payload)
            prompt_description = _prompt_description_text(prompt_payload)
            prompt_keywords = list(prompt_payload.get("keywords") or [])

            st.markdown("### Prompt")
            st.markdown(
                f"## {prompt_title} {prompt_payload.get('id')}"
            )
            if prompt_description:
                st.caption(prompt_description)
            st.caption(
                f"type={prompt_payload.get('type')} | "
                f"tree={prompt_payload.get('tree_id')} | "
                f"branch={prompt_payload.get('branch')} | "
                f"archived={prompt_payload.get('is_archived')}"
            )
            if prompt_keywords:
                st.caption(
                    "Keywords: " + ", ".join(prompt_keywords)
                )
            if prompt_file_error:
                st.error(
                    "This prompt's file could not be loaded. "
                    "Metadata is still available so the panel can stay open."
                )
                st.code(str(prompt_file_error))
            elif used_path_fallback:
                st.warning(
                    "The stored prompt path was stale. "
                    "The panel loaded the text from the canonical fallback path."
                )

            st.markdown("**Prompt Text**")
            if prompt_file_error:
                st.caption(
                    "Prompt text is unavailable until the underlying file is restored."
                )
            else:
                st.text_area(
                    "Prompt Text",
                    value=prompt_payload.get("text") or "",
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"prompt_workspace_text_{selected_prompt_id}",
                )

            _render_prompt_usage_results_block(
                garden_root=garden_root,
                prompt_bundle=prompt_bundle,
                selected_prompt_id=selected_prompt_id,
            )

            st.markdown("### Prompt Actions")
            action_left, action_right = st.columns(2, gap="large")
            with action_left:
                st.caption(
                    "Branch to preserve this prompt's history while editing a new child version."
                )
                if st.button(
                    "Branch Prompt",
                    use_container_width=True,
                    key=f"prompt_authoring_branch_{selected_prompt_id}",
                ):
                    _open_authoring_mode(
                        _AUTHORING_BRANCH_PROMPT,
                        source_tab="Prompt Workspace",
                        prompt_payload=prompt_payload,
                    )
            with action_right:
                st.markdown("**Archive Prompt**")
                if prompt_payload.get("is_archived"):
                    st.info(
                        "This prompt is already archived. You can still branch it if you want to continue from it."
                    )
                with st.form(
                    f"prompt_workspace_archive_{selected_prompt_id}"
                ):
                    archive_reason = st.text_area(
                        "Archive Reason",
                        height=100,
                        help="Optional note that will be stored in prompt metadata.",
                    )
                    archive_submitted = st.form_submit_button(
                        "Archive Prompt",
                        use_container_width=True,
                        disabled=bool(prompt_payload.get("is_archived")),
                    )
                if archive_submitted:
                    garden = PromptGarden(garden_root)
                    garden.init()
                    try:
                        garden.archive_prompt(
                            selected_prompt_id,
                            reason=archive_reason,
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _queue_prompt_workspace_refresh(
                            f"Prompt `{selected_prompt_id}` archived.",
                            selected_prompt_id=selected_prompt_id,
                            ensure_archived_visible=True,
                        )

            with st.expander("Prompt Metadata", expanded=False):
                metadata_left, metadata_right = st.columns(2)
                with metadata_left:
                    st.json({
                        "id": prompt_payload.get("id"),
                        "title": prompt_payload.get("display_title") or prompt_payload.get("title"),
                        "description": prompt_payload.get("description"),
                        "keywords": prompt_payload.get("keywords", []),
                        "type": prompt_payload.get("type"),
                        "tree_id": prompt_payload.get("tree_id"),
                        "branch": prompt_payload.get("branch"),
                        "parent_id": prompt_payload.get("parent_id"),
                        "tags": prompt_payload.get("tags", []),
                        "created_at": prompt_payload.get("created_at"),
                        "updated_at": prompt_payload.get("updated_at"),
                        "combo_count": prompt_summary.get("combo_count"),
                        "experiment_count": prompt_summary.get("experiment_count"),
                        "child_count": prompt_summary.get("child_count"),
                    })
                with metadata_right:
                    st.json({
                        "path": prompt_payload.get("path"),
                        "resolved_path": prompt_payload.get("resolved_path"),
                        "file_exists": prompt_payload.get("file_exists"),
                        "used_path_fallback": prompt_payload.get("used_path_fallback"),
                        "stats": prompt_payload.get("stats") or {},
                        "metadata": prompt_payload.get("metadata") or {},
                    })

            if prompt_payload.get("type") == "fewshot":
                with st.expander("Few-Shot Inspection", expanded=False):
                    if prompt_file_error:
                        st.caption(
                            "Few-shot inspection is unavailable until the underlying prompt file is restored."
                        )
                    elif parsed_fewshot_examples is not None:
                        st.dataframe(
                            parsed_fewshot_examples,
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif parsed_fewshot_error:
                        st.warning(
                            "This few-shot prompt could not be parsed as JSON."
                        )
                        st.code(parsed_fewshot_error)
                    else:
                        st.caption(
                            "No few-shot examples are available for this prompt."
                        )

            with st.expander("Prompt Lineage", expanded=False):
                st.markdown("**Lineage**")
                st.dataframe(
                    prompt_bundle.get("lineage_rows", []),
                    use_container_width=True,
                    hide_index=True,
                )
                child_rows = prompt_bundle.get("child_rows", [])
                if child_rows:
                    st.markdown("**Direct Children**")
                    st.dataframe(
                        child_rows,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("This prompt currently has no direct child prompts.")

    st.markdown("### Workspace Actions")
    if st.button(
        "Create Root Prompt",
        use_container_width=True,
        key="prompt_authoring_create_root",
    ):
        _open_authoring_mode(
            _AUTHORING_CREATE_ROOT_PROMPT,
            source_tab="Prompt Workspace",
        )

    _render_prompt_maintenance_panel(
        garden_root=garden_root,
        prompt_bundle=prompt_bundle,
        selected_prompt_id=selected_prompt_id,
    )


def _render_combo_explorer(
    garden_root: str,
    combo_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Combo Explorer")
    if not combo_rows:
        st.info("No combos were found in this Prompt Garden workspace.")
        if st.button(
            "Create Combo",
            use_container_width=True,
            key="combo_authoring_create_combo_empty",
        ):
            _open_authoring_mode(
                _AUTHORING_CREATE_COMBO,
                source_tab="Combo Explorer",
            )
        return

    explorer_col, detail_col = st.columns([1, 2], gap="large")

    status_options = sorted(
        {
            row.get("status")
            for row in combo_rows
            if row.get("status")
        }
    )
    test_status_options = sorted(
        {
            row.get("test_status")
            for row in combo_rows
            if row.get("test_status")
        }
    )
    kind_options = sorted(
        {
            row.get("kind")
            for row in combo_rows
            if row.get("kind")
        }
    )
    tag_options = sorted(
        {
            tag
            for row in combo_rows
            for tag in row.get("tags", [])
        }
    )

    with explorer_col:
        search_query = st.text_input(
            "Search Combo",
            help="Search by id, title, status, test status, kind, tags, notes, or prompt ids.",
            key="combo_explorer_search",
        ).strip().lower()
        selected_statuses = st.multiselect(
            "Statuses",
            options=status_options,
            key="combo_explorer_statuses",
        )
        selected_test_statuses = st.multiselect(
            "Test Statuses",
            options=test_status_options,
            key="combo_explorer_test_statuses",
        )
        selected_kinds = st.multiselect(
            "Kinds",
            options=kind_options,
            key="combo_explorer_kinds",
        )
        selected_tags = st.multiselect(
            "Tags",
            options=tag_options,
            key="combo_explorer_tags",
        )
        include_archived = st.checkbox(
            "Include Archived Combos",
            value=True,
            key="combo_explorer_include_archived",
        )

        filtered_combo_rows = []
        for row in combo_rows:
            if selected_statuses and row.get("status") not in selected_statuses:
                continue
            if (
                selected_test_statuses
                and row.get("test_status") not in selected_test_statuses
            ):
                continue
            if selected_kinds and row.get("kind") not in selected_kinds:
                continue
            if not include_archived and row.get("is_archived"):
                continue
            if selected_tags and not set(selected_tags).issubset(
                set(row.get("tags", []))
            ):
                continue
            if not _matches_combo_search(row, search_query):
                continue
            filtered_combo_rows.append(row)

        st.caption(f"Filtered combos: {len(filtered_combo_rows)}")
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "test_status": row.get("test_status"),
                    "kind": row.get("kind"),
                    "experiment_count": row.get("experiment_count"),
                    "prompt_ids": ", ".join(
                        f"{role}:{prompt_id}"
                        for role, prompt_id in row.get("prompt_ids", {}).items()
                    ),
                    "is_archived": row.get("is_archived"),
                }
                for row in filtered_combo_rows[:20]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered_combo_rows:
            selected_combo_id = None
        else:
            option_map = {
                _combo_option_label(row): row["id"]
                for row in filtered_combo_rows
            }
            combo_option_labels = list(option_map.keys())
            selected_combo_from_state = st.session_state.get(
                _COMBO_SELECTION_KEY
            )
            default_index = 0
            if selected_combo_from_state:
                for index, label in enumerate(combo_option_labels):
                    if option_map[label] == selected_combo_from_state:
                        default_index = index
                        break
            selected_combo_label = st.selectbox(
                "Combo Detail",
                options=combo_option_labels,
                index=default_index,
                key="combo_explorer_selected_combo",
            )
            selected_combo_id = option_map[selected_combo_label]
            st.session_state[_COMBO_SELECTION_KEY] = selected_combo_id

    with detail_col:
        if not filtered_combo_rows:
            st.info("No combos match the current explorer filters.")
        else:
            combo_bundle = load_combo_explorer_bundle(
                garden_root,
                selected_combo_id,
            )
            combo_payload = combo_bundle.get("combo") or {}
            combo_summary = combo_bundle.get("summary") or {}
            prompt_role_rows = combo_bundle.get("prompt_role_rows") or []
            dependent_experiment_rows = (
                combo_bundle.get("dependent_experiment_rows") or []
            )
            derived_combo_rows = combo_bundle.get("derived_combo_rows") or []
            dependency_report = combo_bundle.get("dependency_report") or {}

            st.markdown("### Combo Detail")
            st.caption(
                f"{combo_payload.get('id')} | "
                f"status={combo_payload.get('status')} | "
                f"test={combo_payload.get('test_status')} | "
                f"kind={combo_summary.get('kind')} | "
                f"archived={combo_payload.get('is_archived')}"
            )

            metric_cols = st.columns(5)
            metric_cols[0].metric("Prompt Roles", len(prompt_role_rows))
            metric_cols[1].metric(
                "Experiments",
                combo_summary.get("experiment_count", 0),
            )
            metric_cols[2].metric(
                "Result Experiments",
                combo_summary.get("result_experiment_count", 0),
            )
            metric_cols[3].metric(
                "Runs",
                combo_summary.get("run_count", 0),
            )
            metric_cols[4].metric(
                "Normalized Artifacts",
                combo_summary.get("normalized_artifact_count", 0),
            )

            st.json({
                "title": combo_payload.get("title"),
                "score": combo_payload.get("score"),
                "tags": combo_payload.get("tags", []),
                "prompt_ids": combo_payload.get("prompt_ids", {}),
                "missing_prompt_ids": combo_summary.get("missing_prompt_ids", []),
                "experiment_ids": combo_summary.get("experiment_ids", []),
                "result_experiment_ids": combo_summary.get("result_experiment_ids", []),
                "created_at": combo_payload.get("created_at"),
                "updated_at": combo_payload.get("updated_at"),
            })

            summary_tab, roles_tab, experiments_tab, notes_tab = st.tabs(
                ["Summary", "Prompt Roles", "Experiments", "Notes & Metadata"]
            )

            with summary_tab:
                summary_left, summary_right = st.columns(2)
                with summary_left:
                    st.markdown("**Prompt Membership**")
                    st.dataframe(
                        [
                            {
                                "role": row.get("role"),
                                "prompt_id": row.get("prompt_id"),
                                "prompt_title": row.get("prompt_title"),
                                "prompt_type": row.get("prompt_type"),
                                "branch": row.get("branch"),
                                "word_count": row.get("word_count"),
                                "is_archived": row.get("is_archived"),
                            }
                            for row in prompt_role_rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                with summary_right:
                    st.markdown("**Dependency Health**")
                    st.json({
                        "safe_to_delete": dependency_report.get("safe_to_delete"),
                        "blockers": dependency_report.get("blockers", []),
                        "derived_combo_ids": dependency_report.get("derived_combo_ids", []),
                        "run_ids": dependency_report.get("run_ids", []),
                    })

            with roles_tab:
                st.markdown("**Prompt Roles**")
                if not prompt_role_rows:
                    st.caption("This combo currently has no prompt-role members.")
                else:
                    st.dataframe(
                        [
                            {
                                "role": row.get("role"),
                                "prompt_id": row.get("prompt_id"),
                                "prompt_title": row.get("prompt_title"),
                                "prompt_type": row.get("prompt_type"),
                                "tree_id": row.get("tree_id"),
                                "branch": row.get("branch"),
                                "tags": ", ".join(row.get("tags", [])),
                                "prompt_exists": row.get("prompt_exists"),
                                "is_archived": row.get("is_archived"),
                            }
                            for row in prompt_role_rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    prompt_option_map = {
                        (
                            f"{row.get('role')} | {row.get('prompt_id')}"
                            f" | {row.get('prompt_title') or ''}"
                        ): row.get("prompt_id")
                        for row in prompt_role_rows
                        if row.get("prompt_exists")
                    }
                    if prompt_option_map:
                        selected_prompt_label = st.selectbox(
                            "Inspect Member Prompt",
                            options=list(prompt_option_map.keys()),
                            key=f"combo_explorer_member_prompt_{selected_combo_id}",
                        )
                        st.markdown("**Prompt Preview**")
                        _render_prompt_inline_preview(
                            garden_root=garden_root,
                            prompt_id=prompt_option_map[selected_prompt_label],
                        )
                    else:
                        st.caption(
                            "No active prompt nodes are available for inline preview."
                        )

            with experiments_tab:
                st.markdown("**Experiment Membership**")
                if dependent_experiment_rows:
                    st.dataframe(
                        [
                            {
                                "id": row.get("id"),
                                "name": row.get("name"),
                                "status": row.get("status"),
                                "combo_count": row.get("combo_count"),
                                "tested_combo_count": row.get("tested_combo_count"),
                                "average_score": row.get("average_score"),
                                "normalized_artifact_count": row.get("normalized_artifact_count"),
                            }
                            for row in dependent_experiment_rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("This combo is not currently attached to any experiment.")

                st.markdown("**Derived Combos**")
                if derived_combo_rows:
                    st.dataframe(
                        [
                            {
                                "id": row.get("id"),
                                "title": row.get("title"),
                                "status": row.get("status"),
                                "test_status": row.get("test_status"),
                                "experiment_count": row.get("experiment_count"),
                            }
                            for row in derived_combo_rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No derived combos currently reference this combo.")

            with notes_tab:
                st.markdown("**Notes**")
                if combo_payload.get("notes"):
                    st.write(combo_payload.get("notes"))
                else:
                    st.caption("No notes are stored for this combo yet.")

                meta_left, meta_right = st.columns(2)
                with meta_left:
                    st.markdown("**Stats**")
                    st.json(combo_payload.get("stats") or {})
                with meta_right:
                    st.markdown("**Metadata**")
                    st.json(combo_payload.get("metadata") or {})

                with st.expander("Full Dependency Report"):
                    st.code(
                        json.dumps(
                            dependency_report,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        language="json",
                    )

        st.markdown("### Combo Actions")
        if st.button(
            "Create Combo",
            use_container_width=True,
            key="combo_authoring_create_combo",
        ):
            _open_authoring_mode(
                _AUTHORING_CREATE_COMBO,
                source_tab="Combo Explorer",
            )


def _render_experiment_command_builder(
    garden_root: str,
    experiment: dict[str, Any],
    composition_rows: Sequence[dict[str, Any]],
) -> None:
    experiment_id = str(experiment.get("id") or "")
    attached_combo_rows = [
        row for row in composition_rows
        if row.get("combo_id")
    ]
    combo_option_map = {
        (
            f"{row.get('position')}. {row.get('combo_id')}"
            + (
                f" | {row.get('combo_title')}"
                if row.get("combo_title") else " | missing combo"
            )
        ): str(row.get("combo_id"))
        for row in attached_combo_rows
    }

    case_set_rows = list_case_set_rows(Path(garden_root))
    preset_options = [
        row["reference"]
        for row in case_set_rows
    ]
    preset_options.append("(custom)")
    selected_preset = st.selectbox(
        "Case Set Preset",
        options=preset_options,
        key=f"command_builder_case_set_preset_{experiment_id}",
    )
    if selected_preset == "(custom)":
        case_set_reference = st.text_input(
            "Case Set Reference",
            value=case_set_rows[0]["reference"] if case_set_rows else "",
            help="Workspace case-set id, direct file path, or built-in default id.",
            key=f"command_builder_case_set_custom_{experiment_id}",
        ).strip()
    else:
        case_set_reference = selected_preset

    try:
        case_set_payload = load_case_set_payload(
            Path(garden_root),
            case_set_reference or None,
        )
        case_set_error = None
    except Exception as exc:
        case_set_payload = None
        case_set_error = str(exc)

    resolved_case_path = resolve_case_set_path(
        Path(garden_root),
        case_set_reference or None,
    )
    case_rows = (
        list(case_set_payload.get("cases", []))
        if case_set_payload else []
    )
    case_option_map = {
        _case_option_label(case_row): str(case_row.get("id"))
        for case_row in case_rows
        if case_row.get("id")
    }

    config_col, preview_col = st.columns([1, 1], gap="large")

    with config_col:
        st.markdown("**Execution Settings**")
        model = st.text_input(
            "Model",
            value="phi4-mini",
            key=f"command_builder_model_{experiment_id}",
        ).strip() or "phi4-mini"
        bot_variant = st.selectbox(
            "Bot Variant",
            options=["rag", "legacy"],
            key=f"command_builder_bot_variant_{experiment_id}",
        )

        use_fewshot = st.checkbox(
            "Use Few-Shot Prompt",
            value=True,
            key=f"command_builder_use_fewshot_{experiment_id}",
        )
        fewshot_id = None
        if use_fewshot:
            fewshot_id = st.text_input(
                "Few-Shot Prompt ID",
                value="fsh_000002",
                key=f"command_builder_fewshot_id_{experiment_id}",
            ).strip() or "fsh_000002"

        rag_widget_value = st.checkbox(
            "Use RAG",
            value=True,
            disabled=(bot_variant == "legacy"),
            key=f"command_builder_use_rag_{experiment_id}",
        )
        use_rag = bot_variant == "rag" and rag_widget_value
        run_mode = st.selectbox(
            "Run Mode",
            options=["missing", "failed", "all"],
            key=f"command_builder_run_mode_{experiment_id}",
        )

        advanced_col_1, advanced_col_2 = st.columns(2)
        with advanced_col_1:
            rag_k = st.number_input(
                "RAG K",
                min_value=1,
                value=4,
                step=1,
                key=f"command_builder_rag_k_{experiment_id}",
            )
            candidate_k = st.number_input(
                "Candidate K",
                min_value=1,
                value=12,
                step=1,
                key=f"command_builder_candidate_k_{experiment_id}",
            )
        with advanced_col_2:
            max_context_chars = st.number_input(
                "Max Context Chars",
                min_value=500,
                value=6500,
                step=100,
                key=f"command_builder_max_context_{experiment_id}",
            )
            max_history_messages = st.number_input(
                "Max History Messages",
                min_value=1,
                value=12,
                step=1,
                key=f"command_builder_max_history_{experiment_id}",
            )

        st.markdown("**Combo Filters**")
        st.caption(
            "`only_*` filters take precedence over matching `skip_*` filters."
        )
        only_combo_labels = st.multiselect(
            "Only Attached Combos",
            options=list(combo_option_map.keys()),
            key=f"command_builder_only_combos_{experiment_id}",
        )
        skip_combo_labels = st.multiselect(
            "Skip Attached Combos",
            options=list(combo_option_map.keys()),
            key=f"command_builder_skip_combos_{experiment_id}",
        )

        st.markdown("**Case Filters**")
        if case_set_error:
            st.error(case_set_error)
        elif not case_option_map:
            st.warning("The selected case set currently exposes no usable case ids.")
        else:
            only_case_labels = st.multiselect(
                "Only Cases",
                options=list(case_option_map.keys()),
                key=f"command_builder_only_cases_{experiment_id}",
            )
            skip_case_labels = st.multiselect(
                "Skip Cases",
                options=list(case_option_map.keys()),
                key=f"command_builder_skip_cases_{experiment_id}",
            )
        if case_set_error or not case_option_map:
            only_case_labels = []
            skip_case_labels = []

        with st.expander("Available Case Sets", expanded=False):
            st.dataframe(
                case_set_rows,
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Selected Case Set Preview", expanded=False):
            if case_set_payload:
                st.caption(
                    "Source: "
                    + (
                        str(resolved_case_path)
                        if resolved_case_path is not None
                        else "built-in default payload"
                    )
                )
                st.json({
                    "id": case_set_payload.get("id"),
                    "name": case_set_payload.get("name"),
                    "case_count": len(case_rows),
                })
                st.dataframe(
                    [
                        {
                            "id": case_row.get("id"),
                            "question": case_row.get("question"),
                        }
                        for case_row in case_rows[:20]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    only_case_ids = [
        case_option_map[label]
        for label in only_case_labels
    ]
    skip_case_ids = [
        case_option_map[label]
        for label in skip_case_labels
    ]
    only_combo_ids = [
        combo_option_map[label]
        for label in only_combo_labels
    ]
    skip_combo_ids = [
        combo_option_map[label]
        for label in skip_combo_labels
    ]

    full_config = _build_runner_config(
        garden_root,
        experiment_id,
        model=model,
        bot_variant=bot_variant,
        fewshot_id=fewshot_id,
        use_rag=use_rag,
        rag_k=int(rag_k),
        candidate_k=int(candidate_k),
        max_context_chars=int(max_context_chars),
        max_history_messages=int(max_history_messages),
        case_set=case_set_reference or None,
        only_case_ids=(),
        skip_case_ids=(),
        only_combo_ids=(),
        skip_combo_ids=(),
        run_mode=run_mode,
        dry_run=False,
    )
    filtered_config = _build_runner_config(
        garden_root,
        experiment_id,
        model=model,
        bot_variant=bot_variant,
        fewshot_id=fewshot_id,
        use_rag=use_rag,
        rag_k=int(rag_k),
        candidate_k=int(candidate_k),
        max_context_chars=int(max_context_chars),
        max_history_messages=int(max_history_messages),
        case_set=case_set_reference or None,
        only_case_ids=only_case_ids,
        skip_case_ids=skip_case_ids,
        only_combo_ids=only_combo_ids,
        skip_combo_ids=skip_combo_ids,
        run_mode=run_mode,
        dry_run=False,
    )
    preview_config = _build_runner_config(
        garden_root,
        experiment_id,
        model=model,
        bot_variant=bot_variant,
        fewshot_id=fewshot_id,
        use_rag=use_rag,
        rag_k=int(rag_k),
        candidate_k=int(candidate_k),
        max_context_chars=int(max_context_chars),
        max_history_messages=int(max_history_messages),
        case_set=case_set_reference or None,
        only_case_ids=only_case_ids,
        skip_case_ids=skip_case_ids,
        only_combo_ids=only_combo_ids,
        skip_combo_ids=skip_combo_ids,
        run_mode=run_mode,
        dry_run=True,
    )
    filters_active = any([
        only_case_ids,
        skip_case_ids,
        only_combo_ids,
        skip_combo_ids,
    ])

    full_command = build_runner_command(
        full_config,
        include_filters=False,
        dry_run=False,
    )
    filtered_command = build_runner_command(
        filtered_config,
        include_filters=True,
        dry_run=False,
    )
    preview_command = build_runner_command(
        preview_config,
        include_filters=True,
        dry_run=True,
    )

    plan_payload = None
    plan_error = None
    try:
        plan_payload = plan_prompt_experiment(preview_config)
    except Exception as exc:
        plan_error = str(exc)

    with preview_col:
        st.markdown("**Generated Commands**")
        st.markdown("Full experiment command")
        st.code(full_command, language="powershell")

        if filters_active:
            st.markdown("Filtered command")
            st.code(filtered_command, language="powershell")
        else:
            st.caption(
                "No case or combo filters are active, so the filtered command matches the full experiment command."
            )

        st.markdown("Dry-run preview command")
        st.code(preview_command, language="powershell")

        st.markdown("**Dry-Run Preview**")
        if plan_error:
            st.error(plan_error)
        elif plan_payload is not None:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Selected Cases", plan_payload.get("case_count", 0))
            metric_cols[1].metric("Selected Combos", plan_payload.get("combo_count", 0))
            metric_cols[2].metric("Targets To Run", plan_payload.get("target_count", 0))
            metric_cols[3].metric(
                "Skipped Existing",
                plan_payload.get("skipped_existing_count", 0),
            )

            st.json({
                "experiment_id": plan_payload.get("experiment_id"),
                "case_set_id": plan_payload.get("case_set_id"),
                "case_filters_active": plan_payload.get("case_filters_active"),
                "raw_scope_dir": plan_payload.get("raw_scope_dir"),
                "normalized_scope_dir": plan_payload.get("normalized_scope_dir"),
                "report_scope_dir": plan_payload.get("report_scope_dir"),
                "execution_signature": (
                    (plan_payload.get("execution") or {}).get("signature")
                ),
            })
            st.dataframe(
                plan_payload.get("targets_preview", []),
                use_container_width=True,
                hide_index=True,
            )


def _render_experiment_builder_and_editor(
    garden_root: str,
    experiment_rows: list[dict[str, Any]],
    combo_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Experiment Builder & Editor")
    _consume_experiment_flash_message()

    status_options = _experiment_status_options(experiment_rows)
    combo_option_map = {
        _combo_option_label(row): row["id"]
        for row in combo_rows
    }

    garden = PromptGarden(garden_root)
    garden.init()

    explorer_col, detail_col = st.columns([1, 2], gap="large")

    with explorer_col:
        search_query = st.text_input(
            "Search Experiment",
            help="Search by id, name, status, goal, hypothesis, tags, notes, or combo ids.",
            key="experiment_explorer_search",
        ).strip().lower()
        include_archived = st.checkbox(
            "Include Archived Experiments",
            value=True,
            key="experiment_explorer_include_archived",
        )

        filtered_experiment_rows = []
        for row in experiment_rows:
            if not include_archived and row.get("is_archived"):
                continue
            if not _matches_experiment_search(row, search_query):
                continue
            filtered_experiment_rows.append(row)

        st.caption(f"Filtered experiments: {len(filtered_experiment_rows)}")
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "combo_count": row.get("combo_count"),
                    "tested_combo_count": row.get("tested_combo_count"),
                    "missing_combo_count": row.get("missing_combo_count"),
                    "is_archived": row.get("is_archived"),
                }
                for row in filtered_experiment_rows[:20]
            ],
            use_container_width=True,
            hide_index=True,
        )

        selection_rows = sorted(
            filtered_experiment_rows,
            key=lambda row: (
                row.get("created_at") or "",
                row.get("id") or "",
            ),
            reverse=True,
        )
        selected_experiment_id = st.session_state.get(
            _EXPERIMENT_SELECTION_KEY
        )
        if selected_experiment_id not in {
            row.get("id")
            for row in selection_rows
        }:
            selected_experiment_id = (
                selection_rows[0]["id"]
                if selection_rows else None
            )

        if selection_rows:
            lookup_mode = st.radio(
                "Load Experiment",
                options=["By Name", "By ID"],
                horizontal=True,
                key="experiment_lookup_mode",
            )
            if lookup_mode == "By ID":
                ordered_rows = sorted(
                    selection_rows,
                    key=lambda row: (
                        row.get("id") or "",
                    ),
                )
                label_to_id = {
                    row["id"]: row["id"]
                    for row in ordered_rows
                }
                default_label = selected_experiment_id
            else:
                ordered_rows = sorted(
                    selection_rows,
                    key=lambda row: (
                        (row.get("name") or "").lower(),
                        row.get("id") or "",
                    ),
                )
                label_to_id = {
                    f"{row.get('name') or '(unnamed)'} | {row['id']}": row["id"]
                    for row in ordered_rows
                }
                default_label = next(
                    label
                    for label, experiment_id in label_to_id.items()
                    if experiment_id == selected_experiment_id
                )

            selection_labels = list(label_to_id.keys())
            selected_label = st.selectbox(
                "Experiment Detail",
                options=selection_labels,
                index=selection_labels.index(default_label),
                key="experiment_builder_selected_experiment",
            )
            selected_experiment_id = label_to_id[selected_label]
            st.session_state[_EXPERIMENT_SELECTION_KEY] = (
                selected_experiment_id
            )
        else:
            selected_experiment_id = None

        with st.expander(
            "Create Experiment",
            expanded=not experiment_rows,
        ):
            with st.form("experiment_create_form"):
                create_name = st.text_input("Name")
                create_goal = st.text_area(
                    "Goal",
                    height=120,
                )
                create_hypothesis = st.text_area(
                    "Hypothesis",
                    height=120,
                )
                create_notes = st.text_area(
                    "Notes",
                    height=140,
                )
                create_tags_text = st.text_input(
                    "Tags",
                    help="Comma-separated tags.",
                )
                create_status = st.selectbox(
                    "Initial Status",
                    options=status_options,
                    index=status_options.index("planned"),
                )
                create_combo_labels = st.multiselect(
                    "Attach Combos Now",
                    options=list(combo_option_map.keys()),
                    help="Optional. You can also attach combos later from the editor.",
                )
                create_submitted = st.form_submit_button(
                    "Create Experiment",
                    use_container_width=True,
                )

            if create_submitted:
                try:
                    created = garden.create_experiment(
                        name=create_name,
                        goal=create_goal,
                        hypothesis=create_hypothesis,
                        notes=create_notes,
                        tags=_parse_tag_text(create_tags_text),
                        status=create_status,
                        combo_ids=[
                            combo_option_map[label]
                            for label in create_combo_labels
                        ],
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_experiment_refresh(
                        f"Experiment `{created['id']}` created.",
                        selected_experiment_id=created["id"],
                    )

    with detail_col:
        if not experiment_rows:
            st.info(
                "No experiments exist yet. Create one from the form on the left."
            )
            return
        if not selected_experiment_id:
            st.info("No experiments match the current filters.")
            return

        composition_bundle = load_experiment_composition_bundle(
            garden_root,
            selected_experiment_id,
        )
        experiment = composition_bundle.get("experiment") or {}
        summary = composition_bundle.get("summary") or {}
        composition_rows = composition_bundle.get("combo_rows") or []

        st.markdown("### Experiment Detail")
        st.caption(
            f"{experiment.get('id')} | "
            f"status={experiment.get('status')} | "
            f"combos={summary.get('combo_count', 0)} | "
            f"tested={summary.get('tested_combo_count', 0)} | "
            f"missing={summary.get('missing_combo_count', 0)}"
        )

        overview_tab, metadata_tab, combo_manager_tab, command_tab, composition_tab = st.tabs(
            [
                "Overview",
                "Metadata",
                "Combo Manager",
                "Command Builder",
                "Composition",
            ]
        )

        with overview_tab:
            average_score = summary.get("average_score")
            metric_cols = st.columns(5)
            metric_cols[0].metric("Combos", summary.get("combo_count", 0))
            metric_cols[1].metric(
                "Tested Combos",
                summary.get("tested_combo_count", 0),
            )
            metric_cols[2].metric(
                "Untested Combos",
                summary.get("untested_combo_count", 0),
            )
            metric_cols[3].metric(
                "Missing Combos",
                summary.get("missing_combo_count", 0),
            )
            metric_cols[4].metric(
                "Average Score",
                "-" if average_score is None else average_score,
            )

            st.json({
                "id": experiment.get("id"),
                "name": experiment.get("name"),
                "status": experiment.get("status"),
                "tags": experiment.get("tags", []),
                "created_at": experiment.get("created_at"),
                "updated_at": experiment.get("updated_at"),
                "combo_ids": experiment.get("combo_ids", []),
            })

            info_left, info_right = st.columns(2)
            with info_left:
                st.markdown("**Goal**")
                st.write(experiment.get("goal") or "(empty)")
                st.markdown("**Hypothesis**")
                st.write(experiment.get("hypothesis") or "(empty)")
            with info_right:
                st.markdown("**Notes**")
                st.write(experiment.get("notes") or "(empty)")
                st.markdown("**Summary Payload**")
                st.json(experiment.get("summary") or {})

        with metadata_tab:
            with st.form(
                f"experiment_metadata_form_{selected_experiment_id}"
            ):
                edit_name = st.text_input(
                    "Name",
                    value=experiment.get("name") or "",
                )
                edit_goal = st.text_area(
                    "Goal",
                    value=experiment.get("goal") or "",
                    height=120,
                )
                edit_hypothesis = st.text_area(
                    "Hypothesis",
                    value=experiment.get("hypothesis") or "",
                    height=120,
                )
                edit_notes = st.text_area(
                    "Notes",
                    value=experiment.get("notes") or "",
                    height=180,
                )
                edit_tags = st.text_input(
                    "Tags",
                    value=", ".join(experiment.get("tags", [])),
                    help="Comma-separated tags.",
                )
                current_status = experiment.get("status") or "planned"
                editable_status_options = list(status_options)
                if current_status not in editable_status_options:
                    editable_status_options.append(current_status)
                edit_status = st.selectbox(
                    "Status",
                    options=editable_status_options,
                    index=editable_status_options.index(current_status),
                )
                metadata_submitted = st.form_submit_button(
                    "Save Experiment Metadata",
                    use_container_width=True,
                )

            if metadata_submitted:
                try:
                    garden.update_experiment_metadata(
                        selected_experiment_id,
                        name=edit_name,
                        goal=edit_goal,
                        hypothesis=edit_hypothesis,
                        notes=edit_notes,
                        tags=_parse_tag_text(edit_tags),
                        status=edit_status,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_experiment_refresh(
                        f"Experiment `{selected_experiment_id}` updated.",
                        selected_experiment_id=selected_experiment_id,
                    )

        with combo_manager_tab:
            attach_col, detach_col = st.columns(2, gap="large")

            with attach_col:
                st.markdown("**Attach Combos**")
                available_combo_rows = [
                    row for row in combo_rows
                    if row.get("id") not in set(experiment.get("combo_ids", []))
                ]
                available_combo_option_map = {
                    _combo_option_label(row): row["id"]
                    for row in available_combo_rows
                }
                if not available_combo_option_map:
                    st.caption(
                        "All known combos are already attached to this experiment."
                    )
                else:
                    with st.form(
                        f"experiment_attach_form_{selected_experiment_id}"
                    ):
                        attach_combo_labels = st.multiselect(
                            "Available Combos",
                            options=list(available_combo_option_map.keys()),
                        )
                        attach_role = st.text_input(
                            "Link Role",
                            value="candidate",
                        )
                        attach_notes = st.text_area(
                            "Link Notes",
                            height=120,
                        )
                        attach_submitted = st.form_submit_button(
                            "Attach Selected Combos",
                            use_container_width=True,
                        )

                    if attach_submitted:
                        try:
                            garden.attach_combos_to_experiment(
                                experiment_id=selected_experiment_id,
                                combo_ids=[
                                    available_combo_option_map[label]
                                    for label in attach_combo_labels
                                ],
                                role=attach_role.strip() or "candidate",
                                notes=attach_notes,
                            )
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            _queue_experiment_refresh(
                                f"Attached {len(attach_combo_labels)} combo(s) to `{selected_experiment_id}`.",
                                selected_experiment_id=selected_experiment_id,
                            )

            with detach_col:
                st.markdown("**Detach Combo**")
                if not composition_rows:
                    st.caption(
                        "This experiment currently has no attached combos."
                    )
                else:
                    detach_option_map = {
                        (
                            f"{row.get('position')}. {row.get('combo_id')}"
                            + (
                                f" | {row.get('combo_title')}"
                                if row.get("combo_title") else " | missing combo"
                            )
                        ): row.get("combo_id")
                        for row in composition_rows
                    }
                    selected_detach_label = st.selectbox(
                        "Attached Combo",
                        options=list(detach_option_map.keys()),
                        key=f"experiment_detach_combo_{selected_experiment_id}",
                    )
                    selected_detach_combo_id = detach_option_map[
                        selected_detach_label
                    ]
                    detach_preview = garden.preview_detach_combo_from_experiment(
                        selected_experiment_id,
                        selected_detach_combo_id,
                    )
                    st.json({
                        "safe_to_detach": detach_preview.get("safe_to_detach"),
                        "combo_exists": detach_preview.get("combo_exists"),
                        "result_count": detach_preview.get("result_count"),
                        "compact_result_row_count": detach_preview.get("compact_result_row_count"),
                        "run_ids": detach_preview.get("run_ids", []),
                        "blockers": detach_preview.get("blockers", []),
                    })
                    if detach_preview.get("blockers"):
                        st.warning(
                            "Detach is blocked until these dependencies are cleared: "
                            + ", ".join(detach_preview.get("blockers", []))
                        )
                    if st.button(
                        "Detach Selected Combo",
                        use_container_width=True,
                        disabled=not detach_preview.get("safe_to_detach"),
                        key=f"detach_combo_button_{selected_experiment_id}",
                    ):
                        try:
                            garden.detach_combo_from_experiment(
                                selected_experiment_id,
                                selected_detach_combo_id,
                            )
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            _queue_experiment_refresh(
                                f"Detached `{selected_detach_combo_id}` from `{selected_experiment_id}`.",
                                selected_experiment_id=selected_experiment_id,
                            )

        with command_tab:
            _render_experiment_command_builder(
                garden_root=garden_root,
                experiment=experiment,
                composition_rows=composition_rows,
            )

        with composition_tab:
            _render_experiment_composition_block(composition_bundle)


def _render_prompt_maintenance_panel(
    garden_root: str,
    prompt_bundle: dict[str, Any],
    selected_prompt_id: str | None,
) -> None:
    st.markdown("### Prompt Safety")
    if not selected_prompt_id:
        st.info(
            "Select a prompt above to inspect delete blockers and the prompt danger zone."
        )
        return

    garden = PromptGarden(garden_root)
    garden.init()

    prompt_payload = prompt_bundle.get("prompt") or {}
    dependency_summary = prompt_bundle.get("dependency_summary") or {}
    dependency_report = {
        "id": dependency_summary.get("id"),
        "safe_to_delete": dependency_summary.get("safe_to_delete"),
        "blockers": dependency_summary.get("blockers", []),
        "child_prompt_ids": dependency_summary.get("child_prompt_ids", []),
        "combo_ids": dependency_summary.get("combo_ids", []),
        "experiment_ids": dependency_summary.get("experiment_ids", []),
        "is_archived": dependency_summary.get("is_archived"),
    }
    usage_section = dependency_summary.get("usage") or {}
    usage_rows = list(usage_section.get("rows", []))
    delete_safety = dependency_summary.get("delete_safety") or {}
    blocker_rows = list(delete_safety.get("blocker_rows") or [])
    recommended_actions = list(
        delete_safety.get("recommended_actions") or []
    )
    dependent_combo_rows = list(
        prompt_bundle.get("dependent_combo_rows") or []
    )
    dependent_experiment_rows = list(
        prompt_bundle.get("dependent_experiment_rows") or []
    )

    st.caption(
        f"{prompt_payload.get('id')} | "
        f"type={prompt_payload.get('type')} | "
        f"tree={prompt_payload.get('tree_id')} | "
        f"archived={prompt_payload.get('is_archived')}"
    )

    report_col, action_col = st.columns(2, gap="large")
    with report_col:
        st.markdown("**Where This Prompt Is Used**")
        metric_cols = st.columns(3)
        usage_row_by_code = {
            row.get("code"): row
            for row in usage_rows
        }
        metric_cols[0].metric(
            "Child Prompts",
            (usage_row_by_code.get("child_prompts") or {}).get("count", 0),
        )
        metric_cols[1].metric(
            "Combos",
            (usage_row_by_code.get("combos") or {}).get("count", 0),
        )
        metric_cols[2].metric(
            "Experiments",
            (usage_row_by_code.get("experiments") or {}).get("count", 0),
        )
        st.caption(usage_section.get("headline") or "")

        child_ids = (
            (usage_row_by_code.get("child_prompts") or {}).get("ids")
            or []
        )
        if child_ids:
            st.write(
                "Child prompts: "
                + _dependency_id_preview(child_ids)
            )

        if dependent_combo_rows:
            combo_labels = [
                f"{row.get('id')} ({row.get('title') or 'untitled'})"
                for row in dependent_combo_rows[:5]
            ]
            hidden_count = max(len(dependent_combo_rows) - len(combo_labels), 0)
            suffix = f" +{hidden_count} more" if hidden_count else ""
            st.write(
                "Combos: "
                + ", ".join(combo_labels)
                + suffix
            )
        else:
            st.write("Combos: -")

        if dependent_experiment_rows:
            experiment_labels = [
                f"{row.get('id')} ({row.get('name') or 'unnamed'})"
                for row in dependent_experiment_rows[:5]
            ]
            hidden_count = max(
                len(dependent_experiment_rows) - len(experiment_labels),
                0,
            )
            suffix = f" +{hidden_count} more" if hidden_count else ""
            st.write(
                "Experiments: "
                + ", ".join(experiment_labels)
                + suffix
            )
        else:
            st.write("Experiments: -")

        st.markdown("**Delete Readiness**")
        if delete_safety.get("status") == "safe":
            st.success(
                delete_safety.get("headline")
                or "This prompt can be deleted safely."
            )
        else:
            st.warning(
                delete_safety.get("headline")
                or "Delete is currently blocked."
            )

        if delete_safety.get("summary"):
            st.caption(str(delete_safety.get("summary")))

        with st.expander("Debug Dependency JSON", expanded=False):
            st.json(dependency_summary or dependency_report)

    with action_col:
        st.markdown("**Danger Zone**")
        st.error(
            "Permanent delete removes the prompt file and node record from Prompt Garden."
        )
        st.info(
            "Use `Archive Prompt` above when you want to retire a prompt without losing its lineage or historical context."
        )

        if delete_safety.get("status") == "safe":
            st.success(
                delete_safety.get("headline")
                or "This prompt can be deleted safely."
            )
        else:
            st.warning(
                delete_safety.get("headline")
                or "Delete is currently blocked."
            )

        if delete_safety.get("summary"):
            st.caption(str(delete_safety.get("summary")))

        if recommended_actions:
            st.markdown("**What needs to happen first**")
            for action in recommended_actions:
                st.write(action)

        if blocker_rows:
            st.markdown("**Delete blockers**")
            for blocker in blocker_rows:
                st.write(
                    f"{blocker.get('label')}: {blocker.get('message')}"
                )
                blocker_ids = blocker.get("ids") or []
                if blocker_ids:
                    st.caption(
                        "Affected ids: "
                        + _dependency_id_preview(blocker_ids)
                    )
                related_experiment_ids = (
                    blocker.get("related_experiment_ids") or []
                )
                if related_experiment_ids:
                    st.caption(
                        "Related experiments: "
                        + _dependency_id_preview(related_experiment_ids)
                    )
        else:
            st.caption(
                "No blockers remain. This prompt is archived and no active dependencies still reference it."
            )

        st.markdown("**Delete Prompt Permanently**")
        delete_confirmed = st.checkbox(
            "I understand this removes the prompt file permanently.",
            key=f"cleanup_prompt_confirm_{selected_prompt_id}",
        )
        if not dependency_report.get("safe_to_delete"):
            st.caption(
                "Delete stays disabled until every blocker above is resolved."
            )
        elif not delete_confirmed:
            st.caption(
                "Tick the confirmation box to enable permanent delete."
            )
        if st.button(
            "Delete Prompt",
            use_container_width=True,
            disabled=(
                not dependency_report.get("safe_to_delete")
                or not delete_confirmed
            ),
            key=f"cleanup_prompt_delete_{selected_prompt_id}",
        ):
            try:
                garden.delete_prompt(selected_prompt_id)
            except Exception as exc:
                st.error(str(exc))
            else:
                _queue_prompt_workspace_refresh(
                    f"Prompt `{selected_prompt_id}` deleted permanently.",
                    clear_selected_prompt=True,
                )


def _render_cleanup_combo_panel(
    garden: PromptGarden,
    combo_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Combo Cleanup")
    if not combo_rows:
        st.info("No combos are available for cleanup.")
        return

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        search_query = st.text_input(
            "Search Combo",
            help="Search by id, title, status, test status, kind, tags, notes, or prompt ids.",
            key="cleanup_combo_search",
        ).strip().lower()
        include_archived = st.checkbox(
            "Include Archived Combos",
            value=True,
            key="cleanup_combo_include_archived",
        )
        filtered_rows = [
            row for row in combo_rows
            if _matches_combo_search(row, search_query)
            and (include_archived or not row.get("is_archived"))
        ]
        st.caption(f"Combo candidates: {len(filtered_rows)}")
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "test_status": row.get("test_status"),
                    "experiment_count": row.get("experiment_count"),
                    "result_experiment_count": row.get("result_experiment_count"),
                    "is_archived": row.get("is_archived"),
                }
                for row in filtered_rows[:20]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered_rows:
            selected_combo_id = None
        else:
            option_map = {
                _combo_option_label(row): row["id"]
                for row in filtered_rows
            }
            selected_label = st.selectbox(
                "Combo To Clean Up",
                options=list(option_map.keys()),
                key="cleanup_combo_selected_id",
            )
            selected_combo_id = option_map[selected_label]

    with right_col:
        if not filtered_rows or not selected_combo_id:
            st.info("No combos match the current cleanup filters.")
            return

        combo_bundle = load_combo_explorer_bundle(
            str(garden.root),
            selected_combo_id,
        )
        combo_payload = combo_bundle.get("combo") or {}
        dependency_report = garden.inspect_combo_dependencies(
            selected_combo_id
        )

        st.caption(
            f"{combo_payload.get('id')} | "
            f"status={combo_payload.get('status')} | "
            f"test={combo_payload.get('test_status')} | "
            f"archived={combo_payload.get('is_archived')}"
        )
        st.json({
            "title": combo_payload.get("title"),
            "tags": combo_payload.get("tags", []),
            "prompt_ids": combo_payload.get("prompt_ids", {}),
            "experiment_ids": dependency_report.get("experiment_ids", []),
            "result_experiment_ids": dependency_report.get("result_experiment_ids", []),
            "derived_combo_ids": dependency_report.get("derived_combo_ids", []),
        })

        report_col, action_col = st.columns(2, gap="large")
        with report_col:
            st.markdown("**Dependency Preview**")
            st.json(dependency_report)
            if dependency_report.get("safe_to_delete"):
                st.success("This combo can be deleted safely.")
            else:
                st.warning(
                    "Delete is currently blocked by: "
                    + ", ".join(dependency_report.get("blockers", []))
                )
        with action_col:
            st.markdown("**Archive Combo**")
            with st.form(f"cleanup_combo_archive_{selected_combo_id}"):
                archive_reason = st.text_area(
                    "Archive Reason",
                    height=100,
                    help="Optional note that will be stored in combo metadata.",
                )
                archive_submitted = st.form_submit_button(
                    "Archive Combo",
                    use_container_width=True,
                    disabled=bool(dependency_report.get("is_archived")),
                )
            if archive_submitted:
                try:
                    garden.archive_combo(
                        selected_combo_id,
                        reason=archive_reason,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_cleanup_refresh(
                        f"Combo `{selected_combo_id}` archived."
                    )

            st.markdown("**Delete Combo Permanently**")
            delete_confirmed = st.checkbox(
                "I understand this removes the combo and its links permanently.",
                key=f"cleanup_combo_confirm_{selected_combo_id}",
            )
            if st.button(
                "Delete Combo",
                use_container_width=True,
                disabled=(
                    not dependency_report.get("safe_to_delete")
                    or not delete_confirmed
                ),
                key=f"cleanup_combo_delete_{selected_combo_id}",
            ):
                try:
                    garden.delete_combo(selected_combo_id)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_cleanup_refresh(
                        f"Combo `{selected_combo_id}` deleted permanently."
                    )


def _render_cleanup_experiment_panel(
    garden: PromptGarden,
    experiment_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Experiment Cleanup")
    if not experiment_rows:
        st.info("No experiments are available for cleanup.")
        return

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        search_query = st.text_input(
            "Search Experiment",
            help="Search by id, name, status, goal, hypothesis, tags, notes, or combo ids.",
            key="cleanup_experiment_search",
        ).strip().lower()
        include_archived = st.checkbox(
            "Include Archived Experiments",
            value=True,
            key="cleanup_experiment_include_archived",
        )
        filtered_rows = [
            row for row in experiment_rows
            if _matches_experiment_search(row, search_query)
            and (include_archived or not row.get("is_archived"))
        ]
        st.caption(f"Experiment candidates: {len(filtered_rows)}")
        st.dataframe(
            [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "combo_count": row.get("combo_count"),
                    "tested_combo_count": row.get("tested_combo_count"),
                    "normalized_artifact_count": row.get("normalized_artifact_count"),
                    "is_archived": row.get("is_archived"),
                }
                for row in filtered_rows[:20]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered_rows:
            selected_experiment_id = None
        else:
            option_map = {
                _experiment_option_label(row): row["id"]
                for row in filtered_rows
            }
            selected_label = st.selectbox(
                "Experiment To Clean Up",
                options=list(option_map.keys()),
                key="cleanup_experiment_selected_id",
            )
            selected_experiment_id = option_map[selected_label]

    with right_col:
        if not filtered_rows or not selected_experiment_id:
            st.info("No experiments match the current cleanup filters.")
            return

        composition_bundle = load_experiment_composition_bundle(
            str(garden.root),
            selected_experiment_id,
        )
        experiment = composition_bundle.get("experiment") or {}
        summary = composition_bundle.get("summary") or {}
        dependency_report = garden.inspect_experiment_dependencies(
            selected_experiment_id
        )

        st.caption(
            f"{experiment.get('id')} | "
            f"status={experiment.get('status')} | "
            f"archived={dependency_report.get('is_archived')}"
        )
        st.json({
            "name": experiment.get("name"),
            "tags": experiment.get("tags", []),
            "combo_ids": experiment.get("combo_ids", []),
            "tested_combo_count": summary.get("tested_combo_count"),
            "average_score": summary.get("average_score"),
            "final_result_text": experiment.get("final_result_text"),
        })

        report_col, action_col = st.columns(2, gap="large")
        with report_col:
            st.markdown("**Dependency Preview**")
            st.json(dependency_report)
            if dependency_report.get("safe_to_delete"):
                st.success("This experiment can be deleted safely.")
            else:
                st.warning(
                    "Delete is currently blocked by: "
                    + ", ".join(dependency_report.get("blockers", []))
                )
        with action_col:
            st.markdown("**Archive Experiment**")
            with st.form(f"cleanup_experiment_archive_{selected_experiment_id}"):
                archive_reason = st.text_area(
                    "Archive Reason",
                    height=100,
                    help="Optional note that will be stored in experiment metadata.",
                )
                archive_submitted = st.form_submit_button(
                    "Archive Experiment",
                    use_container_width=True,
                    disabled=bool(dependency_report.get("is_archived")),
                )
            if archive_submitted:
                try:
                    garden.archive_experiment(
                        selected_experiment_id,
                        reason=archive_reason,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_cleanup_refresh(
                        f"Experiment `{selected_experiment_id}` archived."
                    )

            st.markdown("**Delete Experiment Permanently**")
            delete_confirmed = st.checkbox(
                "I understand this removes the experiment record and related links permanently.",
                key=f"cleanup_experiment_confirm_{selected_experiment_id}",
            )
            if st.button(
                "Delete Experiment",
                use_container_width=True,
                disabled=(
                    not dependency_report.get("safe_to_delete")
                    or not delete_confirmed
                ),
                key=f"cleanup_experiment_delete_{selected_experiment_id}",
            ):
                try:
                    garden.delete_experiment(selected_experiment_id)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _queue_cleanup_refresh(
                        f"Experiment `{selected_experiment_id}` deleted permanently."
                    )


def _render_cleanup_workspace(
    garden_root: str,
    combo_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
) -> None:
    st.markdown("### Cleanup")
    st.warning(
        "Archive before delete whenever possible. Permanent delete only becomes available "
        "when the dependency preview reports `safe_to_delete=true`."
    )
    st.info(
        "Prompt inspection, archive, and delete now live only in Prompt Workspace. "
        "Cleanup is reserved for combo and experiment cleanup."
    )
    st.caption(
        "Need to retire or delete a prompt? Open Prompt Workspace to review its text, usage, and danger zone in one place."
    )
    if st.button(
        "Open Prompt Workspace",
        use_container_width=True,
        key="cleanup_open_prompt_workspace",
    ):
        _queue_control_section_redirect("Prompt Workspace")
        st.rerun()
    _consume_cleanup_flash_message()

    garden = PromptGarden(garden_root)
    garden.init()

    combo_tab, experiment_tab = st.tabs(
        ["Combo Cleanup", "Experiment Cleanup"]
    )

    with combo_tab:
        _render_cleanup_combo_panel(
            garden=garden,
            combo_rows=combo_rows,
        )

    with experiment_tab:
        _render_cleanup_experiment_panel(
            garden=garden,
            experiment_rows=experiment_rows,
        )


def _render_experiment_composition_block(
    composition_bundle: dict[str, Any],
) -> None:
    experiment = composition_bundle.get("experiment") or {}
    summary = composition_bundle.get("summary") or {}
    combo_rows = composition_bundle.get("combo_rows") or []

    st.markdown("### Experiment Overview")
    st.json({
        "id": experiment.get("id"),
        "name": experiment.get("name"),
        "status": experiment.get("status"),
        "goal": experiment.get("goal"),
        "hypothesis": experiment.get("hypothesis"),
        "tags": experiment.get("tags", []),
        "combo_count": summary.get("combo_count"),
        "tested_combo_count": summary.get("tested_combo_count"),
        "untested_combo_count": summary.get("untested_combo_count"),
        "missing_combo_count": summary.get("missing_combo_count"),
        "average_score": summary.get("average_score"),
    })

    st.markdown("### Combo Composition")
    if not combo_rows:
        st.info("This experiment currently has no attached combos.")
        return

    for row in combo_rows:
        title = (
            f"{row.get('position')}. {row.get('combo_id')}"
            + (
                f" | {row.get('combo_title')}"
                if row.get("combo_title") else ""
            )
        )
        with st.expander(title):
            if not row.get("combo_exists"):
                st.warning(
                    "This combo id is attached in the experiment record, "
                    "but the combo is missing from the combo registry."
                )
                st.json({
                    "combo_id": row.get("combo_id"),
                    "result_status": row.get("result_status"),
                    "result_score": row.get("result_score"),
                })
                continue

            st.caption(
                f"status={row.get('combo_status')} | "
                f"test_status={row.get('combo_test_status')} | "
                f"score={row.get('combo_score')} | "
                f"kind={row.get('combo_kind')}"
            )
            st.dataframe(
                [
                    {
                        "role": member.get("role"),
                        "prompt_id": member.get("prompt_id"),
                        "prompt_title": member.get("prompt_title"),
                        "prompt_type": member.get("prompt_type"),
                        "branch": member.get("branch"),
                        "word_count": member.get("word_count"),
                        "is_archived": member.get("is_archived"),
                    }
                    for member in row.get("prompt_member_rows", [])
                ],
                use_container_width=True,
                hide_index=True,
            )
            if row.get("has_result"):
                st.markdown("**Latest Stored Result Summary**")
                st.write(row.get("result_text") or "(empty)")


def render_control_surface(
    garden_root: str,
    scopes: Sequence[dict[str, Any]],
) -> None:
    """Render the control landing surface for the new top-level shell."""

    index_bundle = load_garden_index_bundle(garden_root)
    prompt_rows = index_bundle.get("prompt_rows") or []
    combo_rows = index_bundle.get("combo_rows") or []
    experiment_rows = index_bundle.get("experiment_rows") or []

    st.subheader("Control")
    _consume_authoring_flash_message()
    st.caption(
        "Use this surface to orient yourself in the workspace, inspect what "
        "exists, manage experiments, and clean up clutter safely."
    )

    quick_col_1, quick_col_2, quick_col_3 = st.columns(3)
    with quick_col_1:
        st.markdown("**1. Survey the workspace**")
        st.write(
            "Check how many prompts, combos, experiments, and review scopes are present."
        )
    with quick_col_2:
        st.markdown("**2. Pick an experiment**")
        st.write(
            "Open one experiment composition preview to recall which combos and prompts it contains."
        )
    with quick_col_3:
        st.markdown("**3. Switch to analysis**")
        st.write(
            "Open the Analysis surface when you want to inspect normalized answers and review results."
        )

    pending_control_section = st.session_state.pop(
        _CONTROL_SECTION_REDIRECT_KEY,
        None,
    )
    if pending_control_section is not None:
        _set_control_section(pending_control_section)
    _set_control_section(
        st.session_state.get(_CONTROL_SECTION_KEY)
    )
    st.caption("Control sections")
    selected_control_section = st.radio(
        "Control sections",
        options=list(_CONTROL_SECTION_OPTIONS),
        key=_CONTROL_SECTION_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_control_section == "Prompt Workspace":
        _render_prompt_explorer(
            garden_root=garden_root,
            prompt_rows=prompt_rows,
        )

    elif selected_control_section == "Combo Explorer":
        _render_combo_explorer(
            garden_root=garden_root,
            combo_rows=combo_rows,
        )

    elif selected_control_section == "Experiments":
        _render_experiment_builder_and_editor(
            garden_root=garden_root,
            experiment_rows=experiment_rows,
            combo_rows=combo_rows,
        )

    elif selected_control_section == "Cleanup":
        _render_cleanup_workspace(
            garden_root=garden_root,
            combo_rows=combo_rows,
            experiment_rows=experiment_rows,
        )

    elif selected_control_section == "Review Scopes":
        st.markdown("### Review Scope Summary")
        if not scopes:
            st.info(
                "No normalized review scopes are available yet. "
                "Run `scripts/run_prompt_experiment.py` to create them."
            )
        else:
            st.dataframe(
                [
                    {
                        "scope": row.get("scope"),
                        "experiment_name": row.get("experiment_name"),
                        "experiment_status": row.get("experiment_status"),
                        "artifact_count": row.get("artifact_count"),
                        "tags": ", ".join(row.get("experiment_tags", [])),
                    }
                    for row in scopes
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "These scopes are what the Analysis surface reads when you inspect experiment outputs."
            )

    _render_shared_authoring_block(
        garden_root=garden_root,
        prompt_rows=prompt_rows,
    )

    with st.expander("What Comes Next"):
        st.write(
            "The next authoring steps will turn this shared bottom workspace into the real "
            "save surface for root prompts, prompt branches, and new combos."
        )


__all__ = [
    "build_workspace_status_summary",
    "render_control_surface",
    "render_workspace_status_header",
]
