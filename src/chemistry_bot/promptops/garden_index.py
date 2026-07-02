"""Relationship and usage summary helpers for Prompt Garden workspaces."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .garden import PromptGarden


def _is_archived_payload(
    payload: dict[str, Any],
    *,
    metadata_field: str = "metadata",
    status_field: str | None = None,
) -> bool:
    metadata = payload.get(metadata_field) or {}
    if metadata.get("archived"):
        return True
    if status_field is not None and payload.get(status_field) == "archived":
        return True
    return "archived" in (payload.get("tags") or [])


def _inline_preview(
    text: str,
    limit: int = 180,
) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _normalized_prompt_keywords(
    metadata: dict[str, Any] | None,
) -> list[str]:
    raw_keywords = (metadata or {}).get("keywords") or []
    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()

    for item in raw_keywords:
        keyword = " ".join(str(item or "").split()).strip()
        if not keyword:
            continue
        normalized_key = keyword.casefold()
        if normalized_key in seen_keywords:
            continue
        seen_keywords.add(normalized_key)
        normalized_keywords.append(keyword)

    return normalized_keywords


def _prompt_display_title(
    node: dict[str, Any] | None,
) -> str:
    payload = node or {}
    metadata = payload.get("metadata") or {}

    explicit_title = " ".join(
        str(metadata.get("display_title") or "").split()
    ).strip()
    if explicit_title:
        return explicit_title

    if metadata.get("kind") == "context_variant":
        role = str(
            metadata.get("prompt_role")
            or payload.get("type")
            or "prompt"
        ).strip() or "prompt"
        return f"Contextual {role} prompt"

    title = " ".join(str(payload.get("title") or "").split()).strip()
    if title:
        return title

    prompt_type = str(payload.get("type") or "prompt").strip() or "prompt"
    return f"{prompt_type.title()} prompt"


def _prompt_description(
    node: dict[str, Any] | None,
) -> str:
    payload = node or {}
    metadata = payload.get("metadata") or {}
    explicit_description = " ".join(
        str(metadata.get("description") or "").split()
    ).strip()
    if explicit_description:
        return explicit_description

    if metadata.get("kind") == "context_variant":
        role = str(
            metadata.get("prompt_role")
            or payload.get("type")
            or "prompt"
        ).strip() or "prompt"
        parent_id = str(metadata.get("parent_id") or "").strip()
        if parent_id:
            return (
                f"Auto-generated contextual {role} prompt derived from {parent_id}."
            )
        return f"Auto-generated contextual {role} prompt."

    return ""


def _safe_get_node(
    garden: PromptGarden,
    prompt_id: str,
) -> dict[str, Any] | None:
    try:
        return garden.get_node(prompt_id)
    except KeyError:
        return None


def _safe_get_combo(
    garden: PromptGarden,
    combo_id: str,
) -> dict[str, Any] | None:
    try:
        return garden.get_combo(combo_id)
    except KeyError:
        return None


def _prompt_text_status(
    garden: PromptGarden,
    prompt_id: str,
) -> dict[str, Any]:
    return garden.prompt_text_status(prompt_id)


def _all_experiment_ids(
    garden: PromptGarden,
) -> list[str]:
    indexed_ids = {
        row["id"]
        for row in garden.list_experiments()
        if row.get("id")
    }
    file_ids = {
        path.stem
        for path in garden.experiments_dir.glob("*.json")
        if path.is_file()
    }
    return sorted(indexed_ids | file_ids)


def _combo_experiment_ids_from_workspace(
    garden: PromptGarden,
    combo_id: str,
) -> list[str]:
    experiment_ids: set[str] = set()

    for experiment_id in _all_experiment_ids(garden):
        try:
            experiment = garden.get_experiment(experiment_id)
        except KeyError:
            continue
        if combo_id in experiment.get("combo_ids", []):
            experiment_ids.add(experiment_id)

    return sorted(experiment_ids)


def _combo_result_experiment_ids_from_workspace(
    garden: PromptGarden,
    combo_id: str,
) -> list[str]:
    experiment_ids: set[str] = set()

    for experiment_id in _all_experiment_ids(garden):
        try:
            experiment = garden.get_experiment(experiment_id)
        except KeyError:
            continue
        if combo_id in {
            result["combo_id"]
            for result in experiment.get("results", [])
        }:
            experiment_ids.add(experiment_id)

    return sorted(experiment_ids)


def _prompt_experiment_ids_from_workspace(
    garden: PromptGarden,
    prompt_id: str,
) -> list[str]:
    combo_ids = set(garden.prompt_combo_ids(prompt_id))
    experiment_ids: set[str] = set()

    if not combo_ids:
        return []

    for experiment_id in _all_experiment_ids(garden):
        try:
            experiment = garden.get_experiment(experiment_id)
        except KeyError:
            continue
        if combo_ids.intersection(experiment.get("combo_ids", [])):
            experiment_ids.add(experiment_id)

    return sorted(experiment_ids)


def _prompt_member_rows(
    garden: PromptGarden,
    prompt_ids: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for role, prompt_id in sorted(prompt_ids.items()):
        node = _safe_get_node(garden, prompt_id)
        stats = (node or {}).get("stats") or {}
        metadata = (node or {}).get("metadata") or {}
        rows.append({
            "role": role,
            "prompt_id": prompt_id,
            "prompt_exists": node is not None,
            "prompt_title": _prompt_display_title(node),
            "prompt_description": _prompt_description(node),
            "prompt_type": (node or {}).get("type"),
            "tree_id": (node or {}).get("tree_id"),
            "branch": (node or {}).get("branch"),
            "path": (node or {}).get("path"),
            "word_count": stats.get("word_count"),
            "char_count": stats.get("char_count"),
            "sentence_count": stats.get("sentence_count"),
            "tags": list((node or {}).get("tags", [])),
            "keywords": _normalized_prompt_keywords(metadata),
            "is_archived": (
                _is_archived_payload(node)
                if node is not None else False
            ),
        })

    return rows


def prompt_usage_rows(
    garden: PromptGarden,
    *,
    prompt_types: Sequence[str] = (),
    tree_ids: Sequence[str] = (),
    include_archived: bool = True,
) -> list[dict[str, Any]]:
    """Return prompt-centric usage summaries for one workspace."""

    selected_prompt_types = set(prompt_types)
    selected_tree_ids = set(tree_ids)
    rows: list[dict[str, Any]] = []

    for node in garden.list_nodes():
        if selected_prompt_types and node.get("type") not in selected_prompt_types:
            continue
        if selected_tree_ids and node.get("tree_id") not in selected_tree_ids:
            continue

        is_archived = _is_archived_payload(node)
        if not include_archived and is_archived:
            continue

        prompt_id = node["id"]
        child_rows = garden.get_children(prompt_id)
        combo_ids = garden.prompt_combo_ids(prompt_id)
        experiment_ids = _prompt_experiment_ids_from_workspace(
            garden,
            prompt_id,
        )
        lineage = garden.get_lineage(prompt_id)
        prompt_status = _prompt_text_status(garden, prompt_id)
        prompt_text = str(prompt_status.get("text") or "")
        stats = node.get("stats") or {}
        metadata = node.get("metadata") or {}
        keywords = _normalized_prompt_keywords(metadata)

        rows.append({
            "id": prompt_id,
            "title": node.get("title"),
            "display_title": _prompt_display_title(node),
            "description": _prompt_description(node),
            "keywords": keywords,
            "type": node.get("type"),
            "tree_id": node.get("tree_id"),
            "branch": node.get("branch"),
            "parent_id": node.get("parent_id"),
            "path": node.get("path"),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at"),
            "tags": list(node.get("tags", [])),
            "metadata_kind": metadata.get("kind"),
            "is_archived": is_archived,
            "lineage_ids": [row["id"] for row in lineage],
            "depth": max(len(lineage) - 1, 0),
            "child_prompt_ids": [row["id"] for row in child_rows],
            "child_count": len(child_rows),
            "combo_ids": combo_ids,
            "combo_count": len(combo_ids),
            "experiment_ids": experiment_ids,
            "experiment_count": len(experiment_ids),
            "word_count": stats.get("word_count"),
            "char_count": stats.get("char_count"),
            "sentence_count": stats.get("sentence_count"),
            "placeholder_count": stats.get("placeholder_count"),
            "text_preview": _inline_preview(prompt_text) if prompt_text else "",
            "file_exists": prompt_status.get("file_exists"),
            "file_error": prompt_status.get("error"),
            "resolved_path": prompt_status.get("resolved_path"),
            "used_path_fallback": prompt_status.get("used_path_fallback"),
        })

    return sorted(
        rows,
        key=lambda row: (
            row.get("type") or "",
            row.get("tree_id") or "",
            row.get("created_at") or "",
            row.get("id") or "",
        ),
    )


def combo_usage_rows(
    garden: PromptGarden,
    *,
    include_archived: bool = True,
) -> list[dict[str, Any]]:
    """Return combo-centric usage summaries for one workspace."""

    rows: list[dict[str, Any]] = []

    for combo in garden.list_combos():
        is_archived = _is_archived_payload(
            combo,
            status_field="status",
        )
        if not include_archived and is_archived:
            continue

        combo_id = combo["id"]
        stats = combo.get("stats") or {}
        metadata = combo.get("metadata") or {}
        prompt_members = _prompt_member_rows(
            garden,
            combo.get("prompt_ids", {}),
        )
        dependency_report = garden.inspect_combo_dependencies(combo_id)
        experiment_ids = _combo_experiment_ids_from_workspace(
            garden,
            combo_id,
        )
        result_experiment_ids = _combo_result_experiment_ids_from_workspace(
            garden,
            combo_id,
        )

        rows.append({
            "id": combo_id,
            "title": combo.get("title"),
            "status": combo.get("status"),
            "test_status": combo.get("test_status"),
            "score": combo.get("score"),
            "notes": combo.get("notes"),
            "tags": list(combo.get("tags", [])),
            "created_at": combo.get("created_at"),
            "updated_at": combo.get("updated_at"),
            "kind": metadata.get("kind"),
            "base_combo_id": metadata.get("base_combo_id"),
            "is_archived": is_archived,
            "prompt_roles": [row["role"] for row in prompt_members],
            "prompt_ids": dict(combo.get("prompt_ids", {})),
            "prompt_member_rows": prompt_members,
            "prompt_titles_by_role": {
                row["role"]: row["prompt_title"]
                for row in prompt_members
            },
            "prompt_types_by_role": {
                row["role"]: row["prompt_type"]
                for row in prompt_members
            },
            "missing_prompt_ids": [
                row["prompt_id"]
                for row in prompt_members
                if not row["prompt_exists"]
            ],
            "experiment_ids": experiment_ids,
            "experiment_count": len(experiment_ids),
            "result_experiment_ids": result_experiment_ids,
            "result_experiment_count": len(
                result_experiment_ids
            ),
            "run_count": len(dependency_report["run_ids"]),
            "derived_combo_ids": dependency_report["derived_combo_ids"],
            "derived_combo_count": len(
                dependency_report["derived_combo_ids"]
            ),
            "raw_artifact_count": len(
                dependency_report["raw_artifact_paths"]
            ),
            "normalized_artifact_count": len(
                dependency_report["normalized_artifact_paths"]
            ),
            "total_word_count": stats.get("total_word_count"),
            "total_char_count": stats.get("total_char_count"),
            "total_sentence_count": stats.get("total_sentence_count"),
        })

    return sorted(
        rows,
        key=lambda row: (
            row.get("created_at") or "",
            row.get("id") or "",
        ),
        reverse=True,
    )


def experiment_summary_rows(
    garden: PromptGarden,
    *,
    include_archived: bool = True,
) -> list[dict[str, Any]]:
    """Return experiment-centric summary rows for one workspace."""

    rows: list[dict[str, Any]] = []

    for experiment_id in _all_experiment_ids(garden):
        experiment = garden.get_experiment(experiment_id)
        is_archived = _is_archived_payload(
            experiment,
            status_field="status",
        )
        if not include_archived and is_archived:
            continue

        combo_ids = list(experiment.get("combo_ids", []))
        result_combo_ids = [
            result["combo_id"]
            for result in experiment.get("results", [])
        ]
        summary = experiment.get("summary") or {}
        dependency_report = garden.inspect_experiment_dependencies(
            experiment["id"]
        )
        missing_combo_ids = [
            combo_id
            for combo_id in combo_ids
            if _safe_get_combo(garden, combo_id) is None
        ]

        rows.append({
            "id": experiment["id"],
            "name": experiment.get("name"),
            "status": experiment.get("status"),
            "goal": experiment.get("goal"),
            "hypothesis": experiment.get("hypothesis"),
            "notes": experiment.get("notes"),
            "tags": list(experiment.get("tags", [])),
            "created_at": experiment.get("created_at"),
            "updated_at": experiment.get("updated_at"),
            "is_archived": is_archived,
            "combo_ids": combo_ids,
            "combo_count": len(combo_ids),
            "result_combo_ids": result_combo_ids,
            "tested_combo_count": len(result_combo_ids),
            "untested_combo_count": max(
                len(combo_ids) - len(result_combo_ids),
                0,
            ),
            "missing_combo_ids": missing_combo_ids,
            "missing_combo_count": len(missing_combo_ids),
            "average_score": summary.get("average_score"),
            "best_score": summary.get("best_score"),
            "worst_score": summary.get("worst_score"),
            "run_count": len(dependency_report["run_ids"]),
            "raw_artifact_count": len(
                dependency_report["raw_artifact_paths"]
            ),
            "normalized_artifact_count": len(
                dependency_report["normalized_artifact_paths"]
            ),
            "report_file_count": len(
                dependency_report["report_file_paths"]
            ),
        })

    return sorted(
        rows,
        key=lambda row: (
            row.get("created_at") or "",
            row.get("id") or "",
        ),
        reverse=True,
    )


def experiment_composition_rows(
    garden: PromptGarden,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Return per-combo composition rows for one experiment."""

    experiment = garden.get_experiment(experiment_id)
    result_by_combo_id = {
        result["combo_id"]: result
        for result in experiment.get("results", [])
    }
    rows: list[dict[str, Any]] = []

    for position, combo_id in enumerate(experiment.get("combo_ids", []), start=1):
        combo = _safe_get_combo(garden, combo_id)
        result = result_by_combo_id.get(combo_id) or {}

        if combo is None:
            rows.append({
                "experiment_id": experiment_id,
                "position": position,
                "combo_id": combo_id,
                "combo_exists": False,
                "combo_title": None,
                "combo_status": None,
                "combo_test_status": None,
                "combo_score": None,
                "combo_kind": None,
                "combo_tags": [],
                "prompt_ids": {},
                "prompt_count": 0,
                "prompt_member_rows": [],
                "missing_prompt_ids": [],
                "has_result": bool(result),
                "result_status": result.get("status"),
                "result_score": result.get("score"),
                "result_text": result.get("result_text"),
            })
            continue

        prompt_members = _prompt_member_rows(
            garden,
            combo.get("prompt_ids", {}),
        )
        metadata = combo.get("metadata") or {}

        rows.append({
            "experiment_id": experiment_id,
            "position": position,
            "combo_id": combo_id,
            "combo_exists": True,
            "combo_title": combo.get("title"),
            "combo_status": combo.get("status"),
            "combo_test_status": combo.get("test_status"),
            "combo_score": combo.get("score"),
            "combo_kind": metadata.get("kind"),
            "combo_tags": list(combo.get("tags", [])),
            "prompt_ids": dict(combo.get("prompt_ids", {})),
            "prompt_count": len(prompt_members),
            "prompt_member_rows": prompt_members,
            "missing_prompt_ids": [
                row["prompt_id"]
                for row in prompt_members
                if not row["prompt_exists"]
            ],
            "system_prompt_id": (combo.get("prompt_ids") or {}).get("system"),
            "user_prompt_id": (combo.get("prompt_ids") or {}).get("user"),
            "fewshot_prompt_id": (combo.get("prompt_ids") or {}).get("fewshot"),
            "has_result": bool(result),
            "result_status": result.get("status"),
            "result_score": result.get("score"),
            "result_text": result.get("result_text"),
        })

    return rows


def experiment_composition(
    garden: PromptGarden,
    experiment_id: str,
) -> dict[str, Any]:
    """Return one experiment together with composition-focused summary data."""

    experiment = garden.get_experiment(experiment_id)
    combo_rows = experiment_composition_rows(garden, experiment_id)
    prompt_ids = sorted(
        {
            prompt_id
            for row in combo_rows
            for prompt_id in row.get("prompt_ids", {}).values()
        }
    )
    missing_combo_ids = [
        row["combo_id"]
        for row in combo_rows
        if not row.get("combo_exists")
    ]
    summary_row = next(
        (
            row for row in experiment_summary_rows(garden)
            if row["id"] == experiment_id
        ),
        None,
    )

    return {
        "experiment": experiment,
        "summary": summary_row,
        "combo_rows": combo_rows,
        "combo_count": len(combo_rows),
        "missing_combo_ids": missing_combo_ids,
        "missing_combo_count": len(missing_combo_ids),
        "prompt_ids": prompt_ids,
        "prompt_id_count": len(prompt_ids),
    }


def prompt_detail_bundle(
    garden: PromptGarden,
    prompt_id: str,
) -> dict[str, Any]:
    """Return one prompt together with text, lineage, and dependency details."""

    node = garden.get_node(prompt_id)
    prompt_status = _prompt_text_status(garden, prompt_id)
    prompt_text = str(prompt_status.get("text") or "")
    summary_row = next(
        (
            row for row in prompt_usage_rows(garden)
            if row["id"] == prompt_id
        ),
        None,
    )
    lineage_rows = []
    for lineage_node in garden.get_lineage(prompt_id):
        lineage_rows.append({
            "id": lineage_node.get("id"),
            "title": lineage_node.get("title"),
            "display_title": _prompt_display_title(lineage_node),
            "description": _prompt_description(lineage_node),
            "type": lineage_node.get("type"),
            "tree_id": lineage_node.get("tree_id"),
            "branch": lineage_node.get("branch"),
            "parent_id": lineage_node.get("parent_id"),
            "created_at": lineage_node.get("created_at"),
            "is_archived": _is_archived_payload(lineage_node),
        })

    child_rows = []
    for child_node in garden.get_children(prompt_id):
        child_rows.append({
            "id": child_node.get("id"),
            "title": child_node.get("title"),
            "display_title": _prompt_display_title(child_node),
            "description": _prompt_description(child_node),
            "type": child_node.get("type"),
            "branch": child_node.get("branch"),
            "created_at": child_node.get("created_at"),
            "is_archived": _is_archived_payload(child_node),
        })

    combo_ids = garden.prompt_combo_ids(prompt_id)
    combo_summary_rows = {
        row["id"]: row
        for row in combo_usage_rows(garden)
    }
    dependent_combo_rows = [
        combo_summary_rows[combo_id]
        for combo_id in combo_ids
        if combo_id in combo_summary_rows
    ]

    experiment_ids = _prompt_experiment_ids_from_workspace(
        garden,
        prompt_id,
    )
    experiment_summary_by_id = {
        row["id"]: row
        for row in experiment_summary_rows(garden)
    }
    dependent_experiment_rows = [
        experiment_summary_by_id[experiment_id]
        for experiment_id in experiment_ids
        if experiment_id in experiment_summary_by_id
    ]

    parsed_fewshot_examples: list[dict[str, Any]] | None = None
    parsed_fewshot_error: str | None = None
    if (
        node.get("type") == "fewshot"
        and not prompt_status.get("error")
        and prompt_text
    ):
        try:
            parsed = json.loads(prompt_text)
        except json.JSONDecodeError as error:
            parsed_fewshot_error = str(error)
        else:
            if isinstance(parsed, list):
                parsed_fewshot_examples = [
                    item if isinstance(item, dict) else {"value": item}
                    for item in parsed
                ]
            else:
                parsed_fewshot_examples = [{"value": parsed}]

    return {
        "prompt": {
            "id": node.get("id"),
            "title": node.get("title"),
            "display_title": _prompt_display_title(node),
            "description": _prompt_description(node),
            "keywords": _normalized_prompt_keywords(node.get("metadata") or {}),
            "type": node.get("type"),
            "tree_id": node.get("tree_id"),
            "branch": node.get("branch"),
            "parent_id": node.get("parent_id"),
            "path": node.get("path"),
            "resolved_path": prompt_status.get("resolved_path"),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at"),
            "tags": list(node.get("tags", [])),
            "metadata": dict(node.get("metadata") or {}),
            "stats": dict(node.get("stats") or {}),
            "is_archived": _is_archived_payload(node),
            "text": prompt_text,
            "text_preview": _inline_preview(prompt_text) if prompt_text else "",
            "file_exists": prompt_status.get("file_exists"),
            "file_error": prompt_status.get("error"),
            "used_path_fallback": prompt_status.get("used_path_fallback"),
        },
        "summary": summary_row,
        "lineage_rows": lineage_rows,
        "child_rows": child_rows,
        "dependent_combo_rows": dependent_combo_rows,
        "dependent_experiment_rows": dependent_experiment_rows,
        "parsed_fewshot_examples": parsed_fewshot_examples,
        "parsed_fewshot_error": parsed_fewshot_error,
    }


def prompt_similarity_items(
    garden: PromptGarden,
    *,
    include_archived: bool = True,
) -> list[dict[str, Any]]:
    """Return prompt items with full text and metadata for similarity workflows."""

    items: list[dict[str, Any]] = []

    for row in prompt_usage_rows(
        garden,
        include_archived=include_archived,
    ):
        prompt_id = row["id"]
        prompt_status = _prompt_text_status(garden, prompt_id)
        prompt_text = str(prompt_status.get("text") or "")
        if not prompt_text:
            continue
        items.append({
            "item_id": prompt_id,
            "label": (
                f"{prompt_id} | {row.get('title', '')}"
                f" | type={row.get('type', '-')}"
                f" | branch={row.get('branch', '-')}"
            ),
            "group_id": row.get("tree_id"),
            "text": prompt_text,
            "id": prompt_id,
            "title": row.get("title"),
            "type": row.get("type"),
            "tree_id": row.get("tree_id"),
            "branch": row.get("branch"),
            "parent_id": row.get("parent_id"),
            "path": row.get("path"),
            "tags": list(row.get("tags", [])),
            "is_archived": row.get("is_archived"),
            "lineage_ids": list(row.get("lineage_ids", [])),
            "child_prompt_ids": list(row.get("child_prompt_ids", [])),
            "combo_ids": list(row.get("combo_ids", [])),
            "experiment_ids": list(row.get("experiment_ids", [])),
            "combo_count": row.get("combo_count"),
            "experiment_count": row.get("experiment_count"),
            "child_count": row.get("child_count"),
            "depth": row.get("depth"),
            "word_count": row.get("word_count"),
            "char_count": row.get("char_count"),
            "sentence_count": row.get("sentence_count"),
            "placeholder_count": row.get("placeholder_count"),
            "text_preview": _inline_preview(prompt_text),
        })

    return items


def combo_detail_bundle(
    garden: PromptGarden,
    combo_id: str,
) -> dict[str, Any]:
    """Return one combo together with prompt-role and experiment details."""

    combo = garden.get_combo(combo_id)
    summary_row = next(
        (
            row for row in combo_usage_rows(garden)
            if row["id"] == combo_id
        ),
        None,
    )
    dependency_report = garden.inspect_combo_dependencies(combo_id)
    experiment_summary_by_id = {
        row["id"]: row
        for row in experiment_summary_rows(garden)
    }
    dependent_experiment_rows = [
        experiment_summary_by_id[experiment_id]
        for experiment_id in dependency_report.get("experiment_ids", [])
        if experiment_id in experiment_summary_by_id
    ]
    prompt_summary_by_id = {
        row["id"]: row
        for row in prompt_usage_rows(garden)
    }
    prompt_role_rows = []
    for member in summary_row.get("prompt_member_rows", []) if summary_row else []:
        prompt_id = member.get("prompt_id")
        prompt_role_rows.append({
            **member,
            "prompt_summary": prompt_summary_by_id.get(prompt_id),
        })

    combo_summary_by_id = {
        row["id"]: row
        for row in combo_usage_rows(garden)
    }
    derived_combo_rows = [
        combo_summary_by_id[derived_combo_id]
        for derived_combo_id in dependency_report.get("derived_combo_ids", [])
        if derived_combo_id in combo_summary_by_id
    ]

    return {
        "combo": {
            "id": combo.get("id"),
            "title": combo.get("title"),
            "status": combo.get("status"),
            "test_status": combo.get("test_status"),
            "score": combo.get("score"),
            "notes": combo.get("notes"),
            "tags": list(combo.get("tags", [])),
            "created_at": combo.get("created_at"),
            "updated_at": combo.get("updated_at"),
            "metadata": dict(combo.get("metadata") or {}),
            "stats": dict(combo.get("stats") or {}),
            "prompt_ids": dict(combo.get("prompt_ids", {})),
            "is_archived": _is_archived_payload(
                combo,
                status_field="status",
            ),
        },
        "summary": summary_row,
        "prompt_role_rows": prompt_role_rows,
        "dependent_experiment_rows": dependent_experiment_rows,
        "derived_combo_rows": derived_combo_rows,
        "dependency_report": dependency_report,
    }


__all__ = [
    "combo_detail_bundle",
    "combo_usage_rows",
    "experiment_composition",
    "experiment_composition_rows",
    "experiment_summary_rows",
    "prompt_detail_bundle",
    "prompt_similarity_items",
    "prompt_usage_rows",
]
