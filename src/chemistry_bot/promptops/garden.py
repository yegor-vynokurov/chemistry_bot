"""Prompt Garden workspace management for prompts, combos, and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import HtmlDiff, unified_diff
from itertools import product
from pathlib import Path
from typing import Any, Sequence
import hashlib
import html as html_lib
import json
import re


@dataclass
class PromptNode:
    id: str
    type: str
    tree_id: str
    path: str
    parent_id: str | None
    branch: str
    title: str
    created_at: str
    tags: list[str]
    metadata: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None


@dataclass
class PromptExperiment:
    id: str
    name: str
    goal: str
    hypothesis: str
    notes: str
    tags: list[str]
    status: str
    created_at: str


class PromptGarden:
    """
    Local prompt graph, combo registry and experiment tracker.

    Storage model:
    - prompt texts: prompts/<type>/<id>.md
    - prompt nodes: registry/nodes.jsonl
    - graph edges: registry/edges.jsonl
    - prompt combinations: registry/combos.jsonl
    - experiment nodes: registry/experiment_nodes.jsonl
    - experiment objects: experiments/<experiment_id>.json
    - legacy / compact result summaries: registry/experiments.jsonl
    - raw run logs: registry/runs.jsonl

    Experiments are graph nodes. They connect to combos through edges.
    Results are persisted as JSON inside experiments/<experiment_id>.json.
    """

    DEFAULT_PROMPT_TYPES = {
        "system": "sys",
        "user": "usr",
        "safety": "saf",
        "assistant": "ast",
        "tool": "tol",
        "fewshot": "fsh",
    }

    CONTEXT_RENDER_VERSION = "context-render-v1"
    COMBO_GENERATOR_VERSION = "combo-generator-v1"
    PROMPT_STATS_VERSION = "prompt-stats-v1"

    def __init__(
        self,
        root: str | Path,
        prompt_types: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.prompt_types = {
            **self.DEFAULT_PROMPT_TYPES,
            **(prompt_types or {}),
        }

        self.prompts_dir = self.root / "prompts"
        self.cases_dir = self.root / "cases"
        self.registry_dir = self.root / "registry"
        self.experiments_dir = self.root / "experiments"
        self.runs_dir = self.root / "runs"
        self.raw_runs_dir = self.runs_dir / "raw"
        self.normalized_runs_dir = self.runs_dir / "normalized"
        self.reports_dir = self.root / "reports"
        self.cache_dir = self.root / "cache"
        self.embedding_cache_dir = self.cache_dir / "embeddings"

        self.nodes_path = self.registry_dir / "nodes.jsonl"
        self.edges_path = self.registry_dir / "edges.jsonl"
        self.combos_path = self.registry_dir / "combos.jsonl"
        self.experiment_nodes_path = self.registry_dir / "experiment_nodes.jsonl"
        self.experiments_path = self.registry_dir / "experiments.jsonl"
        self.runs_path = self.registry_dir / "runs.jsonl"

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.raw_runs_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_runs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        for prompt_type in self.prompt_types:
            (self.prompts_dir / prompt_type).mkdir(
                parents=True,
                exist_ok=True,
            )

        for path in [
            self.nodes_path,
            self.edges_path,
            self.combos_path,
            self.experiment_nodes_path,
            self.experiments_path,
            self.runs_path,
        ]:
            path.touch(exist_ok=True)

    def ensure_run_scope_dirs(
        self,
        scope: str,
    ) -> tuple[Path, Path]:
        """
        Ensure raw and normalized run directories exist for a scope.

        The scope is normally an experiment id such as ``exp_000007`` or the
        special ad-hoc marker ``_adhoc``.
        """

        raw_scope_dir = self.raw_runs_dir / scope
        normalized_scope_dir = self.normalized_runs_dir / scope
        raw_scope_dir.mkdir(parents=True, exist_ok=True)
        normalized_scope_dir.mkdir(parents=True, exist_ok=True)
        return raw_scope_dir, normalized_scope_dir

    def ensure_report_scope_dir(
        self,
        scope: str,
    ) -> Path:
        """Ensure the derived report directory exists for a scope."""

        report_scope_dir = self.reports_dir / scope
        report_scope_dir.mkdir(parents=True, exist_ok=True)
        return report_scope_dir

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        rows: list[dict[str, Any]] = []

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

        return rows

    @staticmethod
    def _write_jsonl(
        path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_jsonl(
        path: Path,
        row: dict[str, Any],
    ) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_nodes(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._write_jsonl(self.nodes_path, rows)

    def _save_combos(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._write_jsonl(self.combos_path, rows)

    def _save_compact_experiment_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._write_jsonl(self.experiments_path, rows)

    def _save_runs(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._write_jsonl(self.runs_path, rows)

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/")

    @staticmethod
    def _ensure_tag(
        tags: list[str] | None,
        tag: str,
    ) -> list[str]:
        final_tags = list(tags or [])
        if tag not in final_tags:
            final_tags.append(tag)
        return final_tags

    @staticmethod
    def _stable_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def _hash_data(cls, value: Any) -> str:
        return hashlib.sha256(
            cls._stable_json(value).encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.strip() + "\n"

    @staticmethod
    def _delete_file_if_present(path: Path) -> None:
        if path.exists():
            path.unlink()

    @staticmethod
    def _remove_directory_if_empty(path: Path) -> None:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    @staticmethod
    def _normalize_prompt_description(
        value: str | None,
    ) -> str:
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def _normalize_keywords(
        keywords: Sequence[str] | None,
    ) -> list[str]:
        normalized_keywords: list[str] = []
        seen_keywords: set[str] = set()

        for item in keywords or []:
            keyword = " ".join(str(item or "").split()).strip()
            if not keyword:
                continue
            normalized_key = keyword.casefold()
            if normalized_key in seen_keywords:
                continue
            seen_keywords.add(normalized_key)
            normalized_keywords.append(keyword)

        return normalized_keywords

    @classmethod
    def _apply_prompt_descriptor_metadata(
        cls,
        metadata: dict[str, Any] | None,
        *,
        description: str | None = None,
        keywords: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        final_metadata = dict(metadata or {})

        if description is not None:
            normalized_description = cls._normalize_prompt_description(
                description
            )
            if normalized_description:
                final_metadata["description"] = normalized_description
            else:
                final_metadata.pop("description", None)

        if keywords is not None:
            normalized_keywords = cls._normalize_keywords(keywords)
            if normalized_keywords:
                final_metadata["keywords"] = normalized_keywords
            else:
                final_metadata.pop("keywords", None)

        return final_metadata

    @classmethod
    def _archive_metadata(
        cls,
        metadata: dict[str, Any] | None,
        reason: str = "",
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        final_metadata = dict(metadata or {})
        final_metadata["archived"] = True
        final_metadata["archived_at"] = cls._now()
        if reason.strip():
            final_metadata["archive_reason"] = reason.strip()
        if extra_fields:
            final_metadata.update(extra_fields)
        return final_metadata

    @staticmethod
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
        tags = payload.get("tags") or []
        return "archived" in tags

    def _remove_edges(
        self,
        *,
        from_id: str | None = None,
        to_id: str | None = None,
        kinds: set[str] | None = None,
    ) -> int:
        edges = self.list_edges()
        kept_edges: list[dict[str, Any]] = []
        removed_count = 0

        for edge in edges:
            if from_id is not None and edge.get("from") != from_id:
                kept_edges.append(edge)
                continue
            if to_id is not None and edge.get("to") != to_id:
                kept_edges.append(edge)
                continue
            if kinds is not None and edge.get("kind") not in kinds:
                kept_edges.append(edge)
                continue
            removed_count += 1

        if removed_count:
            self._write_jsonl(self.edges_path, kept_edges)
        return removed_count

    def _scope_artifact_paths(
        self,
        scope: str,
    ) -> dict[str, list[str]]:
        raw_scope_dir = self.raw_runs_dir / scope
        normalized_scope_dir = self.normalized_runs_dir / scope
        report_scope_dir = self.reports_dir / scope

        def _relative_json_files(path: Path) -> list[str]:
            if not path.exists():
                return []
            return sorted(
                self._relative(child)
                for child in path.glob("*.json")
                if child.is_file()
            )

        return {
            "raw_artifact_paths": _relative_json_files(raw_scope_dir),
            "normalized_artifact_paths": _relative_json_files(normalized_scope_dir),
            "report_file_paths": _relative_json_files(report_scope_dir),
        }

    def _combo_artifact_paths(
        self,
        combo_id: str,
    ) -> dict[str, list[str]]:
        raw_paths = sorted(
            self._relative(path)
            for path in self.raw_runs_dir.glob(f"**/*{combo_id}*.json")
            if path.is_file()
        )
        normalized_paths = sorted(
            self._relative(path)
            for path in self.normalized_runs_dir.glob(f"**/*{combo_id}*.json")
            if path.is_file()
        )
        return {
            "raw_artifact_paths": raw_paths,
            "normalized_artifact_paths": normalized_paths,
        }

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------

    def _next_prompt_id(self, prompt_type: str) -> str:
        prefix = self.prompt_types.get(prompt_type, "prm")
        numbers: list[int] = []

        for node in self.list_nodes():
            node_id = node["id"]
            if node_id.startswith(prefix + "_"):
                try:
                    numbers.append(int(node_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        return f"{prefix}_{max(numbers, default=0) + 1:06d}"

    def _next_combo_id(self) -> str:
        numbers: list[int] = []

        for combo in self.list_combos():
            combo_id = combo["id"]
            if combo_id.startswith("combo_"):
                try:
                    numbers.append(int(combo_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        return f"combo_{max(numbers, default=0) + 1:06d}"

    def _next_experiment_id(self) -> str:
        numbers: list[int] = []

        for exp in self.list_experiments():
            exp_id = exp["id"]
            if exp_id.startswith("exp_"):
                try:
                    numbers.append(int(exp_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        return f"exp_{max(numbers, default=0) + 1:06d}"

    def _next_run_id(self) -> str:
        numbers: list[int] = []

        for run in self.list_runs():
            run_id = run["id"]
            if run_id.startswith("run_"):
                try:
                    numbers.append(int(run_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue

        return f"run_{max(numbers, default=0) + 1:06d}"

    # ------------------------------------------------------------------
    # Prompt statistics
    # ------------------------------------------------------------------

    @classmethod
    def compute_prompt_stats(cls, text: str) -> dict[str, Any]:
        """
        Compute lightweight prompt characteristics.

        These stats do not enter the LLM prompt. They are for analysis:
        length, density, sentence shape, placeholders, etc.
        Future metrics can be added without breaking the main flow.
        """

        stripped = text.strip()
        lines = stripped.splitlines() if stripped else []
        non_empty_lines = [line for line in lines if line.strip()]

        words = re.findall(
            r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9]+(?:[-'][A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9]+)?",
            stripped,
            flags=re.UNICODE,
        )

        sentence_chunks = re.split(r"(?<=[.!?])\s+", stripped)
        sentences = [
            sentence.strip()
            for sentence in sentence_chunks
            if sentence.strip()
        ]

        placeholders = sorted(set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", stripped)))

        word_lengths = [len(word) for word in words]
        sentence_word_counts = [
            len(re.findall(
                r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9]+(?:[-'][A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9]+)?",
                sentence,
                flags=re.UNICODE,
            ))
            for sentence in sentences
        ]

        return {
            "version": cls.PROMPT_STATS_VERSION,
            "char_count": len(stripped),
            "line_count": len(lines),
            "non_empty_line_count": len(non_empty_lines),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": round(sum(word_lengths) / len(word_lengths), 4) if word_lengths else 0.0,
            "avg_sentence_words": round(sum(sentence_word_counts) / len(sentence_word_counts), 4) if sentence_word_counts else 0.0,
            "placeholder_count": len(placeholders),
            "placeholders": placeholders,
            "brace_count": stripped.count("{") + stripped.count("}"),
        }

    def refresh_node_stats(self, prompt_id: str) -> dict[str, Any]:
        nodes = self.list_nodes()
        updated_node: dict[str, Any] | None = None

        for node in nodes:
            if node["id"] == prompt_id:
                text = self.read_prompt(prompt_id)
                text_to_hash = self._normalize_text(text)
                node["stats"] = self.compute_prompt_stats(text)
                metadata = node.get("metadata") or {}
                metadata["content_hash"] = self._hash_text(text_to_hash)
                node["metadata"] = metadata
                updated_node = node
                break

        if updated_node is None:
            raise KeyError(f"Prompt node not found: {prompt_id}")

        self._write_jsonl(self.nodes_path, nodes)
        return updated_node

    def refresh_all_prompt_stats(self) -> list[dict[str, Any]]:
        refreshed = []
        for node in self.list_nodes():
            refreshed.append(self.refresh_node_stats(node["id"]))
        return refreshed

    # ------------------------------------------------------------------
    # Prompt nodes
    # ------------------------------------------------------------------

    def add_node(
        self,
        title: str,
        text: str,
        prompt_type: str | None = None,
        tree_id: str | None = None,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        branch: str = "main",
        edge_kind: str = "parent_of",
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
        keywords: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Add a prompt node.

        Root node:
            add_node(prompt_type="system", tree_id="system_main", ...)

        Child node:
            add_node(parent_id="sys_000001", ...)

        For child nodes, prompt_type and tree_id are inherited.
        """

        if parent_id is not None:
            parent = self.get_node(parent_id)
            inherited_prompt_type = parent["type"]
            inherited_tree_id = parent["tree_id"]

            if prompt_type is None:
                prompt_type = inherited_prompt_type
            elif prompt_type != inherited_prompt_type:
                raise ValueError(
                    "Child node prompt_type must match parent type: "
                    f"expected {inherited_prompt_type!r}, got {prompt_type!r}"
                )

            if tree_id is None:
                tree_id = inherited_tree_id
            elif tree_id != inherited_tree_id:
                raise ValueError(
                    "Child node tree_id must match parent tree_id: "
                    f"expected {inherited_tree_id!r}, got {tree_id!r}"
                )

        else:
            if prompt_type is None:
                raise ValueError(
                    "prompt_type is required when creating a root node."
                )

            if tree_id is None:
                raise ValueError(
                    "tree_id is required when creating a root node."
                )

        if prompt_type not in self.prompt_types:
            self.prompt_types[prompt_type] = prompt_type[:3].lower()

        prompt_id = self._next_prompt_id(prompt_type)
        prompt_dir = self.prompts_dir / prompt_type
        prompt_dir.mkdir(parents=True, exist_ok=True)

        path = prompt_dir / f"{prompt_id}.md"
        text_to_write = self._normalize_text(text)
        path.write_text(text_to_write, encoding="utf-8")

        created_at = self._now()
        final_metadata = self._apply_prompt_descriptor_metadata(
            metadata,
            description=description,
            keywords=keywords,
        )
        final_metadata.setdefault("content_hash", self._hash_text(text_to_write))

        node = {
            "id": prompt_id,
            "type": prompt_type,
            "tree_id": tree_id,
            "path": self._relative(path),
            "parent_id": parent_id,
            "branch": branch,
            "title": title,
            "created_at": created_at,
            "tags": tags or [],
            "metadata": final_metadata,
            "stats": self.compute_prompt_stats(text_to_write),
        }

        self._append_jsonl(self.nodes_path, node)

        if parent_id is not None:
            self._append_edge(
                from_id=parent_id,
                to_id=prompt_id,
                kind=edge_kind,
                metadata={},
                created_at=created_at,
            )

        return node

    def create_root(
        self,
        prompt_type: str,
        tree_id: str,
        title: str,
        text: str,
        tags: list[str] | None = None,
        branch: str = "main",
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
        keywords: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self.add_node(
            prompt_type=prompt_type,
            tree_id=tree_id,
            title=title,
            text=text,
            tags=tags,
            branch=branch,
            parent_id=None,
            metadata=metadata,
            description=description,
            keywords=keywords,
        )

    def create_child(
        self,
        parent_id: str,
        title: str,
        text: str,
        tags: list[str] | None = None,
        branch: str = "main",
        edge_kind: str = "parent_of",
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
        keywords: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self.add_node(
            parent_id=parent_id,
            title=title,
            text=text,
            tags=tags,
            branch=branch,
            edge_kind=edge_kind,
            metadata=metadata,
            description=description,
            keywords=keywords,
        )

    def list_nodes(
        self,
        prompt_type: str | None = None,
        tree_id: str | None = None,
    ) -> list[dict[str, Any]]:
        nodes = self._read_jsonl(self.nodes_path)

        if prompt_type is not None:
            nodes = [
                node for node in nodes
                if node["type"] == prompt_type
            ]

        if tree_id is not None:
            nodes = [
                node for node in nodes
                if node["tree_id"] == tree_id
            ]

        return nodes

    def get_node(self, prompt_id: str) -> dict[str, Any]:
        for node in self.list_nodes():
            if node["id"] == prompt_id:
                return node

        raise KeyError(f"Prompt node not found: {prompt_id}")

    def _canonical_prompt_id(
        self,
        node: dict[str, Any],
    ) -> str | None:
        prompt_type = str(node.get("type") or "").strip()
        prefix = self.prompt_types.get(prompt_type)
        node_id = str(node.get("id") or "").strip()
        match = re.fullmatch(r"[a-z]+_(\d+)", node_id)
        if not prefix or match is None:
            return None
        return f"{prefix}_{match.group(1)}"

    def prompt_text_status(
        self,
        prompt_id: str,
    ) -> dict[str, Any]:
        """Resolve prompt text together with file-health metadata."""

        node = self.get_node(prompt_id)
        declared_path = str(node.get("path") or "").replace("\\", "/")
        candidate_paths: list[str] = []

        if declared_path:
            candidate_paths.append(declared_path)

        canonical_prompt_id = self._canonical_prompt_id(node)
        prompt_type = str(node.get("type") or "").strip()
        if canonical_prompt_id and prompt_type:
            canonical_path = f"prompts/{prompt_type}/{canonical_prompt_id}.md"
            if canonical_path not in candidate_paths:
                candidate_paths.append(canonical_path)

        for candidate_index, relative_path in enumerate(candidate_paths):
            absolute_path = self.root / relative_path
            if not absolute_path.exists():
                continue

            try:
                text = absolute_path.read_text(encoding="utf-8")
            except OSError as error:
                return {
                    "prompt_id": prompt_id,
                    "declared_path": declared_path or None,
                    "resolved_path": relative_path,
                    "file_exists": False,
                    "used_path_fallback": candidate_index > 0,
                    "text": None,
                    "error": str(error),
                }

            return {
                "prompt_id": prompt_id,
                "declared_path": declared_path or None,
                "resolved_path": relative_path,
                "file_exists": True,
                "used_path_fallback": candidate_index > 0,
                "text": text,
                "error": None,
            }

        tried_paths = candidate_paths or [declared_path or "<missing-path>"]
        return {
            "prompt_id": prompt_id,
            "declared_path": declared_path or None,
            "resolved_path": None,
            "file_exists": False,
            "used_path_fallback": False,
            "text": None,
            "error": (
                f"Prompt file not found for {prompt_id}. "
                f"Tried: {', '.join(tried_paths)}"
            ),
        }

    def read_prompt(self, prompt_id: str) -> str:
        status = self.prompt_text_status(prompt_id)
        prompt_text = status.get("text")
        if prompt_text is None:
            raise FileNotFoundError(str(status.get("error") or prompt_id))
        return str(prompt_text)

    def write_prompt_text(self, prompt_id: str, text: str) -> None:
        """
        Overwrite prompt text and refresh stats.

        Prefer create_child() for versioned edits.
        """

        node = self.get_node(prompt_id)
        (self.root / node["path"]).write_text(
            self._normalize_text(text),
            encoding="utf-8",
        )
        self.refresh_node_stats(prompt_id)

    def show_prompt(self, prompt_id: str) -> None:
        node = self.get_node(prompt_id)
        print(f"{node['id']} | {node['type']} | {node['title']}")
        print(f"stats: {node.get('stats', {})}")
        print("-" * 88)
        print(self.read_prompt(prompt_id))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _append_edge(
        self,
        from_id: str,
        to_id: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        edge = {
            "from": from_id,
            "to": to_id,
            "kind": kind,
            "created_at": created_at or self._now(),
            "metadata": metadata or {},
        }
        self._append_jsonl(self.edges_path, edge)
        return edge

    def list_edges(
        self,
        kind: str | None = None,
        from_id: str | None = None,
        to_id: str | None = None,
    ) -> list[dict[str, Any]]:
        edges = self._read_jsonl(self.edges_path)

        if kind is not None:
            edges = [edge for edge in edges if edge.get("kind") == kind]

        if from_id is not None:
            edges = [edge for edge in edges if edge.get("from") == from_id]

        if to_id is not None:
            edges = [edge for edge in edges if edge.get("to") == to_id]

        return edges

    def _edge_exists(
        self,
        from_id: str,
        to_id: str,
        kind: str,
    ) -> bool:
        return any(
            edge.get("from") == from_id
            and edge.get("to") == to_id
            and edge.get("kind") == kind
            for edge in self.list_edges()
        )

    # ------------------------------------------------------------------
    # Context rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _plain_block(
        title: str,
        values: dict[str, Any],
    ) -> str:
        lines = [f"### {title}"]

        for key, value in values.items():
            if value is None or value == "":
                continue

            label = key.replace("_", " ").title()
            value_text = str(value).strip()

            if not value_text:
                continue

            lines.append(f"- {label}: {value_text}")

        return "\n".join(lines)

    @staticmethod
    def _replace_known_placeholders(
        text: str,
        values: dict[str, Any],
    ) -> str:
        result = text

        for key, value in values.items():
            if value is None:
                value = ""
            result = result.replace("{" + key + "}", str(value))

        return result

    @staticmethod
    def _escape_template_braces(
        text: str,
        runtime_placeholders: tuple[str, ...] = ("question",),
    ) -> str:
        """
        Escape literal braces so LangChain does not treat JSON examples
        or context text as format variables. Runtime placeholders stay active.
        """

        sentinels: dict[str, str] = {}
        prepared = text

        for name in runtime_placeholders:
            placeholder = "{" + name + "}"
            sentinel = f"__PROMPT_GARDEN_RUNTIME_{name.upper()}__"
            sentinels[name] = sentinel
            prepared = prepared.replace(placeholder, sentinel)

        prepared = prepared.replace("{", "{{").replace("}", "}}")

        for name, sentinel in sentinels.items():
            prepared = prepared.replace(sentinel, "{" + name + "}")

        return prepared

    def render_contextual_prompt(
        self,
        parent_id: str,
        prompt_role: str,
        student_context: dict[str, Any],
        teacher_context: dict[str, Any],
        runtime_placeholders: tuple[str, ...] = ("question",),
    ) -> str:
        base_text = self.read_prompt(parent_id)

        legacy_values = {
            "question": "{question}",
            "language": student_context.get("language", ""),
            "type_of_school": student_context.get("school_context", ""),
            "school_class": student_context.get("learner_profile", ""),
            "explanation_level": student_context.get("learning_preferences", ""),
            "protocol_context": student_context.get("protocol_context", ""),
            "notes": student_context.get("notes", ""),
            "level_of_teacher": teacher_context.get("teacher_profile", ""),
            "personality_of_teacher": teacher_context.get("teaching_style", ""),
            "teacher_notes": teacher_context.get("notes", ""),
        }

        rendered_base = self._replace_known_placeholders(base_text, legacy_values)

        if prompt_role == "system":
            context_addition = "\n\n---\n\n## PromptGarden context additions\n"
            context_addition += "These additions were generated automatically from StudentContext and TeacherContext.\n\n"
            context_addition += self._plain_block("Teacher Context", teacher_context)
            context_addition += "\n\n"
            context_addition += self._plain_block("Student Context", student_context)

        elif prompt_role == "user":
            context_addition = "\n\n---\n\n## PromptGarden request context additions\n"
            context_addition += "Use these additions as stable context for the next student question.\n\n"
            context_addition += self._plain_block("Student Context", student_context)
            context_addition += "\n\n"
            context_addition += self._plain_block("Teacher Response Preferences", teacher_context)

        else:
            context_addition = "\n\n---\n\n## PromptGarden Context\n"
            context_addition += self._plain_block("Student Context", student_context)
            context_addition += "\n\n"
            context_addition += self._plain_block("Teacher Context", teacher_context)

        combined = rendered_base.strip() + context_addition

        return self._escape_template_braces(
            combined,
            runtime_placeholders=runtime_placeholders,
        )

    def get_or_create_context_node(
        self,
        parent_id: str,
        prompt_role: str,
        student_context: dict[str, Any],
        teacher_context: dict[str, Any],
        title: str | None = None,
        runtime_placeholders: tuple[str, ...] = ("question",),
    ) -> dict[str, Any]:
        context_payload = {
            "render_version": self.CONTEXT_RENDER_VERSION,
            "parent_id": parent_id,
            "prompt_role": prompt_role,
            "student_context": student_context,
            "teacher_context": teacher_context,
            "runtime_placeholders": runtime_placeholders,
        }
        context_hash = self._hash_data(context_payload)

        for node in self.list_nodes():
            metadata = node.get("metadata") or {}
            if (
                node.get("parent_id") == parent_id
                and metadata.get("kind") == "context_variant"
                and metadata.get("context_hash") == context_hash
                and metadata.get("prompt_role") == prompt_role
            ):
                return node

        text = self.render_contextual_prompt(
            parent_id=parent_id,
            prompt_role=prompt_role,
            student_context=student_context,
            teacher_context=teacher_context,
            runtime_placeholders=runtime_placeholders,
        )

        role_title = str(prompt_role or "prompt").strip() or "prompt"
        node_title = title or f"Contextual {role_title} prompt"
        description = (
            f"Auto-generated contextual {role_title} prompt derived from {parent_id}."
        )

        return self.add_node(
            parent_id=parent_id,
            title=node_title,
            text=text,
            tags=["context", prompt_role, "auto-generated"],
            branch="context",
            edge_kind="context_variant",
            metadata={
                "kind": "context_variant",
                "parent_id": parent_id,
                "prompt_role": prompt_role,
                "context_hash": context_hash,
                "render_version": self.CONTEXT_RENDER_VERSION,
                "student_context": student_context,
                "teacher_context": teacher_context,
            },
            description=description,
            keywords=["context", prompt_role, "auto-generated"],
        )

    def get_or_create_context_combo(
        self,
        base_combo_id: str,
        student_context: dict[str, Any],
        teacher_context: dict[str, Any],
        runtime_placeholders: tuple[str, ...] = ("question",),
    ) -> dict[str, Any]:
        """
        Create or reuse a contextual combo.

        The base combo remains clean. Generated child prompts contain the
        current context additions in the actual .md files.
        """

        base_combo = self.get_combo(base_combo_id)
        context_payload = {
            "render_version": self.CONTEXT_RENDER_VERSION,
            "base_combo_id": base_combo_id,
            "student_context": student_context,
            "teacher_context": teacher_context,
            "runtime_placeholders": runtime_placeholders,
        }
        context_hash = self._hash_data(context_payload)

        for combo in self.list_combos():
            metadata = combo.get("metadata") or {}
            if (
                metadata.get("kind") == "contextual_combo"
                and metadata.get("base_combo_id") == base_combo_id
                and metadata.get("context_hash") == context_hash
            ):
                return combo

        contextual_prompt_ids: dict[str, str] = {}

        for role, prompt_id in base_combo["prompt_ids"].items():
            if role in {"system", "user"}:
                node = self.get_or_create_context_node(
                    parent_id=prompt_id,
                    prompt_role=role,
                    student_context=student_context,
                    teacher_context=teacher_context,
                    title=f"Contextual {role} prompt",
                    runtime_placeholders=runtime_placeholders,
                )
                contextual_prompt_ids[role] = node["id"]
            else:
                contextual_prompt_ids[role] = prompt_id

        return self.create_combo(
            title=f"Contextual combo from {base_combo_id} ({context_hash})",
            prompt_ids=contextual_prompt_ids,
            status="contextual",
            test_status="untested",
            score=None,
            notes="Auto-generated from student and teacher context.",
            tags=["context", "auto-generated"],
            metadata={
                "kind": "contextual_combo",
                "base_combo_id": base_combo_id,
                "context_hash": context_hash,
                "render_version": self.CONTEXT_RENDER_VERSION,
                "student_context": student_context,
                "teacher_context": teacher_context,
            },
        )

    # ------------------------------------------------------------------
    # Tree navigation
    # ------------------------------------------------------------------

    def get_children(
        self,
        prompt_id: str,
    ) -> list[dict[str, Any]]:
        return sorted(
            [
                node for node in self.list_nodes()
                if node["parent_id"] == prompt_id
            ],
            key=lambda node: node["created_at"],
        )

    def get_lineage(
        self,
        prompt_id: str,
    ) -> list[dict[str, Any]]:
        lineage = []
        current = self.get_node(prompt_id)

        while True:
            lineage.append(current)
            parent_id = current.get("parent_id")

            if parent_id is None:
                break

            current = self.get_node(parent_id)

        return list(reversed(lineage))

    def show_tree(
        self,
        tree_id: str,
    ) -> None:
        nodes = self.list_nodes(tree_id=tree_id)
        by_parent: dict[str | None, list[dict[str, Any]]] = {}

        for node in nodes:
            by_parent.setdefault(node["parent_id"], []).append(node)

        def walk(parent_id: str | None, level: int = 0) -> None:
            for child in sorted(
                by_parent.get(parent_id, []),
                key=lambda node: node["created_at"],
            ):
                print(f"{'    ' * level}- {child['id']} [{child['branch']}] {child['title']}")
                walk(child["id"], level + 1)

        walk(None)

    def prompt_combo_ids(
        self,
        prompt_id: str,
    ) -> list[str]:
        self.get_node(prompt_id)
        return sorted(
            combo["id"]
            for combo in self.list_combos()
            if prompt_id in (combo.get("prompt_ids") or {}).values()
        )

    def prompt_experiment_ids(
        self,
        prompt_id: str,
    ) -> list[str]:
        combo_ids = set(self.prompt_combo_ids(prompt_id))
        experiment_ids: set[str] = set()

        if not combo_ids:
            return []

        for experiment_node in self.list_experiments():
            experiment = self.get_experiment(experiment_node["id"])
            if combo_ids.intersection(experiment.get("combo_ids", [])):
                experiment_ids.add(experiment["id"])

        return sorted(experiment_ids)

    def inspect_prompt_dependencies(
        self,
        prompt_id: str,
    ) -> dict[str, Any]:
        node = self.get_node(prompt_id)
        child_ids = [
            child["id"]
            for child in self.get_children(prompt_id)
        ]
        combo_ids = self.prompt_combo_ids(prompt_id)
        experiment_ids = self.prompt_experiment_ids(prompt_id)
        is_archived = self._is_archived_payload(node)
        blockers: list[str] = []

        if not is_archived:
            blockers.append("not_archived")
        if child_ids:
            blockers.append("has_child_prompts")
        if combo_ids:
            blockers.append("used_by_combos")

        return {
            "entity": "prompt",
            "id": prompt_id,
            "type": node.get("type"),
            "tree_id": node.get("tree_id"),
            "title": node.get("title"),
            "path": node.get("path"),
            "parent_id": node.get("parent_id"),
            "child_prompt_ids": child_ids,
            "combo_ids": combo_ids,
            "experiment_ids": experiment_ids,
            "is_archived": is_archived,
            "blockers": blockers,
            "safe_to_delete": not blockers,
        }

    def describe_prompt_dependencies(
        self,
        prompt_id: str,
    ) -> dict[str, Any]:
        """Return operator-readable prompt usage and delete-safety details."""

        def _count_phrase(
            count: int,
            singular: str,
            plural: str | None = None,
        ) -> str:
            noun = singular if count == 1 else (plural or f"{singular}s")
            return f"{count} {noun}"

        report = self.inspect_prompt_dependencies(prompt_id)
        child_ids = list(report.get("child_prompt_ids", []))
        combo_ids = list(report.get("combo_ids", []))
        experiment_ids = list(report.get("experiment_ids", []))

        usage_rows = [
            {
                "code": "child_prompts",
                "label": "Child prompts",
                "count": len(child_ids),
                "ids": child_ids,
                "message": (
                    f"{_count_phrase(len(child_ids), 'child prompt')} branch from this prompt."
                    if child_ids else
                    "No child prompts branch from this prompt."
                ),
            },
            {
                "code": "combos",
                "label": "Combos",
                "count": len(combo_ids),
                "ids": combo_ids,
                "message": (
                    f"This prompt is used by {_count_phrase(len(combo_ids), 'combo')}."
                    if combo_ids else
                    "This prompt is not currently used by any combo."
                ),
            },
            {
                "code": "experiments",
                "label": "Experiments",
                "count": len(experiment_ids),
                "ids": experiment_ids,
                "message": (
                    f"Those combos appear in {_count_phrase(len(experiment_ids), 'experiment')}."
                    if experiment_ids else
                    "No experiment currently references this prompt through a combo."
                ),
            },
        ]

        blocker_rows: list[dict[str, Any]] = []
        if "not_archived" in report.get("blockers", []):
            blocker_rows.append({
                "code": "not_archived",
                "label": "Archive first",
                "message": (
                    "This prompt is not archived yet. "
                    "Permanent delete stays disabled until it is archived."
                ),
                "count": 1,
                "ids": [],
            })
        if "has_child_prompts" in report.get("blockers", []):
            blocker_rows.append({
                "code": "has_child_prompts",
                "label": "Child prompts still depend on it",
                "message": (
                    f"{_count_phrase(len(child_ids), 'child prompt')} still branch from this prompt."
                ),
                "count": len(child_ids),
                "ids": child_ids,
            })
        if "used_by_combos" in report.get("blockers", []):
            blocker_rows.append({
                "code": "used_by_combos",
                "label": "Combos still reference it",
                "message": (
                    f"{_count_phrase(len(combo_ids), 'combo')} still include this prompt"
                    + (
                        f" across {_count_phrase(len(experiment_ids), 'experiment')}."
                        if experiment_ids else "."
                    )
                ),
                "count": len(combo_ids),
                "ids": combo_ids,
                "related_experiment_ids": experiment_ids,
            })

        if report.get("safe_to_delete"):
            delete_headline = "This prompt can be deleted safely."
            delete_summary = (
                "It is already archived and no child prompts or combos still reference it."
            )
            recommended_actions: list[str] = []
        else:
            delete_headline = (
                "Delete is blocked until this prompt is archived and the remaining dependencies are cleared."
                if not report.get("is_archived") else
                "Delete is blocked until the remaining dependencies are cleared."
            )
            summary_parts: list[str] = []
            recommended_actions = []
            if not report.get("is_archived"):
                summary_parts.append("Archive it first.")
                recommended_actions.append(
                    "Archive this prompt before attempting permanent delete."
                )
            if child_ids:
                summary_parts.append(
                    f"Resolve {_count_phrase(len(child_ids), 'child prompt')}."
                )
                recommended_actions.append(
                    f"Rebranch, archive, or remove {_count_phrase(len(child_ids), 'child prompt')} that still depend on this prompt."
                )
            if combo_ids:
                summary_parts.append(
                    f"Remove or replace it in {_count_phrase(len(combo_ids), 'combo')}."
                )
                recommended_actions.append(
                    f"Remove or replace this prompt in {_count_phrase(len(combo_ids), 'combo')} before deleting it."
                )
            delete_summary = " ".join(summary_parts)

        if child_ids or combo_ids or experiment_ids:
            usage_headline = (
                f"This prompt is connected to {_count_phrase(len(combo_ids), 'combo')}"
                + (
                    f" across {_count_phrase(len(experiment_ids), 'experiment')}"
                    if combo_ids or experiment_ids else ""
                )
                + (
                    f" and {_count_phrase(len(child_ids), 'child prompt')}."
                    if child_ids else "."
                )
            )
        else:
            usage_headline = (
                "This prompt currently has no child prompts, combos, or experiments attached."
            )

        return {
            **report,
            "usage": {
                "headline": usage_headline,
                "rows": usage_rows,
            },
            "delete_safety": {
                "status": (
                    "safe" if report.get("safe_to_delete") else "blocked"
                ),
                "headline": delete_headline,
                "summary": delete_summary,
                "blocker_rows": blocker_rows,
                "recommended_actions": recommended_actions,
            },
        }

    def archive_prompt(
        self,
        prompt_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        nodes = self.list_nodes()
        updated_node: dict[str, Any] | None = None

        for node in nodes:
            if node["id"] != prompt_id:
                continue
            node["metadata"] = self._archive_metadata(
                node.get("metadata"),
                reason=reason,
            )
            node["tags"] = self._ensure_tag(
                node.get("tags"),
                "archived",
            )
            node["updated_at"] = self._now()
            updated_node = node
            break

        if updated_node is None:
            raise KeyError(f"Prompt node not found: {prompt_id}")

        self._save_nodes(nodes)
        return updated_node

    def delete_prompt(
        self,
        prompt_id: str,
    ) -> dict[str, Any]:
        report = self.inspect_prompt_dependencies(prompt_id)
        if not report["safe_to_delete"]:
            raise ValueError(
                "Prompt cannot be deleted safely: "
                + ", ".join(report["blockers"])
            )

        node = self.get_node(prompt_id)
        nodes = [
            existing
            for existing in self.list_nodes()
            if existing["id"] != prompt_id
        ]
        self._save_nodes(nodes)
        self._remove_edges(from_id=prompt_id)
        self._remove_edges(to_id=prompt_id)
        self._delete_file_if_present(self.root / node["path"])

        return {
            "deleted_id": prompt_id,
            "deleted_path": node.get("path"),
            "entity": "prompt",
        }

    # ------------------------------------------------------------------
    # Diffs
    # ------------------------------------------------------------------

    def unified_diff_text(
        self,
        old_id: str,
        new_id: str,
    ) -> str:
        old = self.read_prompt(old_id).splitlines(keepends=True)
        new = self.read_prompt(new_id).splitlines(keepends=True)

        return "".join(
            unified_diff(
                old,
                new,
                fromfile=old_id,
                tofile=new_id,
                lineterm="",
            )
        )

    def show_diff(
        self,
        old_id: str,
        new_id: str,
        html: bool = True,
    ) -> None:
        old_text = self.read_prompt(old_id)
        new_text = self.read_prompt(new_id)

        if not html:
            print(self.unified_diff_text(old_id, new_id))
            return

        try:
            from IPython.display import HTML, display
        except ImportError:
            print(self.unified_diff_text(old_id, new_id))
            return

        html_diff = HtmlDiff(wrapcolumn=100).make_file(
            old_text.splitlines(),
            new_text.splitlines(),
            fromdesc=html_lib.escape(old_id),
            todesc=html_lib.escape(new_id),
        )
        display(HTML(html_diff))

    # ------------------------------------------------------------------
    # Combos
    # ------------------------------------------------------------------

    def _combo_key(
        self,
        prompt_ids: dict[str, str],
    ) -> str:
        return self._hash_data({
            "prompt_ids": prompt_ids,
        })

    def combo_prompt_stats(
        self,
        combo_id: str,
    ) -> dict[str, Any]:
        combo = self.get_combo(combo_id)
        prompt_stats: dict[str, Any] = {}
        total_words = 0
        total_chars = 0
        total_sentences = 0

        for role, prompt_id in combo["prompt_ids"].items():
            node = self.get_node(prompt_id)
            stats = node.get("stats") or self.compute_prompt_stats(
                self.read_prompt(prompt_id)
            )

            prompt_stats[role] = {
                "prompt_id": prompt_id,
                **stats,
            }

            total_words += stats.get("word_count", 0)
            total_chars += stats.get("char_count", 0)
            total_sentences += stats.get("sentence_count", 0)

        return {
            "roles": prompt_stats,
            "total_word_count": total_words,
            "total_char_count": total_chars,
            "total_sentence_count": total_sentences,
        }

    def _existing_combo_by_key(
        self,
        combo_key: str,
    ) -> dict[str, Any] | None:
        for combo in self.list_combos():
            metadata = combo.get("metadata") or {}
            if metadata.get("combo_key") == combo_key:
                return combo
        return None

    def find_combo_by_prompt_ids(
        self,
        prompt_ids: dict[str, str],
    ) -> dict[str, Any] | None:
        combo_key = self._combo_key(prompt_ids)
        return self._existing_combo_by_key(combo_key)

    def create_combo(
        self,
        title: str,
        prompt_ids: dict[str, str],
        status: str = "untested",
        test_status: str = "untested",
        score: float | None = None,
        notes: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for prompt_id in prompt_ids.values():
            self.get_node(prompt_id)

        combo_key = self._combo_key(prompt_ids)
        final_metadata = metadata.copy() if metadata else {}
        final_metadata.setdefault("combo_key", combo_key)

        combo = {
            "id": self._next_combo_id(),
            "title": title,
            "prompt_ids": prompt_ids,
            "status": status,
            "test_status": test_status,
            "score": score,
            "notes": notes,
            "tags": tags or [],
            "metadata": final_metadata,
            "stats": {
                **self.combo_prompt_stats_from_prompt_ids(prompt_ids),
                "version": "combo-stats-v1",
            },
            "created_at": self._now(),
        }

        self._append_jsonl(self.combos_path, combo)
        return combo

    def combo_prompt_stats_from_prompt_ids(
        self,
        prompt_ids: dict[str, str],
    ) -> dict[str, Any]:
        prompt_stats: dict[str, Any] = {}
        total_words = 0
        total_chars = 0
        total_sentences = 0

        for role, prompt_id in prompt_ids.items():
            node = self.get_node(prompt_id)
            stats = node.get("stats") or self.compute_prompt_stats(
                self.read_prompt(prompt_id)
            )
            prompt_stats[role] = {
                "prompt_id": prompt_id,
                **stats,
            }
            total_words += stats.get("word_count", 0)
            total_chars += stats.get("char_count", 0)
            total_sentences += stats.get("sentence_count", 0)

        return {
            "roles": prompt_stats,
            "total_word_count": total_words,
            "total_char_count": total_chars,
            "total_sentence_count": total_sentences,
        }

    def generate_combos(
        self,
        roles_to_prompt_type: dict[str, str] | None = None,
        title_prefix: str = "Auto combo",
        status: str = "untested",
        test_status: str = "untested",
        skip_existing: bool = True,
        include_context_variants: bool = True,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate all combinations of prompt nodes by role.

        Default:
            system prompts x user prompts

        Example:
            generate_combos({
                "system": "system",
                "user": "user",
                "safety": "safety",
            })
        """

        roles_to_prompt_type = roles_to_prompt_type or {
            "system": "system",
            "user": "user",
        }

        role_names = list(roles_to_prompt_type.keys())
        pools: list[list[dict[str, Any]]] = []

        for role in role_names:
            prompt_type = roles_to_prompt_type[role]
            nodes = self.list_nodes(prompt_type=prompt_type)

            if not include_context_variants:
                nodes = [
                    node for node in nodes
                    if (node.get("metadata") or {}).get("kind") != "context_variant"
                ]

            nodes = sorted(nodes, key=lambda node: node["id"])
            pools.append(nodes)

        created: list[dict[str, Any]] = []

        for node_tuple in product(*pools):
            prompt_ids = {
                role: node["id"]
                for role, node in zip(role_names, node_tuple)
            }

            combo_key = self._combo_key(prompt_ids)
            existing = self._existing_combo_by_key(combo_key)

            if skip_existing and existing is not None:
                continue

            title_parts = [
                f"{role}={prompt_id}"
                for role, prompt_id in prompt_ids.items()
            ]

            combo = self.create_combo(
                title=f"{title_prefix}: " + " + ".join(title_parts),
                prompt_ids=prompt_ids,
                status=status,
                test_status=test_status,
                score=None,
                notes="Generated automatically from prompt pools.",
                tags=["auto-generated", "untested", *(tags or [])],
                metadata={
                    "kind": "auto_generated_combo",
                    "combo_key": combo_key,
                    "generator_version": self.COMBO_GENERATOR_VERSION,
                    "roles_to_prompt_type": roles_to_prompt_type,
                },
            )

            created.append(combo)

        return created

    def list_combos(
        self,
        status: str | None = None,
        test_status: str | None = None,
    ) -> list[dict[str, Any]]:
        combos = self._read_jsonl(self.combos_path)

        if status is not None:
            combos = [
                combo for combo in combos
                if combo.get("status") == status
            ]

        if test_status is not None:
            combos = [
                combo for combo in combos
                if combo.get("test_status") == test_status
            ]

        return combos

    def get_combo(
        self,
        combo_id: str,
    ) -> dict[str, Any]:
        for combo in self.list_combos():
            if combo["id"] == combo_id:
                return combo

        raise KeyError(f"Combo not found: {combo_id}")

    def read_combo_prompts(
        self,
        combo_id: str,
    ) -> dict[str, str]:
        combo = self.get_combo(combo_id)

        return {
            role: self.read_prompt(prompt_id)
            for role, prompt_id in combo["prompt_ids"].items()
        }

    def show_combo(
        self,
        combo_id: str,
    ) -> None:
        combo = self.get_combo(combo_id)
        print(f"{combo['id']} | {combo['title']}")
        print(f"status: {combo.get('status')} | test_status: {combo.get('test_status')} | score: {combo.get('score')}")
        print(f"notes: {combo.get('notes', '')}")
        print("prompts:")
        for role, prompt_id in combo["prompt_ids"].items():
            node = self.get_node(prompt_id)
            stats = node.get("stats") or {}
            print(
                f"  - {role}: {prompt_id} | {node['title']} "
                f"| words={stats.get('word_count')}"
            )

    def update_combo_score(
        self,
        combo_id: str,
        score: float,
        status: str | None = None,
        test_status: str | None = "tested",
        notes: str | None = None,
    ) -> dict[str, Any]:
        combos = self.list_combos()
        updated = None

        for combo in combos:
            if combo["id"] == combo_id:
                combo["score"] = score
                combo["updated_at"] = self._now()
                combo["stats"] = {
                    **self.combo_prompt_stats_from_prompt_ids(combo["prompt_ids"]),
                    "version": "combo-stats-v1",
                }

                if status is not None:
                    combo["status"] = status

                if test_status is not None:
                    combo["test_status"] = test_status

                if notes is not None:
                    combo["notes"] = notes

                updated = combo
                break

        if updated is None:
            raise KeyError(f"Combo not found: {combo_id}")

        self._write_jsonl(self.combos_path, combos)
        return updated

    def best_combos(
        self,
        min_score: float = 0.0,
        status: str | None = None,
        test_status: str | None = None,
    ) -> list[dict[str, Any]]:
        combos = self.list_combos(
            status=status,
            test_status=test_status,
        )

        filtered = [
            combo for combo in combos
            if combo.get("score") is not None
            and combo["score"] >= min_score
        ]

        return sorted(
            filtered,
            key=lambda combo: combo["score"],
            reverse=True,
        )

    def list_untested_combos(
        self,
        only_unassigned: bool = False,
    ) -> list[dict[str, Any]]:
        combos = self.list_combos(test_status="untested")

        if not only_unassigned:
            return combos

        used_combo_ids = {
            edge["to"]
            for edge in self.list_edges(kind="experiment_uses_combo")
        }

        return [
            combo for combo in combos
            if combo["id"] not in used_combo_ids
        ]

    def combos_not_in_experiment(
        self,
        experiment_id: str,
        test_status: str | None = None,
    ) -> list[dict[str, Any]]:
        attached_ids = set(self.experiment_combo_ids(experiment_id))
        combos = self.list_combos(test_status=test_status)

        return [
            combo for combo in combos
            if combo["id"] not in attached_ids
        ]

    def derived_combo_ids(
        self,
        combo_id: str,
    ) -> list[str]:
        self.get_combo(combo_id)
        return sorted(
            combo["id"]
            for combo in self.list_combos()
            if (combo.get("metadata") or {}).get("base_combo_id") == combo_id
        )

    def combo_experiment_ids(
        self,
        combo_id: str,
    ) -> list[str]:
        self.get_combo(combo_id)
        experiment_ids: set[str] = set()

        for experiment_node in self.list_experiments():
            experiment = self.get_experiment(experiment_node["id"])
            if combo_id in experiment.get("combo_ids", []):
                experiment_ids.add(experiment["id"])

        return sorted(experiment_ids)

    def combo_result_experiment_ids(
        self,
        combo_id: str,
    ) -> list[str]:
        self.get_combo(combo_id)
        experiment_ids: set[str] = set()

        for experiment_node in self.list_experiments():
            experiment = self.get_experiment(experiment_node["id"])
            if combo_id in {
                result["combo_id"]
                for result in experiment.get("results", [])
            }:
                experiment_ids.add(experiment["id"])

        return sorted(experiment_ids)

    def inspect_combo_dependencies(
        self,
        combo_id: str,
    ) -> dict[str, Any]:
        combo = self.get_combo(combo_id)
        experiment_ids = self.combo_experiment_ids(combo_id)
        result_experiment_ids = self.combo_result_experiment_ids(combo_id)
        run_rows = self.list_runs(combo_id=combo_id)
        run_ids = [run["id"] for run in run_rows]
        compact_result_rows = [
            row for row in self._read_jsonl(self.experiments_path)
            if row.get("combo_id") == combo_id
        ]
        artifact_paths = self._combo_artifact_paths(combo_id)
        derived_combo_ids = self.derived_combo_ids(combo_id)
        is_archived = self._is_archived_payload(
            combo,
            status_field="status",
        )
        blockers: list[str] = []

        if not is_archived:
            blockers.append("not_archived")
        if experiment_ids:
            blockers.append("used_by_experiments")
        if result_experiment_ids:
            blockers.append("has_recorded_results")
        if run_ids:
            blockers.append("has_recorded_runs")
        if compact_result_rows:
            blockers.append("has_compact_result_rows")
        if derived_combo_ids:
            blockers.append("has_derived_combos")
        if artifact_paths["raw_artifact_paths"]:
            blockers.append("has_raw_artifacts")
        if artifact_paths["normalized_artifact_paths"]:
            blockers.append("has_normalized_artifacts")

        return {
            "entity": "combo",
            "id": combo_id,
            "title": combo.get("title"),
            "prompt_ids": dict(combo.get("prompt_ids", {})),
            "experiment_ids": experiment_ids,
            "result_experiment_ids": result_experiment_ids,
            "run_ids": run_ids,
            "compact_result_row_count": len(compact_result_rows),
            "derived_combo_ids": derived_combo_ids,
            "raw_artifact_paths": artifact_paths["raw_artifact_paths"],
            "normalized_artifact_paths": artifact_paths["normalized_artifact_paths"],
            "is_archived": is_archived,
            "blockers": blockers,
            "safe_to_delete": not blockers,
        }

    def preview_detach_combo_from_experiment(
        self,
        experiment_id: str,
        combo_id: str,
    ) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        combo_exists = any(
            combo["id"] == combo_id
            for combo in self.list_combos()
        )
        attached = combo_id in experiment.get("combo_ids", [])
        result_rows = [
            result for result in experiment.get("results", [])
            if result.get("combo_id") == combo_id
        ]
        run_rows = self.list_runs(
            combo_id=combo_id,
            experiment_id=experiment_id,
        )
        compact_result_rows = [
            row for row in self._read_jsonl(self.experiments_path)
            if row.get("experiment_id") == experiment_id
            and row.get("combo_id") == combo_id
        ]
        raw_scope_dir = self.raw_runs_dir / experiment_id
        normalized_scope_dir = self.normalized_runs_dir / experiment_id
        raw_artifact_paths = sorted(
            self._relative(path)
            for path in raw_scope_dir.glob(f"*{combo_id}*.json")
            if path.is_file()
        )
        normalized_artifact_paths = sorted(
            self._relative(path)
            for path in normalized_scope_dir.glob(f"*{combo_id}*.json")
            if path.is_file()
        )
        blockers: list[str] = []

        if not attached:
            blockers.append("combo_not_attached")
        if result_rows:
            blockers.append("has_recorded_results")
        if run_rows:
            blockers.append("has_recorded_runs")
        if compact_result_rows:
            blockers.append("has_compact_result_rows")
        if raw_artifact_paths:
            blockers.append("has_raw_artifacts")
        if normalized_artifact_paths:
            blockers.append("has_normalized_artifacts")

        return {
            "entity": "experiment_combo_link",
            "experiment_id": experiment_id,
            "combo_id": combo_id,
            "combo_exists": combo_exists,
            "attached": attached,
            "result_count": len(result_rows),
            "run_ids": [run["id"] for run in run_rows],
            "compact_result_row_count": len(compact_result_rows),
            "raw_artifact_paths": raw_artifact_paths,
            "normalized_artifact_paths": normalized_artifact_paths,
            "blockers": blockers,
            "safe_to_detach": not blockers,
        }

    def detach_combo_from_experiment(
        self,
        experiment_id: str,
        combo_id: str,
    ) -> dict[str, Any]:
        preview = self.preview_detach_combo_from_experiment(
            experiment_id=experiment_id,
            combo_id=combo_id,
        )
        if not preview["safe_to_detach"]:
            raise ValueError(
                "Combo cannot be detached safely: "
                + ", ".join(preview["blockers"])
            )

        experiment = self.get_experiment(experiment_id)
        experiment["combo_ids"] = [
            attached_id
            for attached_id in experiment.get("combo_ids", [])
            if attached_id != combo_id
        ]
        experiment["updated_at"] = self._now()
        self._save_experiment(experiment)
        self._remove_edges(
            from_id=experiment_id,
            to_id=combo_id,
            kinds={"experiment_uses_combo", "experiment_tested_combo"},
        )
        return self.get_experiment(experiment_id)

    def archive_combo(
        self,
        combo_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        combos = self.list_combos()
        updated_combo: dict[str, Any] | None = None

        for combo in combos:
            if combo["id"] != combo_id:
                continue
            previous_status = combo.get("status")
            previous_test_status = combo.get("test_status")
            combo["status"] = "archived"
            combo["test_status"] = "archived"
            combo["tags"] = self._ensure_tag(
                combo.get("tags"),
                "archived",
            )
            combo["metadata"] = self._archive_metadata(
                combo.get("metadata"),
                reason=reason,
                extra_fields={
                    "archived_from_status": previous_status,
                    "archived_from_test_status": previous_test_status,
                },
            )
            combo["updated_at"] = self._now()
            updated_combo = combo
            break

        if updated_combo is None:
            raise KeyError(f"Combo not found: {combo_id}")

        self._save_combos(combos)
        return updated_combo

    def delete_combo(
        self,
        combo_id: str,
    ) -> dict[str, Any]:
        report = self.inspect_combo_dependencies(combo_id)
        if not report["safe_to_delete"]:
            raise ValueError(
                "Combo cannot be deleted safely: "
                + ", ".join(report["blockers"])
            )

        combos = [
            combo for combo in self.list_combos()
            if combo["id"] != combo_id
        ]
        self._save_combos(combos)
        self._remove_edges(from_id=combo_id)
        self._remove_edges(to_id=combo_id)
        self._save_compact_experiment_rows([
            row for row in self._read_jsonl(self.experiments_path)
            if row.get("combo_id") != combo_id
        ])
        self._save_runs([
            row for row in self._read_jsonl(self.runs_path)
            if row.get("combo_id") != combo_id
        ])

        return {
            "deleted_id": combo_id,
            "entity": "combo",
        }

    # ------------------------------------------------------------------
    # Experiments as graph nodes
    # ------------------------------------------------------------------

    def _experiment_file_path(
        self,
        experiment_id: str,
    ) -> Path:
        return self.experiments_dir / f"{experiment_id}.json"

    def create_experiment(
        self,
        name: str,
        goal: str,
        hypothesis: str,
        notes: str = "",
        tags: list[str] | None = None,
        status: str = "planned",
        combo_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Initialize an experiment before running it.

        name, goal and hypothesis are intentionally required.
        This makes the experiment explicit before results enter the registry.
        """

        if not name.strip():
            raise ValueError("Experiment name is required.")

        if not goal.strip():
            raise ValueError("Experiment goal is required before registering the experiment.")

        if not hypothesis.strip():
            raise ValueError("Experiment hypothesis is required before registering the experiment.")

        experiment_id = self._next_experiment_id()
        now = self._now()

        node = {
            "id": experiment_id,
            "type": "experiment",
            "name": name.strip(),
            "goal": goal.strip(),
            "hypothesis": hypothesis.strip(),
            "notes": notes.strip(),
            "tags": tags or [],
            "status": status,
            "created_at": now,
            "path": self._relative(self._experiment_file_path(experiment_id)),
        }

        experiment_object = {
            **node,
            "combo_ids": [],
            "results": [],
            "summary": {},
            "subjective_summary": {},
            "metadata": {
                "schema_version": "prompt-garden-experiment-v1",
            },
        }

        self._append_jsonl(self.experiment_nodes_path, node)
        self._experiment_file_path(experiment_id).write_text(
            json.dumps(experiment_object, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if combo_ids:
            self.attach_combos_to_experiment(
                experiment_id=experiment_id,
                combo_ids=combo_ids,
            )

        return self.get_experiment(experiment_id)

    def list_experiments(
        self,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        experiments = self._read_jsonl(self.experiment_nodes_path)

        if status is not None:
            experiments = [
                experiment for experiment in experiments
                if experiment.get("status") == status
            ]

        return experiments

    def get_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        path = self._experiment_file_path(experiment_id)

        if not path.exists():
            raise KeyError(f"Experiment not found: {experiment_id}")

        return json.loads(path.read_text(encoding="utf-8"))

    def _save_experiment(
        self,
        experiment: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._experiment_file_path(experiment["id"])
        path.write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Keep index row in sync for status / notes.
        rows = self._read_jsonl(self.experiment_nodes_path)
        for row in rows:
            if row["id"] == experiment["id"]:
                for key in [
                    "name",
                    "goal",
                    "hypothesis",
                    "notes",
                    "tags",
                    "status",
                    "updated_at",
                ]:
                    if key in experiment:
                        row[key] = experiment[key]
                break

        self._write_jsonl(self.experiment_nodes_path, rows)
        return experiment

    def update_experiment_metadata(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        goal: str | None = None,
        hypothesis: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)

        if name is not None:
            if not name.strip():
                raise ValueError("Experiment name cannot be empty.")
            experiment["name"] = name.strip()
        if goal is not None:
            if not goal.strip():
                raise ValueError("Experiment goal cannot be empty.")
            experiment["goal"] = goal.strip()
        if hypothesis is not None:
            if not hypothesis.strip():
                raise ValueError("Experiment hypothesis cannot be empty.")
            experiment["hypothesis"] = hypothesis.strip()
        if notes is not None:
            experiment["notes"] = notes.strip()
        if tags is not None:
            experiment["tags"] = list(tags)
        if status is not None:
            experiment["status"] = status

        experiment["updated_at"] = self._now()
        return self._save_experiment(experiment)

    def inspect_experiment_dependencies(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        combo_ids = list(experiment.get("combo_ids", []))
        result_combo_ids = [
            result["combo_id"]
            for result in experiment.get("results", [])
        ]
        run_rows = self.list_runs(experiment_id=experiment_id)
        compact_result_rows = [
            row for row in self._read_jsonl(self.experiments_path)
            if row.get("experiment_id") == experiment_id
        ]
        artifact_paths = self._scope_artifact_paths(experiment_id)
        is_archived = self._is_archived_payload(
            experiment,
            status_field="status",
        )
        blockers: list[str] = []

        if not is_archived:
            blockers.append("not_archived")
        if result_combo_ids:
            blockers.append("has_recorded_results")
        if run_rows:
            blockers.append("has_recorded_runs")
        if compact_result_rows:
            blockers.append("has_compact_result_rows")
        if artifact_paths["raw_artifact_paths"]:
            blockers.append("has_raw_artifacts")
        if artifact_paths["normalized_artifact_paths"]:
            blockers.append("has_normalized_artifacts")
        if artifact_paths["report_file_paths"]:
            blockers.append("has_report_files")

        return {
            "entity": "experiment",
            "id": experiment_id,
            "name": experiment.get("name"),
            "status": experiment.get("status"),
            "combo_ids": combo_ids,
            "result_combo_ids": result_combo_ids,
            "run_ids": [run["id"] for run in run_rows],
            "compact_result_row_count": len(compact_result_rows),
            "raw_artifact_paths": artifact_paths["raw_artifact_paths"],
            "normalized_artifact_paths": artifact_paths["normalized_artifact_paths"],
            "report_file_paths": artifact_paths["report_file_paths"],
            "is_archived": is_archived,
            "blockers": blockers,
            "safe_to_delete": not blockers,
        }

    def archive_experiment(
        self,
        experiment_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        previous_status = experiment.get("status")
        experiment["status"] = "archived"
        experiment["tags"] = self._ensure_tag(
            experiment.get("tags"),
            "archived",
        )
        experiment["metadata"] = self._archive_metadata(
            experiment.get("metadata"),
            reason=reason,
            extra_fields={
                "archived_from_status": previous_status,
            },
        )
        experiment["updated_at"] = self._now()
        return self._save_experiment(experiment)

    def attach_combo_to_experiment(
        self,
        experiment_id: str,
        combo_id: str,
        role: str = "candidate",
        notes: str = "",
    ) -> dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        self.get_combo(combo_id)

        if combo_id not in experiment["combo_ids"]:
            experiment["combo_ids"].append(combo_id)

        if not self._edge_exists(
            from_id=experiment_id,
            to_id=combo_id,
            kind="experiment_uses_combo",
        ):
            self._append_edge(
                from_id=experiment_id,
                to_id=combo_id,
                kind="experiment_uses_combo",
                metadata={
                    "role": role,
                    "notes": notes,
                },
            )

        experiment["updated_at"] = self._now()
        return self._save_experiment(experiment)

    def attach_combos_to_experiment(
        self,
        experiment_id: str,
        combo_ids: list[str],
        role: str = "candidate",
        notes: str = "",
    ) -> dict[str, Any]:
        for combo_id in combo_ids:
            self.attach_combo_to_experiment(
                experiment_id=experiment_id,
                combo_id=combo_id,
                role=role,
                notes=notes,
            )

        return self.get_experiment(experiment_id)

    def delete_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        report = self.inspect_experiment_dependencies(experiment_id)
        if not report["safe_to_delete"]:
            raise ValueError(
                "Experiment cannot be deleted safely: "
                + ", ".join(report["blockers"])
            )

        experiment_path = self._experiment_file_path(experiment_id)
        experiment_rows = [
            row for row in self._read_jsonl(self.experiment_nodes_path)
            if row.get("id") != experiment_id
        ]
        self._write_jsonl(self.experiment_nodes_path, experiment_rows)
        self._remove_edges(from_id=experiment_id)
        self._remove_edges(to_id=experiment_id)
        self._save_compact_experiment_rows([
            row for row in self._read_jsonl(self.experiments_path)
            if row.get("experiment_id") != experiment_id
        ])
        self._save_runs([
            row for row in self._read_jsonl(self.runs_path)
            if row.get("experiment_id") != experiment_id
        ])
        self._delete_file_if_present(experiment_path)

        self._remove_directory_if_empty(self.raw_runs_dir / experiment_id)
        self._remove_directory_if_empty(self.normalized_runs_dir / experiment_id)
        self._remove_directory_if_empty(self.reports_dir / experiment_id)

        return {
            "deleted_id": experiment_id,
            "deleted_path": self._relative(experiment_path),
            "entity": "experiment",
        }

    def experiment_combo_ids(
        self,
        experiment_id: str,
    ) -> list[str]:
        experiment = self.get_experiment(experiment_id)
        return list(experiment.get("combo_ids", []))

    def experiment_result_combo_ids(
        self,
        experiment_id: str,
    ) -> list[str]:
        experiment = self.get_experiment(experiment_id)
        return [
            result["combo_id"]
            for result in experiment.get("results", [])
        ]

    def list_experiment_untested_combos(
        self,
        experiment_id: str,
    ) -> list[dict[str, Any]]:
        result_combo_ids = set(
            self.experiment_result_combo_ids(experiment_id)
        )

        return [
            self.get_combo(combo_id)
            for combo_id in self.experiment_combo_ids(experiment_id)
            if combo_id not in result_combo_ids
        ]

    def record_experiment_combo_result(
        self,
        experiment_id: str,
        combo_id: str,
        score: float,
        result_text: str,
        subject_score: float | None = None,
        subjective_notes: str = "",
        metrics: dict[str, Any] | None = None,
        case_results: list[dict[str, Any]] | None = None,
        status: str = "tested",
    ) -> dict[str, Any]:
        """
        Store result for one combo inside an experiment object.

        result_text and subject_score are for human-visible judgement.
        metrics and case_results are for semi-automatic tests.
        """

        if not result_text.strip():
            raise ValueError("result_text is required. Describe what you observed.")

        experiment = self.get_experiment(experiment_id)
        self.get_combo(combo_id)

        if combo_id not in experiment.get("combo_ids", []):
            self.attach_combo_to_experiment(experiment_id, combo_id)
            experiment = self.get_experiment(experiment_id)

        result_record = {
            "combo_id": combo_id,
            "score": score,
            "result_text": result_text,
            "subject_score": subject_score,
            "subjective_notes": subjective_notes,
            "metrics": metrics or {},
            "case_results": case_results or [],
            "status": status,
            "prompt_stats": self.combo_prompt_stats(combo_id),
            "created_at": self._now(),
        }

        # Upsert by combo_id.
        results = [
            record for record in experiment.get("results", [])
            if record["combo_id"] != combo_id
        ]
        results.append(result_record)
        experiment["results"] = results
        experiment["updated_at"] = self._now()
        experiment["status"] = "running"

        scores = [record["score"] for record in results]
        subject_scores = [
            record["subject_score"]
            for record in results
            if record.get("subject_score") is not None
        ]

        experiment["summary"] = {
            "tested_combo_count": len(results),
            "attached_combo_count": len(experiment.get("combo_ids", [])),
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "best_score": max(scores) if scores else None,
            "worst_score": min(scores) if scores else None,
        }

        experiment["subjective_summary"] = {
            "average_subject_score": round(sum(subject_scores) / len(subject_scores), 4) if subject_scores else None,
            "subject_score_count": len(subject_scores),
        }

        self._save_experiment(experiment)

        combo_status = "stable" if score >= 0.8 else "needs_work"
        self.update_combo_score(
            combo_id=combo_id,
            score=score,
            status=combo_status,
            test_status="tested",
            notes=result_text,
        )

        if not self._edge_exists(
            from_id=experiment_id,
            to_id=combo_id,
            kind="experiment_tested_combo",
        ):
            self._append_edge(
                from_id=experiment_id,
                to_id=combo_id,
                kind="experiment_tested_combo",
                metadata={
                    "score": score,
                    "subject_score": subject_score,
                    "status": status,
                },
            )

        compact = {
            "experiment_id": experiment_id,
            "combo_id": combo_id,
            "score": score,
            "subject_score": subject_score,
            "result_text": result_text,
            "metrics": metrics or {},
            "created_at": result_record["created_at"],
        }
        self._append_jsonl(self.experiments_path, compact)

        return result_record

    def finalize_experiment(
        self,
        experiment_id: str,
        result_text: str,
        subject_score: float | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        if not result_text.strip():
            raise ValueError("Final result_text is required.")

        experiment = self.get_experiment(experiment_id)
        experiment["status"] = status
        experiment["final_result_text"] = result_text
        experiment["final_subject_score"] = subject_score
        experiment["updated_at"] = self._now()

        return self._save_experiment(experiment)

    # Backward-compatible helper from older notebook versions.
    def add_experiment(
        self,
        combo_id: str,
        task: str,
        model: str,
        result: str,
        score: float,
        notes: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        experiment = self.create_experiment(
            name=f"Legacy experiment for {combo_id}",
            goal=f"Evaluate combo on task: {task}",
            hypothesis="Legacy record imported through add_experiment().",
            notes=notes,
            tags=["legacy"],
            combo_ids=[combo_id],
        )
        self.record_experiment_combo_result(
            experiment_id=experiment["id"],
            combo_id=combo_id,
            score=score,
            result_text=result,
            subject_score=None,
            subjective_notes=notes,
            metrics={
                "task": task,
                "model": model,
                **(metrics or {}),
            },
        )
        return self.get_experiment(experiment["id"])

    def best_experiments(
        self,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        experiments = []

        for exp_node in self.list_experiments():
            exp = self.get_experiment(exp_node["id"])
            average_score = exp.get("summary", {}).get("average_score")

            if average_score is not None and average_score >= min_score:
                experiments.append(exp)

        return sorted(
            experiments,
            key=lambda exp: exp.get("summary", {}).get("average_score", 0.0),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def add_run(
        self,
        combo_id: str,
        task: str,
        model: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | str | None,
        validation_ok: bool,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        self.get_combo(combo_id)

        run = {
            "id": self._next_run_id(),
            "combo_id": combo_id,
            "experiment_id": experiment_id,
            "task": task,
            "model": model,
            "input_data": input_data,
            "output_data": output_data,
            "validation_ok": validation_ok,
            "error": error,
            "metrics": metrics or {},
            "created_at": self._now(),
        }

        self._append_jsonl(self.runs_path, run)
        return run

    def list_runs(
        self,
        combo_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        runs = self._read_jsonl(self.runs_path)

        if combo_id is not None:
            runs = [
                run for run in runs
                if run.get("combo_id") == combo_id
            ]

        if experiment_id is not None:
            runs = [
                run for run in runs
                if run.get("experiment_id") == experiment_id
            ]

        return runs

    # ------------------------------------------------------------------
    # Notebook summaries
    # ------------------------------------------------------------------

    def nodes_table(self) -> list[dict[str, Any]]:
        return self.list_nodes()

    def combos_table(self) -> list[dict[str, Any]]:
        return self.list_combos()

    def experiments_table(self) -> list[dict[str, Any]]:
        return self.list_experiments()

    def experiment_results_table(
        self,
        experiment_id: str,
    ) -> list[dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        rows = []

        for result in experiment.get("results", []):
            combo = self.get_combo(result["combo_id"])
            rows.append({
                "experiment_id": experiment_id,
                "combo_id": result["combo_id"],
                "combo_title": combo["title"],
                "score": result["score"],
                "subject_score": result.get("subject_score"),
                "result_text": result["result_text"],
                "status": result["status"],
                "total_words": result.get("prompt_stats", {}).get("total_word_count"),
                "total_chars": result.get("prompt_stats", {}).get("total_char_count"),
            })

        return rows

    def print_summary(self) -> None:
        print("PromptGarden")
        print(f"root: {self.root}")
        print(f"nodes: {len(self.list_nodes())}")
        print(f"combos: {len(self.list_combos())}")
        print(f"untested combos: {len(self.list_untested_combos())}")
        print(f"experiments: {len(self.list_experiments())}")
        print(f"runs: {len(self.list_runs())}")
