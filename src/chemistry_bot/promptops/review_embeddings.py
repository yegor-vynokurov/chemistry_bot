"""Lightweight embedding-style similarity helpers for Prompt Garden review."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

from .garden import PromptGarden
from .review_compare import (
    field_display_text,
    latest_records_by_case_combo,
    record_case_id,
    record_combo_id,
)
from .review_store import normalize_inline_text


DEFAULT_EMBEDDING_BACKEND = "token_hash_v1"
DEFAULT_EMBEDDING_DIMENSIONS = 256
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _record_id(record: Mapping[str, Any]) -> str:
    value = (
        record.get("id")
        or record.get("row_id")
        or f"{record_case_id(record)}::{record_combo_id(record)}"
    )
    return str(value)


def tokenize_for_embedding(text: Any) -> list[str]:
    """Tokenize text for the lightweight default embedding backend."""

    normalized = normalize_inline_text(text).lower()
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def embedding_cache_filename(
    scope: str,
    field_path: str,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    duplicate_threshold: float | None = None,
) -> str:
    """Return a stable embedding-cache filename for one similarity bundle."""

    same_case_token = "same_case" if same_case_only else "cross_case"
    return (
        f"{scope}__{backend}_{dimensions}d__"
        f"{same_case_token}__{_slugify(field_path)}.json"
        if duplicate_threshold is None
        else (
            f"{scope}__{backend}_{dimensions}d__"
            f"{same_case_token}__dup_{int(round(duplicate_threshold * 1000)):04d}__"
            f"{_slugify(field_path)}.json"
        )
    )


def read_similarity_cache(path: Path) -> dict[str, Any]:
    """Read one similarity-bundle cache file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_similarity_cache(
    cache_dir: Path,
    scope: str,
    bundle: Mapping[str, Any],
    field_path: str,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    duplicate_threshold: float | None = None,
) -> Path:
    """Write one similarity bundle into the Prompt Garden cache zone."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = embedding_cache_filename(
        scope=scope,
        field_path=field_path,
        backend=backend,
        dimensions=dimensions,
        same_case_only=same_case_only,
        duplicate_threshold=duplicate_threshold,
    )
    path = cache_dir / filename
    payload = {
        "schema_version": "prompt-garden-similarity-cache-v1",
        "scope": scope,
        "created_at": PromptGarden._now(),
        "embedding_backend": backend,
        "embedding_dimensions": dimensions,
        "field_path": field_path,
        "same_case_only": same_case_only,
        "duplicate_threshold": duplicate_threshold,
        "bundle": json.loads(
            json.dumps(
                bundle,
                ensure_ascii=False,
                default=str,
            )
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_similarity_cache(
    cache_dir: Path,
    scope: str,
    field_path: str,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    duplicate_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Load a similarity bundle from cache when it already exists."""

    path = cache_dir / embedding_cache_filename(
        scope=scope,
        field_path=field_path,
        backend=backend,
        dimensions=dimensions,
        same_case_only=same_case_only,
        duplicate_threshold=duplicate_threshold,
    )
    if not path.exists():
        return None
    return read_similarity_cache(path)


def token_hash_embedding(
    text: Any,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """
    Build a deterministic token-hash vector.

    This is a lightweight stand-in for real model embeddings so that
    similarity tooling works without extra services or large dependencies.
    """

    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    tokens = tokenize_for_embedding(text)
    if not tokens:
        return [0.0] * dimensions

    features = list(tokens)
    features.extend(
        f"{tokens[index]}__{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )
    vector = [0.0] * dimensions

    for feature in features:
        digest = hashlib.sha256(
            feature.encode("utf-8")
        ).digest()
        slot = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[slot] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * dimensions

    return [
        round(value / norm, 8)
        for value in vector
    ]


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """Return cosine similarity for two equally-sized vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("vector sizes must match")

    if not vector_a:
        return 0.0

    return round(
        sum(left * right for left, right in zip(vector_a, vector_b)),
        6,
    )


def embed_records(
    records: Sequence[Mapping[str, Any]],
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Embed review records with the lightweight default backend."""

    selected_records = (
        latest_records_by_case_combo(records)
        if latest_only else list(records)
    )
    embedded_rows: list[dict[str, Any]] = []

    for record in selected_records:
        text = field_display_text(record, field_path)
        vector = token_hash_embedding(
            text=text,
            dimensions=dimensions,
        )
        embedded_rows.append({
            "record_id": _record_id(record),
            "combo_id": record_combo_id(record),
            "case_id": record_case_id(record),
            "field_path": field_path,
            "text_hash": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16],
            "text_length": len(text),
            "vector": vector,
            "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
            "embedding_dimensions": dimensions,
            "source_record": record,
        })

    return embedded_rows


def pairwise_similarity_rows(
    records: Sequence[Mapping[str, Any]],
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    latest_only: bool = True,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """Build pairwise similarity rows for review records."""

    embedded_rows = embed_records(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        latest_only=latest_only,
    )
    similarity_rows: list[dict[str, Any]] = []

    for left, right in combinations(embedded_rows, 2):
        if same_case_only and left["case_id"] != right["case_id"]:
            continue

        similarity = cosine_similarity(
            left["vector"],
            right["vector"],
        )
        if min_similarity is not None and similarity < min_similarity:
            continue

        similarity_rows.append({
            "left_record_id": left["record_id"],
            "right_record_id": right["record_id"],
            "left_combo_id": left["combo_id"],
            "right_combo_id": right["combo_id"],
            "case_id": left["case_id"],
            "same_case": left["case_id"] == right["case_id"],
            "similarity": similarity,
            "field_path": field_path,
            "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
            "embedding_dimensions": dimensions,
        })

    return sorted(
        similarity_rows,
        key=lambda row: (
            -(row.get("similarity") or 0.0),
            row.get("case_id") or "",
            row.get("left_combo_id") or "",
            row.get("right_combo_id") or "",
        ),
    )


def similarity_matrix(
    records: Sequence[Mapping[str, Any]],
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    latest_only: bool = True,
) -> dict[str, dict[str, float]]:
    """Return a nested similarity matrix keyed by record id."""

    embedded_rows = embed_records(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        latest_only=latest_only,
    )
    matrix: dict[str, dict[str, float]] = {
        row["record_id"]: {row["record_id"]: 1.0}
        for row in embedded_rows
    }

    for left, right in combinations(embedded_rows, 2):
        if same_case_only and left["case_id"] != right["case_id"]:
            continue

        similarity = cosine_similarity(
            left["vector"],
            right["vector"],
        )
        matrix.setdefault(left["record_id"], {})[
            right["record_id"]
        ] = similarity
        matrix.setdefault(right["record_id"], {})[
            left["record_id"]
        ] = similarity

    return matrix


def near_duplicate_clusters(
    records: Sequence[Mapping[str, Any]],
    threshold: float = 0.92,
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Group records whose similarity crosses a near-duplicate threshold."""

    embedded_rows = embed_records(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        latest_only=latest_only,
    )
    by_record_id = {
        row["record_id"]: row
        for row in embedded_rows
    }
    parent = {
        row["record_id"]: row["record_id"]
        for row in embedded_rows
    }

    def find(record_id: str) -> str:
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_rows = pairwise_similarity_rows(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        same_case_only=same_case_only,
        latest_only=latest_only,
        min_similarity=threshold,
    )
    for row in pair_rows:
        union(
            str(row["left_record_id"]),
            str(row["right_record_id"]),
        )

    clusters: dict[str, list[dict[str, Any]]] = {}
    for record_id, row in by_record_id.items():
        root = find(record_id)
        clusters.setdefault(root, []).append(row)

    cluster_rows: list[dict[str, Any]] = []
    cluster_index = 1
    for members in clusters.values():
        if len(members) < 2:
            continue

        member_ids = [row["record_id"] for row in members]
        member_pairs = [
            row for row in pair_rows
            if row["left_record_id"] in member_ids
            and row["right_record_id"] in member_ids
        ]
        similarities = [
            float(row["similarity"])
            for row in member_pairs
        ]
        cluster_rows.append({
            "cluster_id": f"cluster_{cluster_index:03d}",
            "case_id": members[0].get("case_id"),
            "member_record_ids": member_ids,
            "member_combo_ids": [
                row["combo_id"] for row in members
            ],
            "member_count": len(members),
            "min_similarity": min(similarities)
            if similarities else None,
            "max_similarity": max(similarities)
            if similarities else None,
            "average_similarity": round(
                sum(similarities) / len(similarities),
                4,
            ) if similarities else None,
            "field_path": field_path,
            "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        })
        cluster_index += 1

    return sorted(
        cluster_rows,
        key=lambda row: (
            row.get("case_id") or "",
            -(row.get("average_similarity") or 0.0),
            row.get("cluster_id") or "",
        ),
    )


def outlier_rows(
    records: Sequence[Mapping[str, Any]],
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Return records sorted by low peer similarity."""

    embedded_rows = embed_records(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        latest_only=latest_only,
    )
    similarity_index: dict[str, list[float]] = {
        row["record_id"]: []
        for row in embedded_rows
    }

    for pair in pairwise_similarity_rows(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        same_case_only=same_case_only,
        latest_only=latest_only,
    ):
        similarity_index[str(pair["left_record_id"])].append(
            float(pair["similarity"])
        )
        similarity_index[str(pair["right_record_id"])].append(
            float(pair["similarity"])
        )

    rows: list[dict[str, Any]] = []
    for embedded in embedded_rows:
        similarities = similarity_index[embedded["record_id"]]
        average_similarity = (
            round(sum(similarities) / len(similarities), 4)
            if similarities else None
        )
        nearest_similarity = (
            round(max(similarities), 4)
            if similarities else None
        )
        rows.append({
            "record_id": embedded["record_id"],
            "combo_id": embedded["combo_id"],
            "case_id": embedded["case_id"],
            "average_similarity": average_similarity,
            "nearest_neighbor_similarity": nearest_similarity,
            "outlier_score": round(
                1.0 - (average_similarity or 0.0),
                4,
            ) if average_similarity is not None else None,
            "field_path": field_path,
            "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        })

    return sorted(
        rows,
        key=lambda row: (
            -(row.get("outlier_score") or -1.0),
            row.get("case_id") or "",
            row.get("combo_id") or "",
        ),
    )


def similarity_bundle(
    records: Sequence[Mapping[str, Any]],
    field_path: str = "comparison_text",
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    same_case_only: bool = True,
    latest_only: bool = True,
    duplicate_threshold: float = 0.92,
) -> dict[str, Any]:
    """Build one compact similarity bundle for review tooling."""

    pairs = pairwise_similarity_rows(
        records=records,
        field_path=field_path,
        dimensions=dimensions,
        same_case_only=same_case_only,
        latest_only=latest_only,
    )
    return {
        "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        "embedding_dimensions": dimensions,
        "field_path": field_path,
        "record_count": len(
            embed_records(
                records=records,
                field_path=field_path,
                dimensions=dimensions,
                latest_only=latest_only,
            )
        ),
        "pair_count": len(pairs),
        "pairwise_rows": pairs,
        "near_duplicate_clusters": near_duplicate_clusters(
            records=records,
            threshold=duplicate_threshold,
            field_path=field_path,
            dimensions=dimensions,
            same_case_only=same_case_only,
            latest_only=latest_only,
        ),
        "outlier_rows": outlier_rows(
            records=records,
            field_path=field_path,
            dimensions=dimensions,
            same_case_only=same_case_only,
            latest_only=latest_only,
        ),
    }
