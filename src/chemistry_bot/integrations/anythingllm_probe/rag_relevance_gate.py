"""Embedding-based relevance gate for AnythingLLM RAG context.

This module is intentionally pure and testable.  It does not know about
FastAPI, LangGraph, Ollama, or the AnythingLLM tool-call loop.  Callers pass an
already parsed :class:`AdaptedRAGContext` plus an embedding function.  The gate
scores every retrieved chunk against the student's current question, keeps only
high-enough top-k evidence, and returns a filtered context that can safely be
used in the tutor prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Callable, Sequence

from .context_adapter import AdaptedRAGContext, DocumentContextChunk

EmbeddingBatchFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]




@dataclass(frozen=True, slots=True)
class QueryAnchor:
    """A chemically meaningful anchor extracted from the student's question."""

    canonical: str
    kind: str
    family: str
    matched_texts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChunkAnchorMatch:
    """Anchor coverage diagnostics for one candidate chunk."""

    anchor_required: bool
    target_anchor_matched: bool
    supporting_anchor_matched: bool
    generic_topic_matched: bool
    generic_only: bool
    matched_anchors: tuple[str, ...]
    missing_required_anchors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _AnchorFamily:
    canonical: str
    target_terms: tuple[str, ...]
    supporting_terms: tuple[str, ...]
    generic_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GenericTopic:
    canonical: str
    terms: tuple[str, ...]


_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)


_ANCHOR_FAMILIES: tuple[_AnchorFamily, ...] = (
    _AnchorFamily(
        canonical="thiosulphate",
        target_terms=(
            "thiosulphate",
            "thiosulfate",
            "tiosulphate",
            "tiosulfate",
            "sodium thiosulphate",
            "sodium thiosulfate",
        ),
        supporting_terms=("iodine", "iodide", "iodometric", "iodometry"),
        generic_terms=("titration", "titrations", "titrimetric", "titrate", "titrant"),
    ),
    _AnchorFamily(
        canonical="permanganate",
        target_terms=(
            "permanganate",
            "potassium permanganate",
            "kmno4",
            "mno4",
            "mno4-",
            "mno4 −",
        ),
        supporting_terms=(
            "self indicator",
            "self-indicator",
            "endpoint",
            "end point",
            "pink",
            "purple",
            "oxidizing titrant",
            "oxidising titrant",
            "oxidizing agent",
            "oxidising agent",
        ),
        generic_terms=("titration", "titrations", "titrimetric", "titrate", "titrant"),
    ),
    _AnchorFamily(
        canonical="atom_ion",
        target_terms=("atom", "atoms", "ion", "ions"),
        supporting_terms=(
            "electron",
            "electrons",
            "proton",
            "protons",
            "neutron",
            "neutrons",
            "charge",
            "cation",
            "cations",
            "anion",
            "anions",
            "matter",
            "element",
            "elements",
            "nucleus",
            "neutral",
            "gain",
            "gains",
            "gained",
            "loss",
            "loses",
            "lost",
        ),
        generic_terms=("particle", "particles"),
    ),
    _AnchorFamily(
        canonical="primary_standard_naoh",
        target_terms=("naoh", "sodium hydroxide", "primary standard", "primary standards"),
        supporting_terms=(
            "hygroscopic",
            "carbon dioxide",
            "co2",
            "absorbs co2",
            "absorbs carbon dioxide",
            "standard solution",
            "standardization",
            "standardisation",
        ),
        generic_terms=("titration", "titrations", "standard", "standards"),
    ),
)

_GENERIC_TOPICS: tuple[_GenericTopic, ...] = (
    _GenericTopic(
        canonical="titration",
        terms=("titration", "titrations", "titrimetric", "titrate", "titrant"),
    ),
    _GenericTopic(canonical="particle", terms=("particle", "particles")),
)


def normalize_chem_term(text: str) -> str:
    """Normalize a chemistry term for simple anchor matching.

    The function is deliberately lightweight: it handles case, common unicode
    subscript/superscript digits, hyphen variants, and punctuation spacing.  It
    does not try to parse chemistry.  That keeps the anchor layer transparent and
    easy to extend with small curated term lists.
    """
    value = text.translate(_SUBSCRIPT_TRANSLATION).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_normalized_term(normalized_text: str, term: str) -> bool:
    normalized_term = normalize_chem_term(term)
    if not normalized_text or not normalized_term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text) is not None


def _matched_terms(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    normalized_text = normalize_chem_term(text)
    return tuple(
        dict.fromkeys(term for term in terms if _contains_normalized_term(normalized_text, term))
    )


def extract_query_anchors(query: str) -> tuple[QueryAnchor, ...]:
    """Extract a small curated set of chemistry anchors from a question."""
    anchors: list[QueryAnchor] = []
    for family in _ANCHOR_FAMILIES:
        target_matches = _matched_terms(query, family.target_terms)
        if target_matches:
            anchors.append(
                QueryAnchor(
                    canonical=family.canonical,
                    kind="target_anchor",
                    family=family.canonical,
                    matched_texts=target_matches,
                )
            )
        supporting_matches = _matched_terms(query, family.supporting_terms)
        if supporting_matches:
            anchors.append(
                QueryAnchor(
                    canonical=family.canonical,
                    kind="supporting_anchor",
                    family=family.canonical,
                    matched_texts=supporting_matches,
                )
            )

    for topic in _GENERIC_TOPICS:
        topic_matches = _matched_terms(query, topic.terms)
        if topic_matches:
            anchors.append(
                QueryAnchor(
                    canonical=topic.canonical,
                    kind="generic_topic_anchor",
                    family=topic.canonical,
                    matched_texts=topic_matches,
                )
            )
    return tuple(anchors)


def _families_by_name() -> dict[str, _AnchorFamily]:
    return {family.canonical: family for family in _ANCHOR_FAMILIES}


def match_chunk_anchors(text: str, query_anchors: Sequence[QueryAnchor]) -> ChunkAnchorMatch:
    """Compare a candidate chunk with anchors extracted from the question."""
    family_lookup = _families_by_name()
    required_families = tuple(
        dict.fromkeys(
            anchor.family
            for anchor in query_anchors
            if anchor.kind in {"target_anchor", "supporting_anchor"}
            and anchor.family in family_lookup
        )
    )
    anchor_required = bool(required_families)

    matched: list[str] = []
    missing: list[str] = []
    target_matched = False
    supporting_matched = False
    generic_matched = False

    if required_families:
        for family_name in required_families:
            family = family_lookup[family_name]
            target_terms = _matched_terms(text, family.target_terms)
            supporting_terms = _matched_terms(text, family.supporting_terms)
            generic_terms = _matched_terms(text, family.generic_terms)
            if target_terms:
                target_matched = True
                matched.append(f"target:{family.canonical}")
            if supporting_terms:
                supporting_matched = True
                matched.append(f"supporting:{family.canonical}")
            if generic_terms:
                generic_matched = True
                matched.append(f"generic:{family.canonical}")
            if not target_terms and not supporting_terms:
                missing.append(family.canonical)
    else:
        topic_names = tuple(
            dict.fromkeys(anchor.family for anchor in query_anchors if anchor.kind == "generic_topic_anchor")
        )
        topic_lookup = {topic.canonical: topic for topic in _GENERIC_TOPICS}
        for topic_name in topic_names:
            topic = topic_lookup.get(topic_name)
            if topic is None:
                continue
            if _matched_terms(text, topic.terms):
                generic_matched = True
                matched.append(f"generic:{topic.canonical}")

    matched_tuple = tuple(dict.fromkeys(matched))
    return ChunkAnchorMatch(
        anchor_required=anchor_required,
        target_anchor_matched=target_matched,
        supporting_anchor_matched=supporting_matched,
        generic_topic_matched=generic_matched,
        generic_only=generic_matched and not target_matched and not supporting_matched,
        matched_anchors=matched_tuple,
        missing_required_anchors=tuple(dict.fromkeys(missing)),
    )


@dataclass(frozen=True, slots=True)
class RAGRelevanceGateConfig:
    """Configuration for the document evidence gate."""

    enabled: bool = True
    medium_threshold: float = 0.45
    high_threshold: float = 0.58
    top_k: int = 2
    embedding_model_name: str | None = None


@dataclass(frozen=True, slots=True)
class RAGChunkScore:
    """Diagnostic score for one candidate document chunk."""

    index: int
    original_rank: int
    source_document: str | None
    embedding_score: float | None
    lexical_score: float
    decision: str
    reason: str
    preview: str
    anchor_required: bool = False
    target_anchor_matched: bool = False
    supporting_anchor_matched: bool = False
    generic_topic_matched: bool = False
    generic_only: bool = False
    matched_anchors: tuple[str, ...] = ()
    missing_required_anchors: tuple[str, ...] = ()
    anchor_score: float = 0.0
    adjusted_score: float | None = None
    evidence_category: str = "not_evaluated"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RAGRelevanceGateResult:
    """Result of applying the relevance gate to an adapted document context."""

    enabled: bool
    decision: str
    final_relevance: str
    relevance_score: float
    explicit_user_document_request: bool
    context_for_prompt: AdaptedRAGContext | None
    scores: tuple[RAGChunkScore, ...]
    original_chunk_count: int
    passed_chunk_count: int
    rejected_chunk_count: int
    medium_threshold: float
    high_threshold: float
    top_k: int
    embedding_model_name: str | None = None
    embedding_error: str | None = None
    used_fallback_scores: bool = False
    query_anchors: tuple[QueryAnchor, ...] = ()

    @property
    def passed_scores(self) -> tuple[RAGChunkScore, ...]:
        return tuple(score for score in self.scores if score.decision == "passed")

    @property
    def rejected_scores(self) -> tuple[RAGChunkScore, ...]:
        return tuple(score for score in self.scores if score.decision == "rejected")

    def as_dict(self, *, include_context_text: bool = False) -> dict[str, Any]:
        passed_scores = [score.as_dict() for score in self.passed_scores]
        rejected_scores = [score.as_dict() for score in self.rejected_scores]
        evidence_category_counts: dict[str, int] = {}
        for score in self.scores:
            evidence_category_counts[score.evidence_category] = (
                evidence_category_counts.get(score.evidence_category, 0) + 1
            )
        evidence_character_count = (
            self.context_for_prompt.total_character_count
            if self.context_for_prompt is not None
            else 0
        )
        return {
            "enabled": self.enabled,
            "decision": self.decision,
            "final_relevance": self.final_relevance,
            "relevance_score": self.relevance_score,
            "explicit_user_document_request": self.explicit_user_document_request,
            "original_chunk_count": self.original_chunk_count,
            "passed_chunk_count": self.passed_chunk_count,
            "rejected_chunk_count": self.rejected_chunk_count,
            "external_context_used_as_evidence": self.context_for_prompt is not None,
            "evidence_character_count": evidence_character_count,
            "medium_threshold": self.medium_threshold,
            "high_threshold": self.high_threshold,
            "top_k": self.top_k,
            "embedding_model_name": self.embedding_model_name,
            "embedding_error": self.embedding_error,
            "used_fallback_scores": self.used_fallback_scores,
            "query_anchors": [anchor.as_dict() for anchor in self.query_anchors],
            "anchor_summary": {
                "anchor_required": any(score.anchor_required for score in self.scores),
                "target_anchor_matched_count": sum(
                    1 for score in self.scores if score.target_anchor_matched
                ),
                "supporting_anchor_matched_count": sum(
                    1 for score in self.scores if score.supporting_anchor_matched
                ),
                "generic_only_count": sum(1 for score in self.scores if score.generic_only),
            },
            "evidence_category_counts": evidence_category_counts,
            "score_summary": {
                "best_score": self.relevance_score,
                "passed_count": self.passed_chunk_count,
                "rejected_count": self.rejected_chunk_count,
                "passed_indices": [score.index for score in self.passed_scores],
                "rejected_indices": [score.index for score in self.rejected_scores],
            },
            "chunk_scores": [score.as_dict() for score in self.scores],
            "passed_chunk_scores": passed_scores,
            "rejected_chunk_scores": rejected_scores,
            "context_for_prompt": (
                self.context_for_prompt.as_dict(include_text=include_context_text)
                if self.context_for_prompt is not None
                else None
            ),
        }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two vectors, or 0.0 for invalid vectors."""
    if len(left) != len(right) or not left or not right:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        try:
            fa = float(a)
            fb = float(b)
        except (TypeError, ValueError):
            return 0.0
        dot += fa * fb
        left_norm += fa * fa
        right_norm += fb * fb
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _preview(text: str, *, max_chars: int = 180) -> str:
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _label_for_score(score: float, config: RAGRelevanceGateConfig) -> str:
    if score >= config.high_threshold:
        return "high"
    if score >= config.medium_threshold:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _evidence_category(match: ChunkAnchorMatch) -> str:
    """Classify one chunk by anchor coverage before model evidence use.

    The category is intentionally coarse.  It lets the gate distinguish a truly
    on-target fragment from a nearby classroom topic, for example a generic
    titration paragraph retrieved for a thiosulphate question.
    """
    if match.target_anchor_matched:
        # Broad definition anchors such as "ion" are too weak on their own:
        # an "ion exchange" paragraph is nearby chemistry, not a direct
        # answer to "what is an atom or ion?".  Require at least one
        # definition-supporting term such as charge, electron, matter, or
        # nucleus for atom/ion chunks to become prompt evidence.
        if "target:atom_ion" in match.matched_anchors and not match.supporting_anchor_matched:
            return "topic_adjacent" if match.generic_topic_matched else "generic_only"
        return "direct_evidence"
    if match.supporting_anchor_matched:
        return "supporting_evidence"
    if match.anchor_required and match.generic_topic_matched:
        return "topic_adjacent"
    if match.generic_topic_matched:
        return "generic_only"
    return "irrelevant"


def _anchor_score(match: ChunkAnchorMatch) -> float:
    if match.target_anchor_matched and match.supporting_anchor_matched:
        return 1.0
    if match.target_anchor_matched:
        return 0.85
    if match.supporting_anchor_matched:
        return 0.72
    if match.generic_topic_matched:
        return 0.20
    return 0.0


def _adjusted_relevance_score(
    *,
    base_score: float,
    match: ChunkAnchorMatch,
    config: RAGRelevanceGateConfig,
) -> float:
    """Blend embedding/lexical score with anchor coverage.

    Strong target-required questions need target or supporting anchors.  A chunk
    that only says "titration" can stay diagnostically useful, but it must not
    become medium book evidence just because the embedding cloud is similar.
    """
    base = max(0.0, min(1.0, float(base_score)))
    if match.anchor_required:
        if match.target_anchor_matched and match.supporting_anchor_matched:
            return round(min(1.0, base + 0.22), 4)
        if match.target_anchor_matched:
            return round(min(1.0, base + 0.12), 4)
        if match.supporting_anchor_matched:
            return round(min(1.0, base + 0.08), 4)
        # Generic topic overlap is allowed in logs, but not as prompt evidence.
        return round(min(base, max(0.0, config.medium_threshold - 0.0001)), 4)

    if match.generic_topic_matched:
        return round(min(1.0, base + 0.02), 4)
    return round(base, 4)


def _is_prompt_evidence(score_row: RAGChunkScore, *, config: RAGRelevanceGateConfig) -> bool:
    """Return whether a scored chunk may be used as book evidence."""
    if (score_row.adjusted_score or 0.0) < config.medium_threshold:
        return False
    if score_row.anchor_required and score_row.evidence_category in {
        "topic_adjacent",
        "generic_only",
        "irrelevant",
    }:
        return False
    return True


def _is_exercise_like_preview(preview: str) -> bool:
    normalized = normalize_chem_term(preview)
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            "exercise",
            "problem",
            "calculate",
            "calculation",
            "practice",
        )
    )


def _evidence_sort_key(score_row: RAGChunkScore) -> tuple[float, float, float, float, int]:
    """Sort key for choosing the strongest evidence inside one category."""
    model_or_fallback_score = (
        score_row.embedding_score
        if score_row.embedding_score is not None
        else score_row.lexical_score
    )
    adjusted = score_row.adjusted_score or 0.0
    if _is_exercise_like_preview(score_row.preview):
        adjusted = max(0.0, adjusted - 0.18)
    return (
        adjusted,
        score_row.anchor_score,
        model_or_fallback_score or 0.0,
        score_row.lexical_score,
        -score_row.original_rank,
    )


def _best_unselected(
    candidates: Sequence[RAGChunkScore],
    *,
    selected: set[int],
) -> RAGChunkScore | None:
    available = [row for row in candidates if row.index not in selected]
    if not available:
        return None
    return max(available, key=_evidence_sort_key)


def _select_prompt_evidence_rows(
    rows: Sequence[RAGChunkScore],
    *,
    config: RAGRelevanceGateConfig,
) -> list[int]:
    """Choose evidence chunks using anchors before falling back to raw top-k.

    A plain top-k by embedding can over-select generic classroom passages.  The
    mixer gives the first slot to the best anchor-rich explanatory fragment when
    one exists, then adds the best direct target fragment, and only then fills
    the remaining room by adjusted score.  For queries without required anchors
    this deliberately behaves like the original adjusted top-k gate.
    """
    top_k = max(1, int(config.top_k))
    eligible = [row for row in rows if _is_prompt_evidence(row, config=config)]
    if not eligible:
        return []

    has_required_anchor_query = any(row.anchor_required for row in eligible)
    if not has_required_anchor_query:
        return [row.index for row in sorted(eligible, key=_evidence_sort_key, reverse=True)[:top_k]]

    selected: list[int] = []
    selected_set: set[int] = set()

    def add(row: RAGChunkScore | None) -> None:
        if row is None or row.index in selected_set or len(selected) >= top_k:
            return
        selected.append(row.index)
        selected_set.add(row.index)

    # Slot 1: the best explanatory/supporting fragment.  This catches chunks
    # such as self-indicator / pink endpoint for permanganate and iodine /
    # iodometric context for thiosulphate, even when a dry exercise has a higher
    # lexical or embedding score.
    add(_best_unselected([row for row in eligible if row.supporting_anchor_matched], selected=selected_set))

    # Slot 2: the best direct target fragment, usually a formula, equation, or
    # named-reagent passage.
    add(_best_unselected([row for row in eligible if row.target_anchor_matched], selected=selected_set))

    # Fill remaining room by adjusted score while respecting the evidence gate.
    for row in sorted(eligible, key=_evidence_sort_key, reverse=True):
        add(row)
        if len(selected) >= top_k:
            break

    return selected


def _context_from_passed_chunks(
    *,
    original: AdaptedRAGContext,
    passed: Sequence[DocumentContextChunk],
    relevance: str,
    relevance_score: float,
    warning: str | None = None,
) -> AdaptedRAGContext | None:
    if not passed:
        return None
    reranked: list[DocumentContextChunk] = []
    total_chars = 0
    for rank, chunk in enumerate(passed, start=1):
        reranked.append(
            DocumentContextChunk(
                text=chunk.text,
                source_document=chunk.source_document,
                published=chunk.published,
                result_index=chunk.result_index,
                rank=rank,
                relevance_score=chunk.relevance_score,
                matched_query_terms=chunk.matched_query_terms,
            )
        )
        total_chars += len(chunk.text)
    sources = tuple(
        dict.fromkeys(chunk.source_document for chunk in reranked if chunk.source_document)
    )
    warnings = list(original.warnings)
    if warning:
        warnings.append(warning)
    return AdaptedRAGContext(
        query=original.query,
        source_type=original.source_type,
        relevance=relevance,
        relevance_score=relevance_score,
        chunks=tuple(reranked),
        source_documents=sources,
        warnings=tuple(dict.fromkeys(warnings)),
        raw_result_count=original.raw_result_count,
        total_character_count=total_chars,
    )


def _embedding_scores(
    *,
    query: str,
    chunks: Sequence[DocumentContextChunk],
    embed_texts: EmbeddingBatchFn,
) -> list[float]:
    texts = [query] + [chunk.text for chunk in chunks]
    vectors = list(embed_texts(texts))
    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding function returned {len(vectors)} vectors for {len(texts)} texts."
        )
    query_vector = vectors[0]
    return [round(cosine_similarity(query_vector, vector), 4) for vector in vectors[1:]]


def _fallback_scores(chunks: Sequence[DocumentContextChunk]) -> list[float]:
    return [round(max(0.0, min(1.0, chunk.relevance_score)), 4) for chunk in chunks]


def apply_rag_relevance_gate(
    *,
    query: str,
    context: AdaptedRAGContext | None,
    explicit_user_document_request: bool,
    config: RAGRelevanceGateConfig | None = None,
    embed_texts: EmbeddingBatchFn | None = None,
) -> RAGRelevanceGateResult:
    """Filter retrieved document chunks before they enter the tutor prompt.

    When embeddings are unavailable or fail, the function falls back to the
    lexical relevance score already produced by ``context_adapter``.  That keeps
    the request safe and diagnosable without making the whole bridge brittle.
    """
    config = config or RAGRelevanceGateConfig()
    chunks = tuple(context.chunks) if context is not None else ()
    top_k = max(1, int(config.top_k))
    query_anchors = extract_query_anchors(query)

    if not config.enabled:
        passed = chunks[:top_k]
        best_score = max((chunk.relevance_score for chunk in passed), default=0.0)
        final_relevance = context.relevance if context is not None else "none"
        context_for_prompt = _context_from_passed_chunks(
            original=context,
            passed=passed,
            relevance=final_relevance,
            relevance_score=best_score,
            warning="RAG relevance gate disabled; passing adapter-selected context.",
        ) if context is not None else None
        scores = tuple(
            RAGChunkScore(
                index=index,
                original_rank=chunk.rank,
                source_document=chunk.source_document,
                embedding_score=None,
                lexical_score=chunk.relevance_score,
                decision="passed" if index < len(passed) else "rejected",
                reason="gate_disabled" if index < len(passed) else "outside_top_k",
                preview=_preview(chunk.text),
            )
            for index, chunk in enumerate(chunks)
        )
        return RAGRelevanceGateResult(
            enabled=False,
            decision="pass_context" if context_for_prompt else "no_context",
            final_relevance=final_relevance,
            relevance_score=best_score,
            explicit_user_document_request=explicit_user_document_request,
            context_for_prompt=context_for_prompt,
            scores=scores,
            original_chunk_count=len(chunks),
            passed_chunk_count=len(passed),
            rejected_chunk_count=max(0, len(chunks) - len(passed)),
            medium_threshold=config.medium_threshold,
            high_threshold=config.high_threshold,
            top_k=top_k,
            embedding_model_name=config.embedding_model_name,
            query_anchors=query_anchors,
        )

    if context is None or not chunks:
        return RAGRelevanceGateResult(
            enabled=True,
            decision="no_context",
            final_relevance="none",
            relevance_score=0.0,
            explicit_user_document_request=explicit_user_document_request,
            context_for_prompt=None,
            scores=(),
            original_chunk_count=0,
            passed_chunk_count=0,
            rejected_chunk_count=0,
            medium_threshold=config.medium_threshold,
            high_threshold=config.high_threshold,
            top_k=top_k,
            embedding_model_name=config.embedding_model_name,
            query_anchors=query_anchors,
        )

    embedding_error: str | None = None
    used_fallback_scores = False
    if embed_texts is None:
        scores = _fallback_scores(chunks)
        used_fallback_scores = True
        embedding_error = "embedding_function_not_provided"
    else:
        try:
            scores = _embedding_scores(query=query, chunks=chunks, embed_texts=embed_texts)
        except Exception as exc:  # pragma: no cover - exact exception varies by client
            scores = _fallback_scores(chunks)
            used_fallback_scores = True
            embedding_error = f"{type(exc).__name__}: {exc}"

    scored_candidates: list[tuple[int, DocumentContextChunk, float, ChunkAnchorMatch, str, float, float]] = []
    for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
        anchor_match = match_chunk_anchors(chunk.text, query_anchors)
        category = _evidence_category(anchor_match)
        anchor_score = _anchor_score(anchor_match)
        adjusted_score = _adjusted_relevance_score(
            base_score=score,
            match=anchor_match,
            config=config,
        )
        scored_candidates.append(
            (index, chunk, score, anchor_match, category, anchor_score, adjusted_score)
        )

    scored_candidates.sort(
        key=lambda item: (item[6], item[5], item[2], item[1].relevance_score, len(item[1].text)),
        reverse=True,
    )

    best_score = scored_candidates[0][6] if scored_candidates else 0.0
    final_relevance = _label_for_score(best_score, config)

    passed_indices: set[int] = set()
    if final_relevance in {"medium", "high"}:
        provisional_rows = [
            RAGChunkScore(
                index=index,
                original_rank=chunk.rank,
                source_document=chunk.source_document,
                embedding_score=score if not used_fallback_scores else None,
                lexical_score=chunk.relevance_score,
                decision="candidate",
                reason="candidate",
                preview=_preview(chunk.text),
                anchor_required=anchor_match.anchor_required,
                target_anchor_matched=anchor_match.target_anchor_matched,
                supporting_anchor_matched=anchor_match.supporting_anchor_matched,
                generic_topic_matched=anchor_match.generic_topic_matched,
                generic_only=anchor_match.generic_only,
                matched_anchors=anchor_match.matched_anchors,
                missing_required_anchors=anchor_match.missing_required_anchors,
                anchor_score=anchor_score,
                adjusted_score=adjusted_score,
                evidence_category=category,
            )
            for index, chunk, score, anchor_match, category, anchor_score, adjusted_score in scored_candidates
        ]
        passed_index_list = _select_prompt_evidence_rows(
            provisional_rows,
            config=config,
        )
        passed_indices = set(passed_index_list)
        decision = "pass_context" if passed_indices else (
            "explicit_document_not_found" if explicit_user_document_request else "suppress_low_relevance"
        )
    elif explicit_user_document_request:
        decision = "explicit_document_not_found"
    else:
        decision = "suppress_low_relevance"

    score_rows: list[RAGChunkScore] = []
    passed_chunks: list[DocumentContextChunk] = []
    candidates_by_index = {
        index: (chunk, score, anchor_match, category, anchor_score, adjusted_score)
        for index, chunk, score, anchor_match, category, anchor_score, adjusted_score in scored_candidates
    }
    for index in range(len(chunks)):
        chunk, score, anchor_match, category, anchor_score, adjusted_score = candidates_by_index[index]
        if index in passed_indices:
            passed_chunks.append(
                DocumentContextChunk(
                    text=chunk.text,
                    source_document=chunk.source_document,
                    published=chunk.published,
                    result_index=chunk.result_index,
                    rank=chunk.rank,
                    relevance_score=score,
                    matched_query_terms=chunk.matched_query_terms,
                )
            )
            row_decision = "passed"
            reason = "anchor_aware_score_within_top_k_and_above_medium_threshold"
        else:
            row_decision = "rejected"
            if final_relevance in {"none", "low"}:
                reason = "below_adjusted_medium_threshold"
            elif anchor_match.anchor_required and category in {
                "topic_adjacent",
                "generic_only",
                "irrelevant",
            }:
                reason = "missing_required_anchor_evidence"
            else:
                reason = "outside_top_k"
        score_rows.append(
            RAGChunkScore(
                index=index,
                original_rank=chunk.rank,
                source_document=chunk.source_document,
                embedding_score=score if not used_fallback_scores else None,
                lexical_score=chunk.relevance_score,
                decision=row_decision,
                reason=reason,
                preview=_preview(chunk.text),
                anchor_required=anchor_match.anchor_required,
                target_anchor_matched=anchor_match.target_anchor_matched,
                supporting_anchor_matched=anchor_match.supporting_anchor_matched,
                generic_topic_matched=anchor_match.generic_topic_matched,
                generic_only=anchor_match.generic_only,
                matched_anchors=anchor_match.matched_anchors,
                missing_required_anchors=anchor_match.missing_required_anchors,
                anchor_score=anchor_score,
                adjusted_score=adjusted_score,
                evidence_category=category,
            )
        )

    context_for_prompt = _context_from_passed_chunks(
        original=context,
        passed=passed_chunks,
        relevance=final_relevance,
        relevance_score=best_score,
        warning="Embedding relevance gate passed only top-scoring excerpts as evidence.",
    )

    return RAGRelevanceGateResult(
        enabled=True,
        decision=decision,
        final_relevance=final_relevance,
        relevance_score=best_score,
        explicit_user_document_request=explicit_user_document_request,
        context_for_prompt=context_for_prompt,
        scores=tuple(score_rows),
        original_chunk_count=len(chunks),
        passed_chunk_count=len(passed_chunks),
        rejected_chunk_count=len(chunks) - len(passed_chunks),
        medium_threshold=config.medium_threshold,
        high_threshold=config.high_threshold,
        top_k=top_k,
        embedding_model_name=config.embedding_model_name,
        embedding_error=embedding_error,
        used_fallback_scores=used_fallback_scores,
        query_anchors=query_anchors,
    )


__all__ = [
    "ChunkAnchorMatch",
    "QueryAnchor",
    "RAGChunkScore",
    "RAGRelevanceGateConfig",
    "RAGRelevanceGateResult",
    "apply_rag_relevance_gate",
    "extract_query_anchors",
    "match_chunk_anchors",
    "normalize_chem_term",
    "cosine_similarity",
]
