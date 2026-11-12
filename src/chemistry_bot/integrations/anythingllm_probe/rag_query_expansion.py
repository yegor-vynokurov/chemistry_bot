"""Deterministic query expansion for AnythingLLM document retrieval.

The expansion layer never rewrites the student's question.  It only prepares the
technical ``rag-memory`` search string used by AnythingLLM.  The tutor still
answers the original student question, while logs retain both strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .rag_relevance_gate import (
    QueryAnchor,
    RAGRelevanceGateResult,
    extract_query_anchors,
    normalize_chem_term,
)

EXPANSION_REGISTRY_VERSION = "i0.3.1d-anchor-registry-v1"


@dataclass(frozen=True, slots=True)
class AnchorExpansionEntry:
    """Small curated expansion entry for one chemistry anchor family."""

    canonical: str
    aliases: tuple[str, ...]
    supporting_terms: tuple[str, ...] = ()
    definition_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RAGQueryExpansionConfig:
    """Configuration for deterministic AnythingLLM retrieval-query expansion."""

    enabled: bool = True
    retry_enabled: bool = True
    max_added_terms_light: int = 8
    max_added_terms_targeted: int = 14


@dataclass(frozen=True, slots=True)
class RAGQueryExpansionResult:
    """Auditable result of preparing the retrieval query."""

    enabled: bool
    mode: str
    original_query: str
    expanded_query: str
    changed: bool
    reason_codes: tuple[str, ...]
    query_anchors: tuple[QueryAnchor, ...]
    added_terms: tuple[str, ...]
    skipped_terms: tuple[str, ...]
    registry_version: str = EXPANSION_REGISTRY_VERSION
    retry_after_low_relevance: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["query_anchors"] = [anchor.as_dict() for anchor in self.query_anchors]
        return payload


_EXPANSION_REGISTRY: Mapping[str, AnchorExpansionEntry] = {
    "thiosulphate": AnchorExpansionEntry(
        canonical="thiosulphate",
        aliases=(
            "thiosulphate",
            "thiosulfate",
            "tiosulphate",
            "tiosulfate",
            "sodium thiosulfate",
            "sodium thiosulphate",
        ),
        supporting_terms=(
            "iodometric titration",
            "iodometry",
            "iodine",
            "iodide",
            "standardization",
            "standardisation",
            "titrant",
            "analytical chemistry",
        ),
    ),
    "permanganate": AnchorExpansionEntry(
        canonical="permanganate",
        aliases=(
            "permanganate",
            "potassium permanganate",
            "KMnO4",
            "MnO4",
        ),
        supporting_terms=(
            "self-indicator",
            "self indicator",
            "endpoint",
            "end point",
            "pink endpoint",
            "purple",
            "oxidizing titrant",
            "oxidising titrant",
            "redox titration",
            "oxalate",
        ),
    ),
    "atom_ion": AnchorExpansionEntry(
        canonical="atom_ion",
        aliases=("atom", "ion", "atoms", "ions"),
        supporting_terms=("electron", "proton", "neutron", "charge", "cation", "anion"),
        definition_terms=(
            "definition",
            "defined as",
            "is a",
            "smallest unit",
            "charged atom",
            "gained electrons",
            "lost electrons",
        ),
    ),
    "primary_standard_naoh": AnchorExpansionEntry(
        canonical="primary_standard_naoh",
        aliases=("NaOH", "sodium hydroxide", "primary standard", "primary standards"),
        supporting_terms=(
            "hygroscopic",
            "carbon dioxide",
            "CO2",
            "absorbs carbon dioxide",
            "standard solution",
            "standardization",
            "standardisation",
            "titration",
        ),
    ),
}

_GENERIC_TOPIC_TERMS: Mapping[str, tuple[str, ...]] = {
    "titration": ("titration", "titrations", "titrimetric", "titrant"),
    "particle": ("particle", "particles"),
}

_PROTECTED_REASON_CODES = {
    "disabled",
    "no_user_document_retrieval",
    "safety_disabled",
    "off_topic_or_clarification_route",
}


def _unique_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        compact = " ".join(str(value).split())
        if not compact:
            continue
        key = normalize_chem_term(compact)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(compact)
    return tuple(result)


def _normalized_query_terms(query: str) -> set[str]:
    normalized_query = normalize_chem_term(query)
    terms: set[str] = set()
    for token in normalized_query.split():
        terms.add(token)
    # Preserve phrase checks as normalized strings too.
    terms.add(normalized_query)
    return terms


def _query_contains_term(query: str, term: str) -> bool:
    normalized_query = f" {normalize_chem_term(query)} "
    normalized_term = normalize_chem_term(term)
    if not normalized_query.strip() or not normalized_term:
        return False
    return f" {normalized_term} " in normalized_query


def _anchor_families(query_anchors: Sequence[QueryAnchor]) -> tuple[str, ...]:
    families: list[str] = []
    for anchor in query_anchors:
        if anchor.family in _EXPANSION_REGISTRY and anchor.family not in families:
            families.append(anchor.family)
    return tuple(families)


def _has_typo_or_variant(query: str, query_anchors: Sequence[QueryAnchor]) -> bool:
    for anchor in query_anchors:
        if anchor.family not in _EXPANSION_REGISTRY:
            continue
        if anchor.canonical == "thiosulphate" and any(
            normalize_chem_term(text) in {"tiosulphate", "tiosulfate", "thiosulfate"}
            for text in anchor.matched_texts
        ):
            return True
        if anchor.canonical == "permanganate" and any(
            normalize_chem_term(text) in {"kmno4", "mno4"}
            for text in anchor.matched_texts
        ):
            return True
    return False


def _is_short_or_noisy(query: str) -> bool:
    tokens = normalize_chem_term(query).split()
    if len(tokens) <= 7:
        return True
    noisy_markers = ("????", "???", "depend uploaded", "according uploaded")
    return any(marker in query.lower() for marker in noisy_markers)


def _gate_is_retryable(gate_result: RAGRelevanceGateResult | Mapping[str, Any] | None) -> bool:
    if gate_result is None:
        return False
    if isinstance(gate_result, RAGRelevanceGateResult):
        decision = gate_result.decision
        final_relevance = gate_result.final_relevance
        passed_count = gate_result.passed_chunk_count
        category_counts = gate_result.as_dict().get("evidence_category_counts", {})
    else:
        decision = str(gate_result.get("decision", ""))
        final_relevance = str(gate_result.get("final_relevance", ""))
        passed_count = int(gate_result.get("passed_chunk_count") or 0)
        raw_counts = gate_result.get("evidence_category_counts")
        category_counts = raw_counts if isinstance(raw_counts, dict) else {}
    if decision in {"explicit_document_not_found", "suppress_low_relevance"}:
        return True
    if passed_count == 0 and final_relevance in {"low", "none"}:
        return True
    if passed_count == 0 and any(
        category_counts.get(name, 0)
        for name in ("topic_adjacent", "generic_only")
    ):
        return True
    return False


def _terms_for_mode(
    *,
    mode: str,
    query: str,
    query_anchors: Sequence[QueryAnchor],
) -> tuple[str, ...]:
    families = _anchor_families(query_anchors)
    terms: list[str] = []
    for family_name in families:
        entry = _EXPANSION_REGISTRY[family_name]
        terms.extend(entry.aliases)
        if mode == "retry_targeted":
            terms.extend(entry.supporting_terms)
            # Definition cues are only useful for definition-shaped atom/ion
            # questions.  They avoid matching every ionization or ion-exchange
            # paragraph when the user wants a definition.
            if family_name == "atom_ion" and any(
                cue in normalize_chem_term(query).split()
                for cue in ("what", "define", "definition", "meaning")
            ):
                terms.extend(entry.definition_terms)

    if mode == "retry_targeted":
        # Keep method/generic topics only in the targeted retry, where they help
        # narrow the search together with target/supporting anchors.
        generic_families = [
            anchor.family for anchor in query_anchors if anchor.kind == "generic_topic_anchor"
        ]
        for family in generic_families:
            terms.extend(_GENERIC_TOPIC_TERMS.get(family, ()))
    return _unique_preserve_order(terms)


def _compose_expanded_query(
    *,
    original_query: str,
    candidate_terms: Sequence[str],
    max_added_terms: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    added: list[str] = []
    skipped: list[str] = []
    for term in _unique_preserve_order(candidate_terms):
        if _query_contains_term(original_query, term):
            skipped.append(term)
            continue
        if len(added) >= max_added_terms:
            skipped.append(term)
            continue
        added.append(term)
    expanded = original_query.strip()
    if added:
        expanded = f"{expanded} {' '.join(added)}".strip()
    return expanded, tuple(added), tuple(skipped)


def choose_query_expansion(
    *,
    original_query: str,
    should_call_user_documents: bool,
    explicit_document_request: bool,
    route: str | None = None,
    safety_level: str | None = None,
    config: RAGQueryExpansionConfig | None = None,
    first_gate_result: RAGRelevanceGateResult | Mapping[str, Any] | None = None,
    previous_expansion: RAGQueryExpansionResult | Mapping[str, Any] | None = None,
) -> RAGQueryExpansionResult:
    """Choose a deterministic retrieval query for AnythingLLM.

    The returned ``expanded_query`` is a search string only.  The original
    student question must still be used for routing, tutoring, and answer
    generation.
    """
    config = config or RAGQueryExpansionConfig()
    query = original_query.strip()
    query_anchors = extract_query_anchors(query)
    reason_codes: list[str] = []

    if not config.enabled:
        reason_codes.append("disabled")
        return RAGQueryExpansionResult(
            enabled=False,
            mode="none",
            original_query=query,
            expanded_query=query,
            changed=False,
            reason_codes=tuple(reason_codes),
            query_anchors=query_anchors,
            added_terms=(),
            skipped_terms=(),
        )

    if not should_call_user_documents:
        reason_codes.append("no_user_document_retrieval")
    if (safety_level or "safe") != "safe" or route == "safety_redirect":
        reason_codes.append("safety_disabled")
    if route in {"off_topic", "clarification"}:
        reason_codes.append("off_topic_or_clarification_route")

    if any(code in _PROTECTED_REASON_CODES for code in reason_codes):
        return RAGQueryExpansionResult(
            enabled=True,
            mode="none",
            original_query=query,
            expanded_query=query,
            changed=False,
            reason_codes=tuple(reason_codes),
            query_anchors=query_anchors,
            added_terms=(),
            skipped_terms=(),
        )

    families = _anchor_families(query_anchors)
    if not families:
        return RAGQueryExpansionResult(
            enabled=True,
            mode="none",
            original_query=query,
            expanded_query=query,
            changed=False,
            reason_codes=("no_recognized_anchor",),
            query_anchors=query_anchors,
            added_terms=(),
            skipped_terms=(),
        )

    previous_mode = None
    if isinstance(previous_expansion, RAGQueryExpansionResult):
        previous_mode = previous_expansion.mode
    elif isinstance(previous_expansion, Mapping):
        raw_mode = previous_expansion.get("mode")
        previous_mode = str(raw_mode) if raw_mode is not None else None

    retry_requested = first_gate_result is not None
    retry_after_low = bool(
        retry_requested
        and config.retry_enabled
        and previous_mode != "retry_targeted"
        and _gate_is_retryable(first_gate_result)
    )
    if retry_requested and not retry_after_low:
        reasons = ["retry_not_needed"]
        if previous_mode == "retry_targeted":
            reasons.append("already_targeted")
        return RAGQueryExpansionResult(
            enabled=True,
            mode="none",
            original_query=query,
            expanded_query=query,
            changed=False,
            reason_codes=tuple(reasons),
            query_anchors=query_anchors,
            added_terms=(),
            skipped_terms=(),
        )

    if retry_after_low:
        mode = "retry_targeted"
        reason_codes.extend(("first_retrieval_low_or_topic_adjacent", "recognized_anchor"))
        if explicit_document_request:
            reason_codes.append("explicit_document_request")
        max_terms = config.max_added_terms_targeted
    else:
        # I0.3.1D intentionally keeps the first retrieval conservative: add
        # aliases, formulas, and spelling variants, but not broader supporting
        # chemistry terms until the first retrieval proves too weak.
        mode = "light"
        reason_codes.append("recognized_anchor")
        if explicit_document_request:
            reason_codes.append("explicit_document_request")
        if _has_typo_or_variant(query, query_anchors):
            reason_codes.append("typo_or_formula_variant_detected")
        if _is_short_or_noisy(query):
            reason_codes.append("short_or_noisy_query")
        max_terms = config.max_added_terms_light

    terms = _terms_for_mode(mode=mode, query=query, query_anchors=query_anchors)
    expanded_query, added_terms, skipped_terms = _compose_expanded_query(
        original_query=query,
        candidate_terms=terms,
        max_added_terms=max_terms,
    )
    changed = bool(added_terms)
    if not changed:
        reason_codes.append("no_new_terms_added")

    return RAGQueryExpansionResult(
        enabled=True,
        mode=mode if changed else "none",
        original_query=query,
        expanded_query=expanded_query,
        changed=changed,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        query_anchors=query_anchors,
        added_terms=added_terms,
        skipped_terms=skipped_terms,
        retry_after_low_relevance=retry_after_low,
    )


__all__ = [
    "EXPANSION_REGISTRY_VERSION",
    "AnchorExpansionEntry",
    "RAGQueryExpansionConfig",
    "RAGQueryExpansionResult",
    "choose_query_expansion",
]
