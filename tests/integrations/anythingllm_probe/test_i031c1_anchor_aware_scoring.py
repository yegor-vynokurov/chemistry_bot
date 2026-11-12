from __future__ import annotations

import math
import unittest
from typing import Sequence

from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
)
from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    RAGRelevanceGateConfig,
    apply_rag_relevance_gate,
)


def _context(*chunks: DocumentContextChunk, query: str) -> AdaptedRAGContext:
    sources = tuple(dict.fromkeys(chunk.source_document for chunk in chunks if chunk.source_document))
    return AdaptedRAGContext(
        query=query,
        relevance="medium" if chunks else "none",
        relevance_score=max((chunk.relevance_score for chunk in chunks), default=0.0),
        chunks=chunks,
        source_documents=sources,
        raw_result_count=1 if chunks else 0,
        total_character_count=sum(len(chunk.text) for chunk in chunks),
    )


def _chunk(text: str, *, rank: int, lexical: float = 0.0) -> DocumentContextChunk:
    return DocumentContextChunk(
        text=text,
        source_document="book.pdf",
        result_index=0,
        rank=rank,
        relevance_score=lexical,
        matched_query_terms=(),
    )


def _unit_vector(cosine: float) -> tuple[float, float]:
    value = max(-1.0, min(1.0, cosine))
    return (value, math.sqrt(max(0.0, 1.0 - value * value)))


def _fake_embed(cosines_by_text: dict[str, float]):
    def embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
        # Query is always the reference direction; each chunk vector is built so
        # cosine(query, chunk) equals the configured score.
        vectors: list[Sequence[float]] = []
        for text in texts:
            if text not in cosines_by_text:
                vectors.append((1.0, 0.0))
            else:
                vectors.append(_unit_vector(cosines_by_text[text]))
        return vectors

    return embed_texts


class AnchorAwareScoringTests(unittest.TestCase):
    def test_thiosulphate_generic_titration_chunk_cannot_be_medium_evidence(self) -> None:
        query = "according to uploaded book why do we need thiosulphate for titration"
        generic = _chunk(
            "A titration should have a rapid reaction, a clear endpoint, and a suitable titrant.",
            rank=1,
            lexical=0.80,
        )
        ctx = _context(generic, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, generic.text: 0.92}),
        )

        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertEqual(result.final_relevance, "low")
        self.assertIsNone(result.context_for_prompt)
        self.assertEqual(result.passed_chunk_count, 0)
        self.assertEqual(result.scores[0].evidence_category, "topic_adjacent")
        self.assertTrue(result.scores[0].generic_only)
        assert result.scores[0].adjusted_score is not None
        self.assertLess(result.scores[0].adjusted_score, 0.45)

    def test_thiosulphate_iodometric_chunk_passes_as_supporting_evidence(self) -> None:
        query = "according to uploaded book why do we need thiosulphate for titration"
        generic = _chunk(
            "A titration should have a rapid reaction, a clear endpoint, and a suitable titrant.",
            rank=1,
            lexical=0.80,
        )
        iodometric = _chunk(
            "In iodometric analysis iodine is reduced by the titrant at the endpoint.",
            rank=2,
            lexical=0.45,
        )
        ctx = _context(generic, iodometric, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, generic.text: 0.92, iodometric.text: 0.50}),
        )

        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.passed_chunk_count, 1)
        assert result.context_for_prompt is not None
        self.assertIn("iodometric", result.context_for_prompt.chunks[0].text.lower())
        score_by_rank = {score.original_rank: score for score in result.scores}
        self.assertEqual(score_by_rank[1].evidence_category, "topic_adjacent")
        self.assertEqual(score_by_rank[2].evidence_category, "supporting_evidence")
        self.assertEqual(score_by_rank[2].decision, "passed")

    def test_permanganate_self_indicator_chunk_beats_generic_exercise_by_adjusted_score(self) -> None:
        query = "according to the books in the folder why is KMnO4 useful in titration"
        exercise = _chunk(
            "Exercise: calculate the titration volume using KMnO4 in a redox titration.",
            rank=1,
            lexical=0.79,
        )
        endpoint = _chunk(
            "Potassium permanganate acts as a self-indicator because a faint pink endpoint remains.",
            rank=2,
            lexical=0.70,
        )
        ctx = _context(exercise, endpoint, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=1),
            embed_texts=None,
        )

        self.assertTrue(result.used_fallback_scores)
        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.passed_chunk_count, 1)
        assert result.context_for_prompt is not None
        self.assertIn("self-indicator", result.context_for_prompt.chunks[0].text)
        score_by_rank = {score.original_rank: score for score in result.scores}
        self.assertEqual(score_by_rank[1].evidence_category, "direct_evidence")
        self.assertEqual(score_by_rank[2].evidence_category, "direct_evidence")
        assert score_by_rank[2].adjusted_score is not None
        assert score_by_rank[1].adjusted_score is not None
        self.assertGreater(score_by_rank[2].adjusted_score, score_by_rank[1].adjusted_score)
        self.assertEqual(score_by_rank[2].decision, "passed")

    def test_anchor_diagnostics_are_in_result_payload(self) -> None:
        query = "why do we need thiosulfate for titration"
        generic = _chunk("A titration should have a clear endpoint.", rank=1, lexical=0.8)
        ctx = _context(generic, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=False,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, generic.text: 0.9}),
        )
        payload = result.as_dict()

        self.assertIn("query_anchors", payload)
        self.assertIn("anchor_summary", payload)
        self.assertIn("evidence_category_counts", payload)
        self.assertEqual(payload["evidence_category_counts"], {"topic_adjacent": 1})
        self.assertEqual(payload["chunk_scores"][0]["evidence_category"], "topic_adjacent")
        self.assertIn("thiosulphate", payload["chunk_scores"][0]["missing_required_anchors"])


if __name__ == "__main__":
    unittest.main()
