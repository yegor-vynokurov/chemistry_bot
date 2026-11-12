from __future__ import annotations

import unittest
from typing import Sequence

from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
)
from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    RAGRelevanceGateConfig,
    apply_rag_relevance_gate,
    cosine_similarity,
)


def _context(*chunks: DocumentContextChunk, query: str = "student question") -> AdaptedRAGContext:
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


def _chunk(text: str, *, rank: int, lexical: float = 0.0, source: str = "book.pdf") -> DocumentContextChunk:
    return DocumentContextChunk(
        text=text,
        source_document=source,
        result_index=0,
        rank=rank,
        relevance_score=lexical,
        matched_query_terms=(),
    )


def _fake_embed(vectors_by_text: dict[str, Sequence[float]]):
    def embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
        return [vectors_by_text[text] for text in texts]

    return embed_texts


class RAGRelevanceGateTests(unittest.TestCase):
    def test_cosine_similarity_handles_zero_and_mismatch(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)
        self.assertEqual(cosine_similarity([1, 0], [1]), 0.0)

    def test_low_embedding_scores_suppress_chunks(self) -> None:
        query = "why do we need thiosulphate for titration"
        ctx = _context(
            _chunk("A zwitterion contains both positive and negative charges.", rank=1, lexical=0.10),
            _chunk("Amino acids can form zwitterions in water.", rank=2, lexical=0.05),
            query=query,
        )
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=False,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed(
                {
                    query: [1.0, 0.0, 0.0],
                    ctx.chunks[0].text: [0.10, 0.90, 0.0],
                    ctx.chunks[1].text: [0.20, 0.80, 0.0],
                }
            ),
        )
        self.assertEqual(result.decision, "suppress_low_relevance")
        self.assertEqual(result.final_relevance, "low")
        self.assertIsNone(result.context_for_prompt)
        self.assertEqual(result.passed_chunk_count, 0)
        self.assertEqual(result.rejected_chunk_count, 2)
        self.assertTrue(all(row.decision == "rejected" for row in result.scores))

    def test_explicit_document_request_low_scores_returns_document_not_found_decision(self) -> None:
        query = "according to uploaded book why use thiosulphate"
        ctx = _context(
            _chunk("This passage is about amino acid zwitterions.", rank=1, lexical=0.08),
            query=query,
        )
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            embed_texts=_fake_embed({query: [1.0, 0.0], ctx.chunks[0].text: [0.05, 0.95]}),
        )
        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertEqual(result.final_relevance, "low")
        self.assertIsNone(result.context_for_prompt)
        self.assertTrue(result.explicit_user_document_request)

    def test_medium_score_passes_top_chunk(self) -> None:
        query = "weak acid strong base titration"
        ctx = _context(
            _chunk("Weak acid titration with strong base has a buffer region.", rank=1, lexical=0.8),
            _chunk("This separate paragraph is about proteins.", rank=2, lexical=0.1),
            query=query,
        )
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.90, top_k=2),
            embed_texts=_fake_embed(
                {
                    query: [1.0, 0.0],
                    ctx.chunks[0].text: [0.60, 0.80],  # cosine 0.60 -> medium with high threshold 0.90
                    ctx.chunks[1].text: [0.10, 0.99],
                }
            ),
        )
        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.final_relevance, "medium")
        self.assertIsNotNone(result.context_for_prompt)
        assert result.context_for_prompt is not None
        self.assertEqual(len(result.context_for_prompt.chunks), 1)
        self.assertIn("Weak acid", result.context_for_prompt.chunks[0].text)
        self.assertEqual(result.passed_chunk_count, 1)

    def test_high_score_respects_top_k(self) -> None:
        query = "sodium carbonate titration"
        ctx = _context(
            _chunk("Best sodium carbonate titration passage.", rank=1, lexical=0.7),
            _chunk("Second useful carbonate titration passage.", rank=2, lexical=0.6),
            _chunk("Third useful but lower-ranked carbonate passage.", rank=3, lexical=0.6),
            query=query,
        )
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed(
                {
                    query: [1.0, 0.0],
                    ctx.chunks[0].text: [0.95, 0.10],
                    ctx.chunks[1].text: [0.80, 0.20],
                    ctx.chunks[2].text: [0.70, 0.30],
                }
            ),
        )
        self.assertEqual(result.final_relevance, "high")
        self.assertEqual(result.passed_chunk_count, 2)
        assert result.context_for_prompt is not None
        self.assertEqual([chunk.rank for chunk in result.context_for_prompt.chunks], [1, 2])
        self.assertNotIn("Third useful", result.context_for_prompt.to_system_message_text())

    def test_embedding_error_falls_back_to_lexical_scores_without_crashing(self) -> None:
        query = "sodium carbonate titration"
        ctx = _context(
            _chunk("Sodium carbonate is titrated with hydrochloric acid.", rank=1, lexical=0.75),
            query=query,
        )

        def broken_embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
            raise RuntimeError("ollama is sleeping")

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=broken_embed_texts,
        )
        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.final_relevance, "high")
        self.assertTrue(result.used_fallback_scores)
        self.assertIn("RuntimeError", result.embedding_error or "")
        self.assertIsNotNone(result.context_for_prompt)
        self.assertIsNone(result.scores[0].embedding_score)

    def test_no_context_returns_no_context_decision(self) -> None:
        result = apply_rag_relevance_gate(
            query="anything",
            context=None,
            explicit_user_document_request=False,
            embed_texts=_fake_embed({}),
        )
        self.assertEqual(result.decision, "no_context")
        self.assertEqual(result.final_relevance, "none")
        self.assertIsNone(result.context_for_prompt)
        self.assertEqual(result.original_chunk_count, 0)

    def test_as_dict_omits_context_text_by_default(self) -> None:
        query = "weak acid"
        ctx = _context(_chunk("Weak acid titration text.", rank=1, lexical=0.9), query=query)
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            embed_texts=_fake_embed({query: [1.0, 0.0], ctx.chunks[0].text: [1.0, 0.0]}),
        )
        payload = result.as_dict()
        assert payload["context_for_prompt"] is not None
        self.assertEqual(payload["context_for_prompt"]["chunks"][0]["text"], "[omitted]")
        payload_with_text = result.as_dict(include_context_text=True)
        assert payload_with_text["context_for_prompt"] is not None
        self.assertIn("Weak acid", payload_with_text["context_for_prompt"]["chunks"][0]["text"])


if __name__ == "__main__":
    unittest.main()
