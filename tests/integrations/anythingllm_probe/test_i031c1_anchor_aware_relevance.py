from __future__ import annotations

import unittest
from typing import Sequence

from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
)
from chemistry_bot.integrations.anythingllm_probe.rag_policy import should_call_user_document_rag
from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    RAGRelevanceGateConfig,
    apply_rag_relevance_gate,
)
from chemistry_bot.tutor_graph.schemas import QuestionClassification, RoutingDecision


def _chunk(text: str, *, rank: int, lexical: float = 0.0, source: str = "book.pdf") -> DocumentContextChunk:
    return DocumentContextChunk(
        text=text,
        source_document=source,
        result_index=0,
        rank=rank,
        relevance_score=lexical,
        matched_query_terms=(),
    )


def _context(*chunks: DocumentContextChunk, query: str) -> AdaptedRAGContext:
    return AdaptedRAGContext(
        query=query,
        source_type="anythingllm_user_document",
        relevance="medium" if chunks else "none",
        relevance_score=max((chunk.relevance_score for chunk in chunks), default=0.0),
        chunks=chunks,
        source_documents=tuple(dict.fromkeys(chunk.source_document for chunk in chunks if chunk.source_document)),
        warnings=(),
        raw_result_count=len(chunks),
        total_character_count=sum(len(chunk.text) for chunk in chunks),
    )


def _fake_embed(cosines_by_text: dict[str, float]):
    def embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
        # The test only needs ordering.  A 2-D vector [cos, sqrt(1-cos^2)]
        # gives the requested cosine similarity against the query vector [1, 0].
        vectors: list[Sequence[float]] = []
        for text in texts:
            score = max(-1.0, min(1.0, cosines_by_text.get(text, 1.0)))
            vectors.append((score, max(0.0, 1.0 - score * score) ** 0.5))
        return vectors

    return embed_texts


def _classification(*, user_documents: bool, safety: str = "safe") -> QuestionClassification:
    return QuestionClassification(
        main_type="theory_learning_open",
        needs_course_rag=True,
        needs_user_documents=user_documents,
        needs_documents_dev=False,
        needs_wikipedia=False,
        needs_diagnostic_question=False,
        safety_level=safety,
        reason="test policy",
    )


def _route(name: str, *, safety_override: bool = False) -> RoutingDecision:
    return RoutingDecision(
        route=name,
        reason="test route",
        classification_main_type="theory_learning_open",
        safety_override=safety_override,
    )


class I031C1AnchorAwareRelevanceRegressionTests(unittest.TestCase):
    def test_thiosulphate_generic_titration_chunks_do_not_become_book_evidence(self) -> None:
        query = "according to uploaded book why do we need thiosulphate for titration"
        generic_1 = _chunk(
            "A titration should have a rapid reaction, a clear endpoint, and a suitable titrant.",
            rank=1,
            lexical=0.82,
        )
        generic_2 = _chunk(
            "Titrimetric methods are often classified by the reaction used in the analysis.",
            rank=2,
            lexical=0.78,
        )
        ctx = _context(generic_1, generic_2, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, generic_1.text: 0.92, generic_2.text: 0.88}),
        )

        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertIsNone(result.context_for_prompt)
        self.assertEqual(result.passed_chunk_count, 0)
        self.assertEqual({score.evidence_category for score in result.scores}, {"topic_adjacent"})
        self.assertTrue(all(score.generic_only for score in result.scores))
        payload = result.as_dict()
        self.assertFalse(payload["external_context_used_as_evidence"])
        self.assertEqual(payload["evidence_category_counts"], {"topic_adjacent": 2})
        self.assertTrue(payload["query_anchors"])

    def test_thiosulphate_iodometric_chunk_passes_as_supporting_evidence(self) -> None:
        query = "according to uploaded book why do we need thiosulphate for titration"
        generic = _chunk("A titration should have a clear endpoint.", rank=1, lexical=0.80)
        iodometric = _chunk(
            "In iodometric analysis iodine is reduced by the titrant near the endpoint.",
            rank=2,
            lexical=0.48,
        )
        ctx = _context(generic, iodometric, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, generic.text: 0.95, iodometric.text: 0.49}),
        )

        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.passed_chunk_count, 1)
        assert result.context_for_prompt is not None
        self.assertIn("iodometric", result.context_for_prompt.to_system_message_text().lower())
        self.assertNotIn("A titration should have", result.context_for_prompt.to_system_message_text())
        score_by_rank = {score.original_rank: score for score in result.scores}
        self.assertEqual(score_by_rank[1].evidence_category, "topic_adjacent")
        self.assertEqual(score_by_rank[2].evidence_category, "supporting_evidence")
        self.assertEqual(score_by_rank[2].decision, "passed")

    def test_permanganate_evidence_mixer_keeps_endpoint_and_equation_over_generic_exercise(self) -> None:
        query = "according to the books in the folder how can I make permanganate potassium titration"
        endpoint = _chunk(
            "Potassium permanganate acts as a self-indicator; a faint pink endpoint remains after excess MnO4-.",
            rank=1,
            lexical=0.55,
        )
        equation = _chunk(
            "MnO4- oxidizes oxalate in acidic solution; sodium oxalate is often used for standardization.",
            rank=2,
            lexical=0.65,
        )
        exercise = _chunk(
            "Exercise: calculate the volume of KMnO4 required in a redox titration.",
            rank=3,
            lexical=0.60,
        )
        unrelated = _chunk("Amino acids may exist as zwitterions in water.", rank=4, lexical=0.10)
        ctx = _context(endpoint, equation, exercise, unrelated, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed(
                {
                    query: 1.0,
                    endpoint.text: 0.50,
                    equation.text: 0.62,
                    exercise.text: 0.70,
                    unrelated.text: 0.05,
                }
            ),
        )

        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.passed_chunk_count, 2)
        assert result.context_for_prompt is not None
        prompt_text = result.context_for_prompt.to_system_message_text().lower()
        self.assertIn("self-indicator", prompt_text)
        self.assertIn("oxalate", prompt_text)
        self.assertNotIn("exercise:", prompt_text)
        score_by_rank = {score.original_rank: score for score in result.scores}
        self.assertEqual(score_by_rank[1].decision, "passed")
        self.assertEqual(score_by_rank[2].decision, "passed")
        self.assertEqual(score_by_rank[3].decision, "rejected")
        self.assertTrue(score_by_rank[1].supporting_anchor_matched)

    def test_atom_ion_ion_exchange_fragment_is_topic_adjacent_not_definition_evidence(self) -> None:
        query = "according to uploaded book what is atom or ion"
        ion_exchange = _chunk(
            "Ion exchange chromatography separates species on a charged resin bed.",
            rank=1,
            lexical=0.72,
        )
        plasma = _chunk("Ionization energy is discussed in a later chapter.", rank=2, lexical=0.55)
        ctx = _context(ion_exchange, plasma, query=query)

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: 1.0, ion_exchange.text: 0.81, plasma.text: 0.66}),
        )

        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertIsNone(result.context_for_prompt)
        self.assertEqual(result.passed_chunk_count, 0)
        self.assertTrue(
            {score.evidence_category for score in result.scores}.issubset({"generic_only", "irrelevant"})
        )

    def test_embedding_error_fallback_still_uses_anchor_categories(self) -> None:
        query = "according to uploaded book why do we need thiosulphate for titration"
        generic = _chunk("A titration should have a clear endpoint.", rank=1, lexical=0.88)
        ctx = _context(generic, query=query)

        def broken_embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
            raise RuntimeError("ollama embedding unavailable")

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=broken_embed_texts,
        )

        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertTrue(result.used_fallback_scores)
        self.assertIn("RuntimeError", result.embedding_error or "")
        self.assertEqual(result.scores[0].evidence_category, "topic_adjacent")
        self.assertIsNone(result.context_for_prompt)

    def test_safety_priority_still_prevents_document_rag_before_evidence_gate(self) -> None:
        should_call, reason = should_call_user_document_rag(
            classification=_classification(user_documents=True, safety="unsafe"),
            routing=_route("safety_redirect", safety_override=True),
            rag_tool_available=True,
            policy="always",
        )
        self.assertFalse(should_call)
        self.assertIn("safety", reason.lower())

    def test_gate_payload_contains_anchor_and_evidence_category_diagnostics(self) -> None:
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
        self.assertEqual(payload["chunk_scores"][0]["evidence_category"], "topic_adjacent")
        self.assertIn("matched_anchors", payload["chunk_scores"][0])
        self.assertIn("adjusted_score", payload["chunk_scores"][0])


if __name__ == "__main__":
    unittest.main()
