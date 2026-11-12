from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
)
from chemistry_bot.integrations.anythingllm_probe.rag_policy import should_call_user_document_rag
from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    RAGRelevanceGateConfig,
    RAGRelevanceGateResult,
    apply_rag_relevance_gate,
)
from chemistry_bot.integrations.anythingllm_probe.tool_cycle import (
    ToolInteractionLog,
    InteractionLogSettings,
    inspect_tool_cycle,
    make_interaction_id,
    make_tool_call_id,
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


def _context(*chunks: DocumentContextChunk, query: str = "student question") -> AdaptedRAGContext:
    return AdaptedRAGContext(
        query=query,
        source_type="anythingllm_user_document",
        relevance="medium" if chunks else "none",
        relevance_score=max((chunk.relevance_score for chunk in chunks), default=0.0),
        chunks=chunks,
        source_documents=tuple(
            dict.fromkeys(chunk.source_document for chunk in chunks if chunk.source_document)
        ),
        warnings=(),
        raw_result_count=len(chunks),
        total_character_count=sum(len(chunk.text) for chunk in chunks),
    )


def _fake_embed(vectors_by_text: dict[str, Sequence[float]]):
    def embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
        return [vectors_by_text[text] for text in texts]

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


class I031CRelevanceGateRegressionTests(unittest.TestCase):
    def test_low_relevance_non_explicit_suppresses_evidence(self) -> None:
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
        self.assertIsNone(result.context_for_prompt)
        self.assertFalse(result.as_dict()["external_context_used_as_evidence"])
        self.assertEqual(result.as_dict()["passed_chunk_scores"], [])
        self.assertEqual(len(result.as_dict()["rejected_chunk_scores"]), 2)

    def test_explicit_document_low_relevance_returns_deterministic_not_found_decision(self) -> None:
        query = "according to uploaded book why use thiosulphate"
        ctx = _context(_chunk("This passage is about amino acid zwitterions.", rank=1, lexical=0.08), query=query)
        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=_fake_embed({query: [1.0, 0.0], ctx.chunks[0].text: [0.05, 0.95]}),
        )
        self.assertEqual(result.decision, "explicit_document_not_found")
        self.assertFalse(result.as_dict()["external_context_used_as_evidence"])
        self.assertIsNone(result.context_for_prompt)
        self.assertTrue(result.explicit_user_document_request)

    def test_medium_and_high_relevance_pass_only_top_k_evidence(self) -> None:
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
        self.assertEqual(result.decision, "pass_context")
        self.assertEqual(result.final_relevance, "high")
        self.assertEqual(result.passed_chunk_count, 2)
        self.assertEqual(result.rejected_chunk_count, 1)
        assert result.context_for_prompt is not None
        self.assertNotIn("Third useful", result.context_for_prompt.to_system_message_text())
        payload = result.as_dict()
        self.assertEqual(payload["score_summary"]["passed_indices"], [0, 1])
        self.assertEqual(payload["score_summary"]["rejected_indices"], [2])

    def test_embedding_error_falls_back_without_crashing_and_logs_diagnostic(self) -> None:
        query = "weak acid titration"
        ctx = _context(_chunk("Weak acid strong base titration has a buffer region.", rank=1, lexical=0.76), query=query)

        def broken_embed_texts(texts: Sequence[str]) -> list[Sequence[float]]:
            raise RuntimeError("ollama embedding unavailable")

        result = apply_rag_relevance_gate(
            query=query,
            context=ctx,
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=broken_embed_texts,
        )
        self.assertEqual(result.decision, "pass_context")
        self.assertTrue(result.used_fallback_scores)
        self.assertIn("RuntimeError", result.embedding_error or "")
        self.assertIsNone(result.scores[0].embedding_score)
        self.assertTrue(result.as_dict()["external_context_used_as_evidence"])

    def test_safety_priority_prevents_document_rag_even_when_policy_is_always(self) -> None:
        should_call, reason = should_call_user_document_rag(
            classification=_classification(user_documents=True, safety="unsafe"),
            routing=_route("safety_redirect", safety_override=True),
            rag_tool_available=True,
            policy="always",
        )
        self.assertFalse(should_call)
        self.assertIn("safety", reason.lower())

    def test_interaction_log_records_top_level_relevance_gate_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = ToolInteractionLog(InteractionLogSettings(log_dir=Path(tmp), log_content=False))
            interaction_id = make_interaction_id()
            call_id = make_tool_call_id(interaction_id)
            log.start(
                interaction_id=interaction_id,
                tool_call_id=call_id,
                created_at="2026-07-22T12:00:00.000Z",
                session_id="session-1",
                student_id="student-1",
                initial_request_id="req-initial",
                initial_request_fingerprint="fp-initial",
                query="according to uploaded book why use thiosulphate",
                resolution=None,
                parsed_conversation=None,
                routing_preview=None,
            )
            inspection = inspect_tool_cycle(
                {
                    "messages": [
                        {"role": "user", "content": "according to uploaded book why use thiosulphate"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "rag-memory", "arguments": "{}"},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": call_id, "content": "zwitterion text"},
                    ]
                }
            )
            gate_payload = {
                "enabled": True,
                "decision": "explicit_document_not_found",
                "final_relevance": "low",
                "original_chunk_count": 1,
                "passed_chunk_count": 0,
                "rejected_chunk_count": 1,
                "chunk_scores": [{"index": 0, "decision": "rejected", "embedding_score": 0.08}],
            }
            row = log.complete(
                interaction_id=interaction_id,
                result_request_id="req-result",
                completed_at="2026-07-22T12:00:01.000Z",
                inspection=inspection,
                final_response_text="No direct answer in retrieved fragments.",
                persisted_turn_ids={"student": 1, "tutor": 2},
                adapted_context={"relevance": "low", "chunks": [{"text": "[omitted]"}]},
                rag_relevance_gate=gate_payload,
                external_context_used_as_evidence=False,
                graph_result={"deterministic_response": True, "model_invoked": False},
            )
            assert row is not None
            self.assertEqual(row["rag_relevance_gate"]["decision"], "explicit_document_not_found")
            self.assertFalse(row["external_context_used_as_evidence"])
            self.assertTrue(row["external_context_received"])


if __name__ == "__main__":
    unittest.main()
