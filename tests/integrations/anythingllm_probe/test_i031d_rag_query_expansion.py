from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemistry_bot.integrations.anythingllm_probe.rag_query_expansion import (
    RAGQueryExpansionConfig,
    choose_query_expansion,
)
from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    RAGRelevanceGateConfig,
    apply_rag_relevance_gate,
)
from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    AdaptedRAGContext,
    DocumentContextChunk,
)
from chemistry_bot.integrations.anythingllm_probe.tool_cycle import (
    InteractionLogSettings,
    ToolInteractionLog,
    inspect_tool_cycle,
    interaction_id_from_tool_call_id,
    make_interaction_id,
    make_retry_tool_call_id,
    make_tool_call_id,
)


def _chunk(text: str, *, rank: int = 1, lexical: float = 0.5) -> DocumentContextChunk:
    return DocumentContextChunk(
        text=text,
        source_document="book.pdf",
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
        source_documents=("book.pdf",) if chunks else (),
        warnings=(),
        raw_result_count=len(chunks),
        total_character_count=sum(len(chunk.text) for chunk in chunks),
    )


class I031DRAGQueryExpansionTests(unittest.TestCase):
    def test_light_expansion_adds_aliases_but_not_supporting_terms_before_first_retrieval(self) -> None:
        result = choose_query_expansion(
            original_query="depend uploaded book why do we need tiosulphate for titration",
            should_call_user_documents=True,
            explicit_document_request=True,
            route="theory",
            safety_level="safe",
            config=RAGQueryExpansionConfig(max_added_terms_light=8),
        )

        self.assertEqual(result.mode, "light")
        self.assertTrue(result.changed)
        self.assertIn("thiosulfate", result.expanded_query)
        self.assertIn("sodium thiosulfate", result.expanded_query)
        self.assertNotIn("iodometric", result.expanded_query)
        self.assertIn("recognized_anchor", result.reason_codes)
        self.assertIn("typo_or_formula_variant_detected", result.reason_codes)

    def test_no_expansion_when_document_rag_is_not_called_or_safety_blocks(self) -> None:
        no_docs = choose_query_expansion(
            original_query="what is KMnO4",
            should_call_user_documents=False,
            explicit_document_request=False,
            route="theory",
            safety_level="safe",
        )
        self.assertEqual(no_docs.mode, "none")
        self.assertIn("no_user_document_retrieval", no_docs.reason_codes)

        unsafe = choose_query_expansion(
            original_query="depend uploaded book potassium cyanide synthesis",
            should_call_user_documents=True,
            explicit_document_request=True,
            route="safety_redirect",
            safety_level="unsafe",
        )
        self.assertEqual(unsafe.mode, "none")
        self.assertIn("safety_disabled", unsafe.reason_codes)

    def test_targeted_retry_after_topic_adjacent_gate_adds_supporting_terms(self) -> None:
        query = "depend uploaded book why do we need tiosulphate for titration"
        generic = _chunk("Titrimetric methods require a clear endpoint.", lexical=0.92)
        gate = apply_rag_relevance_gate(
            query=query,
            context=_context(generic, query=query),
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=lambda texts: [[1.0, 0.0], [0.95, 0.05]],
        )
        self.assertEqual(gate.decision, "explicit_document_not_found")

        initial = choose_query_expansion(
            original_query=query,
            should_call_user_documents=True,
            explicit_document_request=True,
            route="theory",
            safety_level="safe",
        )
        retry = choose_query_expansion(
            original_query=query,
            should_call_user_documents=True,
            explicit_document_request=True,
            route="theory",
            safety_level="safe",
            first_gate_result=gate,
            previous_expansion=initial.as_dict(),
        )

        self.assertEqual(retry.mode, "retry_targeted")
        self.assertTrue(retry.retry_after_low_relevance)
        self.assertIn("iodometric titration", retry.expanded_query)
        self.assertIn("iodine", retry.expanded_query)
        self.assertIn("first_retrieval_low_or_topic_adjacent", retry.reason_codes)

    def test_no_second_retry_when_previous_expansion_was_already_targeted(self) -> None:
        query = "according to uploaded book what is atom or ion"
        generic = _chunk("Ion exchange chromatography separates charged species.", lexical=0.9)
        gate = apply_rag_relevance_gate(
            query=query,
            context=_context(generic, query=query),
            explicit_user_document_request=True,
            config=RAGRelevanceGateConfig(medium_threshold=0.45, high_threshold=0.58, top_k=2),
            embed_texts=lambda texts: [[1.0, 0.0], [0.9, 0.1]],
        )
        retry = choose_query_expansion(
            original_query=query,
            should_call_user_documents=True,
            explicit_document_request=True,
            route="theory",
            safety_level="safe",
            first_gate_result=gate,
            previous_expansion={"mode": "retry_targeted"},
        )
        self.assertEqual(retry.mode, "none")
        self.assertIn("already_targeted", retry.reason_codes)

    def test_retry_tool_call_id_maps_to_same_interaction(self) -> None:
        interaction_id = make_interaction_id()
        call_id = make_tool_call_id(interaction_id)
        retry_id = make_retry_tool_call_id(interaction_id, 1)
        self.assertEqual(interaction_id_from_tool_call_id(call_id), interaction_id)
        self.assertEqual(interaction_id_from_tool_call_id(retry_id), interaction_id)
        body = {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": "rag-memory", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "first result"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": retry_id, "type": "function", "function": {"name": "rag-memory", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": retry_id, "content": "second result"},
            ]
        }
        inspection = inspect_tool_cycle(body)
        self.assertEqual(inspection.interaction_id, interaction_id)
        self.assertEqual(inspection.phase, "tool_result_return")
        self.assertEqual(len(inspection.tool_results), 2)

    def test_interaction_log_records_initial_and_retry_query_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = ToolInteractionLog(InteractionLogSettings(log_dir=Path(tmp), log_content=True))
            interaction_id = make_interaction_id()
            call_id = make_tool_call_id(interaction_id)
            retry_id = make_retry_tool_call_id(interaction_id, 1)
            initial_expansion = choose_query_expansion(
                original_query="why tiosulphate titration",
                should_call_user_documents=True,
                explicit_document_request=True,
                route="theory",
                safety_level="safe",
            )
            log.start(
                interaction_id=interaction_id,
                tool_call_id=call_id,
                created_at="2026-07-30T04:00:00.000Z",
                session_id="s1",
                student_id="student",
                initial_request_id="req1",
                initial_request_fingerprint="fp1",
                query=initial_expansion.expanded_query,
                original_query="why tiosulphate titration",
                query_expansion=initial_expansion.as_dict(),
                resolution=None,
                parsed_conversation=None,
                routing_preview=None,
            )
            retry_expansion = choose_query_expansion(
                original_query="why tiosulphate titration",
                should_call_user_documents=True,
                explicit_document_request=True,
                route="theory",
                safety_level="safe",
                first_gate_result={"decision": "explicit_document_not_found", "final_relevance": "low", "passed_chunk_count": 0},
                previous_expansion=initial_expansion.as_dict(),
            )
            row = log.record_retry(
                interaction_id=interaction_id,
                request_id="req2",
                retry_tool_call_id=retry_id,
                retry_query=retry_expansion.expanded_query,
                query_expansion=retry_expansion.as_dict(),
                rag_relevance_gate={"decision": "explicit_document_not_found"},
            )
            assert row is not None
            self.assertEqual(row["retry_count"], 1)
            self.assertEqual(row["rag_query_expansion"]["mode"], "light")
            self.assertEqual(row["retry_query_expansion"]["mode"], "retry_targeted")
            self.assertIn("iodine", row["retry_query"])


if __name__ == "__main__":
    unittest.main()
