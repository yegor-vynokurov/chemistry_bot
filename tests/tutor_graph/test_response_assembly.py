from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.response_assembly import (
    assemble_tutor_answer,
    assess_plain_text_reliability,
    build_compact_rescue_payload,
    build_compact_rescue_prompt_text,
    deterministic_reliability_fallback,
    extract_finish_reason,
    extract_ollama_response_metadata,
    extract_response_metadata,
    normalize_followup_question,
    normalize_plain_answer,
    route_requires_followup,
)


class ResponseAssemblyTests(unittest.TestCase):
    def test_plain_text_is_normalized_without_json_contract(self) -> None:
        text = normalize_plain_answer("```text\nAnswer: Atoms contain a nucleus.\n```")
        self.assertEqual(text, "Atoms contain a nucleus.")

    def test_theory_answer_gets_separate_question(self) -> None:
        answer = assemble_tutor_answer(
            route="theory",
            visible_answer="An atom is the basic unit of an element.",
            next_question="Question: How is an atom different from an ion? Extra text",
        )
        self.assertEqual(answer.tutor_move, "brief_answer_plus_check")
        self.assertEqual(answer.visible_answer, "An atom is the basic unit of an element.")
        self.assertEqual(answer.next_question, "How is an atom different from an ion?")
        self.assertNotIn("certainty", answer.model_dump())

    def test_failed_answer_can_suppress_default_followup(self) -> None:
        answer = assemble_tutor_answer(
            route="theory",
            visible_answer="The tutor model failed.",
            ensure_followup=False,
        )
        self.assertEqual(answer.tutor_move, "direct_answer")
        self.assertIsNone(answer.next_question)

    def test_reference_answer_drops_unrequested_followup(self) -> None:
        answer = assemble_tutor_answer(
            route="reference",
            visible_answer="Aluminium has atomic number 13.",
            next_question="Do you want more?",
        )
        self.assertEqual(answer.tutor_move, "direct_answer")
        self.assertIsNone(answer.next_question)

    def test_reaction_safety_note_is_programmatic(self) -> None:
        answer = assemble_tutor_answer(
            route="reaction_illustration",
            visible_answer="Acid-base neutralization forms salt and water.",
        )
        self.assertIsNotNone(answer.safety_note)
        self.assertIn("conceptual", answer.safety_note.lower())

    def test_invalid_followup_becomes_one_question(self) -> None:
        self.assertEqual(
            normalize_followup_question("Check question: Explain the idea in your own words"),
            "Explain the idea in your own words?",
        )

    def test_only_selected_routes_require_followup(self) -> None:
        self.assertTrue(route_requires_followup("theory"))
        self.assertTrue(route_requires_followup("diagnostic"))
        self.assertFalse(route_requires_followup("calculation"))

    def test_plain_text_reliability_accepts_complete_answer(self) -> None:
        result = assess_plain_text_reliability(
            "An atom is the smallest unit of an element that keeps that element's identity."
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.is_truncated)

    def test_plain_text_reliability_detects_empty_answer(self) -> None:
        result = assess_plain_text_reliability("   ")
        self.assertFalse(result.ok)
        self.assertTrue(result.is_empty)
        self.assertIn("empty_text", result.reasons)

    def test_plain_text_reliability_detects_dangling_answer(self) -> None:
        result = assess_plain_text_reliability(
            "Potassium permanganate can act as a self-indicator because it has a "
            "strong purple color, and after the analyte has reacted the product"
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.is_truncated)

    def test_finish_reason_is_extracted_from_ollama_metadata(self) -> None:
        self.assertEqual(extract_finish_reason({"done_reason": "stop"}), "stop")

    def test_length_finish_with_complete_text_is_warning_not_failure(self) -> None:
        result = assess_plain_text_reliability(
            "This answer reached the token budget, but the final sentence is complete.",
            response_metadata={"done_reason": "length"},
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.is_truncated)
        self.assertIn("finish_reason_length", result.warnings)
        self.assertIn("finish_reason_length", result.as_dict()["warnings"])
        self.assertNotIn("finish_reason_length", result.reasons)

    def test_length_finish_with_dangling_text_still_fails(self) -> None:
        result = assess_plain_text_reliability(
            "Potassium permanganate is useful in titration because it makes the math",
            response_metadata={"done_reason": "length"},
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.is_truncated)
        self.assertIn("finish_reason_length", result.warnings)
        self.assertIn("missing_terminal_punctuation", result.reasons)


    def test_ollama_metadata_summary_is_stable_for_nested_metadata(self) -> None:
        metadata = {
            "response_metadata": {
                "model": "gemma4:12b",
                "done_reason": "length",
                "total_duration": 123456,
                "prompt_eval_count": 321,
                "eval_count": 1536,
            },
            "usage_metadata": {
                "input_tokens": 321,
                "output_tokens": 1536,
                "total_tokens": 1857,
            },
        }
        summary = extract_ollama_response_metadata(metadata)
        self.assertEqual(summary["provider"], "ollama")
        self.assertEqual(summary["finish_reason"], "length")
        self.assertEqual(summary["done_reason"], "length")
        self.assertEqual(summary["model"], "gemma4:12b")
        self.assertEqual(summary["prompt_eval_count"], 321)
        self.assertEqual(summary["eval_count"], 1536)
        self.assertEqual(summary["total_duration"], 123456)
        self.assertIn("usage_metadata", summary)

    def test_provider_metadata_can_be_extracted_from_message_like_object(self) -> None:
        class FakeMessage:
            response_metadata = {"done_reason": "stop", "eval_count": 42}
            additional_kwargs = {"nested": {"prompt_eval_count": 11}}
            usage_metadata = {"input_tokens": 11, "output_tokens": 42}

        raw = extract_response_metadata(FakeMessage())
        summary = extract_ollama_response_metadata(raw)
        self.assertEqual(summary["finish_reason"], "stop")
        self.assertEqual(summary["eval_count"], 42)
        self.assertEqual(summary["prompt_eval_count"], 11)


    def test_compact_rescue_payload_excludes_low_relevance_chunks(self) -> None:
        payload = build_compact_rescue_payload(
            route="reference",
            user_text="Why do we need thiosulphate for titration?",
            external_context={
                "relevance": "low",
                "source_documents": ["analytical.pdf"],
                "chunks": [
                    {
                        "rank": 1,
                        "source_document": "analytical.pdf",
                        "relevance_score": 0.12,
                        "text": "A long irrelevant excerpt about zwitterions.",
                    }
                ],
            },
        )
        self.assertEqual(payload["external_context_relevance"], "low")
        self.assertEqual(payload["external_context_sources"], ["analytical.pdf"])
        self.assertEqual(payload["compact_chunks"], [])
        self.assertIn("inconclusive", payload["external_context_note"])

    def test_compact_rescue_payload_keeps_only_two_non_low_chunks(self) -> None:
        payload = build_compact_rescue_payload(
            route="theory",
            user_text="Explain permanganate titration.",
            external_context={
                "relevance": "medium",
                "source_documents": ["book.pdf"],
                "chunks": [
                    {"rank": 1, "source_document": "book.pdf", "relevance_score": 0.8, "text": "First useful chunk."},
                    {"rank": 2, "source_document": "book.pdf", "relevance_score": 0.7, "text": "Second useful chunk."},
                    {"rank": 3, "source_document": "book.pdf", "relevance_score": 0.6, "text": "Third chunk should be omitted."},
                ],
            },
        )
        self.assertEqual(len(payload["compact_chunks"]), 2)
        rendered = build_compact_rescue_prompt_text(payload)
        self.assertIn("First useful chunk", rendered)
        self.assertIn("Second useful chunk", rendered)
        self.assertNotIn("Third chunk should be omitted", rendered)
        self.assertIn("under 120 words", rendered)

    def test_deterministic_fallback_avoids_source_claim_for_low_relevance(self) -> None:
        fallback = deterministic_reliability_fallback(
            route="reference",
            user_text="Why do we need thiosulphate?",
            external_context_relevance="low",
        )
        self.assertIn("retrieved excerpts", fallback)
        self.assertIn("should not claim", fallback)

    def test_low_relevance_explicit_document_fallback_avoids_book_supported_answer(self) -> None:
        fallback = deterministic_reliability_fallback(
            route="theory",
            user_text="According to downloaded books, why do we need thiosulphate?",
            external_context_relevance="low",
            needs_user_documents=True,
        )
        self.assertIn("direct answer", fallback)
        self.assertIn("book-supported", fallback)
        self.assertIn("exact term", fallback)

    def test_reaction_fallback_stays_conceptual_and_nonprocedural(self) -> None:
        fallback = deterministic_reliability_fallback(
            route="reaction_illustration",
            user_text="Can you explain permanganate titration safely?",
        )
        self.assertIn("conceptual", fallback)
        self.assertIn("without giving", fallback)
        self.assertIn("lab procedure", fallback)

    def test_safety_fallback_refuses_steps_but_offers_conceptual_help(self) -> None:
        fallback = deterministic_reliability_fallback(
            route="safety_redirect",
            user_text="how to make cyanide at home",
        )
        self.assertIn("can’t help", fallback)
        self.assertIn("instructions", fallback)
        self.assertIn("safe conceptual", fallback)


if __name__ == "__main__":
    unittest.main()
