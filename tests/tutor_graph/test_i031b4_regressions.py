from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.response_assembly import (
    assess_plain_text_reliability,
    build_compact_rescue_payload,
    build_compact_rescue_prompt_text,
    deterministic_reliability_fallback,
)
from chemistry_bot.tutor_graph.thin_classifier import (
    derive_question_classification,
    explicit_user_document_request,
)


class I031B4RegressionTests(unittest.TestCase):
    def test_length_finish_reason_accepts_complete_answer_with_warning(self) -> None:
        result = assess_plain_text_reliability(
            "Thiosulfate is used because it reacts predictably with iodine in iodometric titrations.",
            response_metadata={"done_reason": "length", "eval_count": 1536},
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.is_truncated)
        self.assertEqual(result.finish_reason, "length")
        self.assertIn("finish_reason_length", result.warnings)
        self.assertNotIn("finish_reason_length", result.reasons)

    def test_length_finish_reason_retries_obviously_cut_off_answer(self) -> None:
        result = assess_plain_text_reliability(
            "Potassium permanganate is useful in titration because the product",
            response_metadata={"done_reason": "length", "eval_count": 1536},
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.is_truncated)
        self.assertIn("finish_reason_length", result.warnings)
        self.assertIn("missing_terminal_punctuation", result.reasons)

    def test_compact_retry_for_low_rag_omits_chunks_and_avoids_source_claims(self) -> None:
        payload = build_compact_rescue_payload(
            route="theory",
            user_text="According to downloaded books, why do we need thiosulphate for titration?",
            external_context={
                "relevance": "low",
                "source_documents": ["analytical_chemistry.pdf"],
                "chunks": [
                    {
                        "rank": 1,
                        "source_document": "analytical_chemistry.pdf",
                        "relevance_score": 0.12,
                        "text": "This unrelated chunk discusses zwitterions and amino acids.",
                    }
                ],
            },
        )
        prompt_text = build_compact_rescue_prompt_text(payload)

        self.assertEqual(payload["compact_chunks"], [])
        self.assertEqual(payload["external_context_relevance"], "low")
        self.assertIn("inconclusive", payload["external_context_note"])
        self.assertIn("Do not claim", payload["external_context_note"])
        self.assertNotIn("zwitterions", prompt_text)
        self.assertIn("under 120 words", prompt_text)

    def test_compact_retry_keeps_only_first_two_non_low_chunks(self) -> None:
        payload = build_compact_rescue_payload(
            route="reaction_illustration",
            user_text="Explain permanganate titration conceptually.",
            external_context={
                "relevance": "medium",
                "source_documents": ["book.pdf"],
                "chunks": [
                    {"rank": 1, "source_document": "book.pdf", "relevance_score": 0.8, "text": "Permanganate is purple."},
                    {"rank": 2, "source_document": "book.pdf", "relevance_score": 0.7, "text": "Manganese products are faintly colored."},
                    {"rank": 3, "source_document": "book.pdf", "relevance_score": 0.6, "text": "This third chunk must stay out."},
                ],
            },
        )
        prompt_text = build_compact_rescue_prompt_text(payload)

        self.assertEqual(len(payload["compact_chunks"]), 2)
        self.assertIn("Permanganate is purple", prompt_text)
        self.assertIn("Manganese products", prompt_text)
        self.assertNotIn("third chunk", prompt_text)
        self.assertIn("Do not include a follow-up question", prompt_text)

    def test_low_rag_explicit_document_fallback_is_honest_about_missing_evidence(self) -> None:
        fallback = deterministic_reliability_fallback(
            route="theory",
            user_text="According to downloaded books, why is NaOH not a primary standard?",
            external_context_relevance="low",
            needs_user_documents=True,
        )

        self.assertIn("retrieved uploaded-document excerpts", fallback)
        self.assertIn("book-supported", fallback)
        self.assertIn("exact term", fallback)

    def test_calculation_guard_blocks_qualitative_titration_questions(self) -> None:
        examples = [
            "Why do we need thiosulphate for titration?",
            "What is the role of thiosulphate in titration?",
            "Can you calculate why thiosulphate is used in titration?",
        ]
        for text in examples:
            with self.subTest(text=text):
                classification, debug = derive_question_classification(
                    "calculation_request",
                    user_text=text,
                    source="llm_label",
                )
                self.assertEqual(classification.main_type, "theory_learning_open")
                self.assertFalse(debug["signals"]["numeric_calculation_intent"])
                self.assertIn("no_numeric_calculation_intent", " ".join(debug["changes"]))

    def test_calculation_guard_keeps_quantitative_calculation_questions(self) -> None:
        examples = [
            "Calculate the molarity if 0.10 mol is dissolved in 500 mL.",
            "How many moles are in 5 g of NaCl?",
            "Balance H2 + O2 -> H2O.",
        ]
        for text in examples:
            with self.subTest(text=text):
                classification, debug = derive_question_classification(
                    "calculation_request",
                    user_text=text,
                    source="llm_label",
                )
                self.assertEqual(classification.main_type, "calculation_request")
                self.assertTrue(debug["signals"]["numeric_calculation_intent"])

    def test_downloaded_books_and_folder_phrases_request_user_documents(self) -> None:
        examples = [
            "According to downloaded books, why is NaOH not a primary standard?",
            "Use the downloaded documents for this answer.",
            "What do the files in the folder say about sodium carbonate titration?",
            "Ответь по скачанным документам, почему NaOH не primary standard?",
            "Поясни за файлами у папці, що пише книжка про титрування.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(explicit_user_document_request(text))
                classification, debug = derive_question_classification(
                    "reference_closed",
                    user_text=text,
                    source="llm_label",
                )
                self.assertTrue(classification.needs_user_documents)
                self.assertTrue(debug["signals"]["explicit_user_document_request"])

    def test_generic_book_mention_does_not_request_user_documents(self) -> None:
        classification, debug = derive_question_classification(
            "reference_closed",
            user_text="Can you recommend a chemistry book for beginners?",
            source="llm_label",
        )

        self.assertFalse(explicit_user_document_request("Can you recommend a chemistry book for beginners?"))
        self.assertFalse(classification.needs_user_documents)
        self.assertFalse(debug["signals"]["explicit_user_document_request"])


if __name__ == "__main__":
    unittest.main()
