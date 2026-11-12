from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.semantic_safety import SemanticSafetyGate
from chemistry_bot.tutor_graph.thin_classifier import (
    derive_question_classification,
    explicit_user_document_request,
    has_numeric_calculation_intent,
    parse_main_type_label,
    rule_based_main_type,
)


class ThinClassifierTests(unittest.TestCase):
    def test_parser_accepts_one_plain_label(self) -> None:
        parsed = parse_main_type_label("theory_learning_open")
        self.assertEqual(parsed.main_type, "theory_learning_open")

    def test_parser_rejects_multiple_labels(self) -> None:
        with self.assertRaises(ValueError):
            parse_main_type_label("reference_closed or theory_learning_open")

    def test_uploaded_book_is_derived_in_code(self) -> None:
        classification, debug = derive_question_classification(
            "reference_closed",
            user_text="According to the uploaded book, why is NaOH not a primary standard?",
            source="llm_label",
        )
        self.assertTrue(classification.needs_user_documents)
        self.assertFalse(classification.needs_wikipedia)
        self.assertTrue(debug["signals"]["explicit_user_document_request"])


    def test_downloaded_books_are_explicit_user_documents(self) -> None:
        classification, debug = derive_question_classification(
            "reference_closed",
            user_text="According to downloaded books, why is NaOH not a primary standard?",
            source="llm_label",
        )
        self.assertTrue(classification.needs_user_documents)
        self.assertFalse(classification.needs_wikipedia)
        self.assertTrue(debug["signals"]["explicit_user_document_request"])

    def test_folder_document_phrases_are_explicit_user_documents(self) -> None:
        phrases = (
            "Please use the books in the folder to explain this.",
            "What do the files in the folder say about sodium carbonate titration?",
            "Use the downloaded documents for this answer.",
            "Answer from the folder, not from general knowledge.",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(explicit_user_document_request(phrase))

    def test_document_phrase_detector_avoids_generic_book_mentions(self) -> None:
        self.assertFalse(explicit_user_document_request("Can you recommend a chemistry book?"))

    def test_atom_uses_course_rag_not_user_book(self) -> None:
        classification, _ = derive_question_classification(
            "theory_learning_open",
            user_text="What is an atom?",
            source="llm_label",
        )
        self.assertTrue(classification.needs_course_rag)
        self.assertFalse(classification.needs_user_documents)

    def test_developer_document_override_forces_user_document_flag(self) -> None:
        classification, debug = derive_question_classification(
            "theory_learning_open",
            user_text="What is an atom?",
            source="llm_label",
            needs_documents_dev=True,
        )
        self.assertTrue(classification.needs_documents_dev)
        self.assertTrue(classification.needs_user_documents)
        self.assertTrue(debug["signals"]["needs_documents_dev"])
        self.assertIn("developer_document_override", classification.reason)

    def test_developer_override_does_not_change_main_type(self) -> None:
        classification, _ = derive_question_classification(
            "reference_closed",
            user_text="How many electrons does aluminium have?",
            needs_documents_dev=True,
        )
        self.assertEqual(classification.main_type, "reference_closed")
        self.assertTrue(classification.needs_user_documents)

    def test_previous_unsafe_label_cannot_leak(self) -> None:
        classification, _ = derive_question_classification(
            "reference_closed",
            user_text="What is an atom?",
            session_state={"last_classification_safety_level": "unsafe"},
            source="llm_label",
        )
        self.assertEqual(classification.safety_level, "safe")

    def test_explosive_request_is_overridden_by_code(self) -> None:
        safety = SemanticSafetyGate(enabled=False).evaluate("How to make black powder?")
        classification, _ = derive_question_classification(
            "theory_learning_open",
            user_text="How to make black powder?",
            source="llm_label",
            safety_assessment=safety,
        )
        self.assertEqual(classification.main_type, "experiment_request_caution")
        self.assertEqual(classification.safety_level, "unsafe")

    def test_new_question_relation_prevents_student_answer_override(self) -> None:
        classification, debug = derive_question_classification(
            "reference_closed",
            user_text="What is an atom?",
            source="llm_label",
            turn_relation="new_question",
        )
        self.assertEqual(classification.main_type, "reference_closed")
        self.assertEqual(classification.turn_relation, "new_question")
        self.assertEqual(debug["signals"]["turn_relation"], "new_question")

    def test_student_answer_relation_is_applied_explicitly(self) -> None:
        classification, _ = derive_question_classification(
            "theory_learning_open",
            user_text="Because it loses an electron.",
            source="llm_label",
            turn_relation="student_answer",
        )
        self.assertEqual(classification.main_type, "student_answer_to_assess")

    def test_unresolved_clarification_choice_becomes_new_question_when_content_is_clear(self) -> None:
        classification, debug = derive_question_classification(
            "reference_closed",
            user_text="Is valence an outdated term?",
            source="llm_label",
            turn_relation="clarification_choice",
            session_state={"last_tutor_move": "clarification_question"},
        )
        self.assertEqual(classification.main_type, "reference_closed")
        self.assertEqual(classification.turn_relation, "new_question")
        self.assertEqual(debug["signals"]["turn_relation_raw"], "clarification_choice")

    def test_safe_redirect_choice_maps_to_reaction_route(self) -> None:
        classification, _ = derive_question_classification(
            "theory_learning_open",
            user_text="Yes, I would like a safe conceptual explanation of the reaction.",
            source="llm_label",
            turn_relation="clarification_choice",
            session_state={"last_tutor_move": "safe_redirect"},
        )
        self.assertEqual(classification.main_type, "reaction_illustration_request")
        self.assertEqual(classification.turn_relation, "clarification_choice")

    def test_rule_fallback_returns_only_label_contract(self) -> None:
        thin = rule_based_main_type("Calculate molarity from moles and volume")
        self.assertEqual(thin.model_dump(), {"main_type": "calculation_request"})

    def test_calculation_route_requires_numeric_intent(self) -> None:
        classification, debug = derive_question_classification(
            "calculation_request",
            user_text="Why do we need thiosulphate for titration?",
            source="llm_label",
        )
        self.assertEqual(classification.main_type, "theory_learning_open")
        self.assertIn("no_numeric_calculation_intent", " ".join(debug["changes"]))
        self.assertFalse(debug["signals"]["numeric_calculation_intent"])


    def test_calculation_guard_blocks_qualitative_role_question(self) -> None:
        classification, debug = derive_question_classification(
            "calculation_request",
            user_text="What is the role of thiosulphate in titration?",
            source="llm_label",
        )
        self.assertEqual(classification.main_type, "theory_learning_open")
        self.assertFalse(debug["signals"]["numeric_calculation_intent"])

    def test_calculation_guard_blocks_calculate_why_question(self) -> None:
        classification, debug = derive_question_classification(
            "calculation_request",
            user_text="Can you calculate why thiosulphate is used in titration?",
            source="llm_label",
        )
        self.assertEqual(classification.main_type, "theory_learning_open")
        self.assertFalse(debug["signals"]["numeric_calculation_intent"])

    def test_rule_fallback_does_not_route_qualitative_ph_as_calculation(self) -> None:
        thin = rule_based_main_type("What is pH?")
        self.assertNotEqual(thin.main_type, "calculation_request")

    def test_calculation_guard_preserves_molar_mass_without_digits(self) -> None:
        self.assertTrue(has_numeric_calculation_intent("Calculate molar mass of NaCl."))

    def test_calculation_guard_preserves_amount_question(self) -> None:
        self.assertTrue(has_numeric_calculation_intent("How many moles are in 5 g of NaCl?"))

    def test_calculation_guard_preserves_balancing_request(self) -> None:
        self.assertTrue(has_numeric_calculation_intent("Balance H2 + O2 -> H2O"))

    def test_real_calculation_route_is_preserved(self) -> None:
        classification, debug = derive_question_classification(
            "calculation_request",
            user_text="Calculate the molarity if 0.10 mol is dissolved in 500 mL.",
            source="llm_label",
        )
        self.assertEqual(classification.main_type, "calculation_request")
        self.assertTrue(debug["signals"]["numeric_calculation_intent"])

    def test_numeric_intent_helper_does_not_treat_theory_as_calculation(self) -> None:
        self.assertFalse(has_numeric_calculation_intent("Why do we need titration?"))
        self.assertTrue(has_numeric_calculation_intent("Calculate pH of a 0.10 M acid."))


if __name__ == "__main__":
    unittest.main()
