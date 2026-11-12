from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.turn_relation import (
    allowed_turn_relations,
    apply_turn_relation,
    build_turn_relation_payload,
    parse_turn_relation_label,
    resolve_clarification_choice_main_type,
    rule_based_turn_relation,
    should_classify_turn_relation,
    turn_relation_context,
)


class TurnRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.awaiting = {
            "awaiting_student_answer": True,
            "last_bot_question": "What is the difference between an atom and an ion?",
            "last_tutor_move": "brief_answer_plus_check",
        }

    def test_parser_accepts_one_plain_label(self) -> None:
        parsed = parse_turn_relation_label("new_question")
        self.assertEqual(parsed.relation, "new_question")

    def test_parser_rejects_multiple_labels(self) -> None:
        with self.assertRaises(ValueError):
            parse_turn_relation_label("student_answer or new_question")

    def test_parser_rejects_state_impossible_label(self) -> None:
        with self.assertRaises(ValueError):
            parse_turn_relation_label(
                "clarification_choice",
                allowed_labels=("student_answer", "new_question", "unclear"),
            )

    def test_relation_call_is_only_needed_when_tutor_awaits_reply(self) -> None:
        self.assertTrue(should_classify_turn_relation(self.awaiting))
        self.assertFalse(should_classify_turn_relation({}))

    def test_check_question_excludes_clarification_choice(self) -> None:
        self.assertEqual(
            allowed_turn_relations(self.awaiting),
            ("student_answer", "new_question", "unclear"),
        )
        self.assertEqual(turn_relation_context(self.awaiting), "answer_to_check_question")

    def test_clarification_question_excludes_student_answer(self) -> None:
        state = {
            "awaiting_student_answer": True,
            "last_bot_question": "Quick answer or step-by-step explanation?",
            "last_tutor_move": "clarification_question",
        }
        self.assertEqual(
            allowed_turn_relations(state),
            ("clarification_choice", "new_question", "unclear"),
        )
        payload = build_turn_relation_payload("Step-by-step, please", state)
        self.assertEqual(
            payload["ALLOWED_LABELS"],
            ["clarification_choice", "new_question", "unclear"],
        )

    def test_safe_redirect_excludes_student_answer(self) -> None:
        state = {
            "awaiting_student_answer": True,
            "last_bot_question": "Would you like a safe conceptual explanation?",
            "last_tutor_move": "safe_redirect",
        }
        self.assertEqual(
            allowed_turn_relations(state),
            ("clarification_choice", "new_question", "unclear"),
        )
        self.assertEqual(turn_relation_context(state), "safe_redirect_selection")

    def test_rule_fallback_recognizes_new_question(self) -> None:
        relation = rule_based_turn_relation(
            "I will answer you later. Now I need to know what an atom is?",
            self.awaiting,
        )
        self.assertEqual(relation.relation, "new_question")

    def test_rule_fallback_recognizes_plain_new_question_after_feedback(self) -> None:
        relation = rule_based_turn_relation(
            "Please explain what a molecule means in terms of electrons.",
            {
                "awaiting_student_answer": True,
                "last_bot_question": "Can you explain why molecules form?",
                "last_tutor_move": "feedback_on_student_answer",
            },
        )
        self.assertEqual(relation.relation, "new_question")

    def test_rule_fallback_recognizes_clarification_choice(self) -> None:
        relation = rule_based_turn_relation(
            "Step-by-step explanation, please.",
            {
                "awaiting_student_answer": True,
                "last_bot_question": (
                    "Do you want a quick factual answer, a step-by-step explanation, "
                    "or help with a chemistry calculation or reaction concept?"
                ),
                "last_tutor_move": "clarification_question",
            },
        )
        self.assertEqual(relation.relation, "clarification_choice")

    def test_rule_fallback_recognizes_safe_redirect_choice(self) -> None:
        relation = rule_based_turn_relation(
            "Yes, I would like a safe conceptual explanation of the reaction.",
            {
                "awaiting_student_answer": True,
                "last_bot_question": (
                    "Would you like a safe conceptual explanation of the reaction, "
                    "the redox roles, or why the materials are hazardous?"
                ),
                "last_tutor_move": "safe_redirect",
            },
        )
        self.assertEqual(relation.relation, "clarification_choice")

    def test_clarification_choice_maps_to_theory(self) -> None:
        main_type, debug = resolve_clarification_choice_main_type(
            "step-by-step explanation please",
            last_tutor_move="clarification_question",
        )
        self.assertEqual(main_type, "theory_learning_open")
        self.assertTrue(debug["matched"])

    def test_safe_redirect_choice_maps_to_reaction_illustration(self) -> None:
        main_type, debug = resolve_clarification_choice_main_type(
            "yes i like a safe conceptual explanation of the reaction",
            last_tutor_move="safe_redirect",
        )
        self.assertEqual(main_type, "reaction_illustration_request")
        self.assertTrue(debug["matched"])

    def test_student_answer_overrides_content_label(self) -> None:
        main_type, changes, details = apply_turn_relation(
            "theory_learning_open",
            relation="student_answer",
            user_text="Because an ion has gained or lost electrons.",
            last_tutor_move="brief_answer_plus_check",
        )
        self.assertEqual(main_type, "student_answer_to_assess")
        self.assertEqual(details["effective_relation"], "student_answer")
        self.assertTrue(changes)

    def test_new_question_preserves_content_label(self) -> None:
        main_type, changes, details = apply_turn_relation(
            "reference_closed",
            relation="new_question",
            user_text="What is an atom?",
            last_tutor_move="brief_answer_plus_check",
        )
        self.assertEqual(main_type, "reference_closed")
        self.assertEqual(details["effective_relation"], "new_question")
        self.assertEqual(changes, [])

    def test_unresolved_clarification_choice_preserves_clear_content_label(self) -> None:
        main_type, changes, details = apply_turn_relation(
            "reference_closed",
            relation="clarification_choice",
            user_text="Is valence an outdated term?",
            last_tutor_move="clarification_question",
        )
        self.assertEqual(main_type, "reference_closed")
        self.assertEqual(details["effective_relation"], "new_question")
        self.assertTrue(changes)


if __name__ == "__main__":
    unittest.main()
