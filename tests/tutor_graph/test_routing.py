"""Pure unit tests for score-free classification routing."""

from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.routing import select_tutor_route
from chemistry_bot.tutor_graph.schemas import QuestionClassification


class RoutingTests(unittest.TestCase):
    def classify(self, **overrides) -> QuestionClassification:
        data = {
            "main_type": "theory_learning_open",
            "needs_course_rag": True,
            "needs_user_documents": False,
            "needs_wikipedia": False,
            "needs_diagnostic_question": False,
            "safety_level": "safe",
            "reason": "derived test policy",
        }
        data.update(overrides)
        return QuestionClassification.model_validate(data)

    def route(self, **overrides) -> str:
        return select_tutor_route(self.classify(**overrides)).route

    def test_reference_route(self) -> None:
        self.assertEqual(self.route(main_type="reference_closed"), "reference")

    def test_theory_route(self) -> None:
        self.assertEqual(self.route(), "theory")

    def test_diagnostic_route(self) -> None:
        self.assertEqual(self.route(needs_diagnostic_question=True), "diagnostic")

    def test_student_answer_route(self) -> None:
        self.assertEqual(self.route(main_type="student_answer_to_assess"), "student_answer")

    def test_reaction_illustration_route(self) -> None:
        self.assertEqual(
            self.route(main_type="reaction_illustration_request", safety_level="caution"),
            "reaction_illustration",
        )

    def test_safe_procedure_request_stays_conceptual_without_hard_refusal(self) -> None:
        decision = select_tutor_route(
            self.classify(
                main_type="experiment_request_caution",
                safety_level="caution",
            )
        )
        self.assertEqual(decision.route, "reaction_illustration")
        self.assertFalse(decision.safety_override)

    def test_calculation_route(self) -> None:
        self.assertEqual(self.route(main_type="calculation_request"), "calculation")

    def test_off_topic_route(self) -> None:
        self.assertEqual(self.route(main_type="off_topic"), "off_topic")

    def test_unclear_label_routes_to_clarification(self) -> None:
        decision = select_tutor_route(self.classify(main_type="unclear"))
        self.assertEqual(decision.route, "clarification")
        self.assertTrue(decision.clarification_fallback)

    def test_unsafe_has_priority(self) -> None:
        decision = select_tutor_route(
            self.classify(main_type="experiment_request_caution", safety_level="unsafe")
        )
        self.assertEqual(decision.route, "safety_redirect")
        self.assertTrue(decision.safety_override)


if __name__ == "__main__":
    unittest.main()
