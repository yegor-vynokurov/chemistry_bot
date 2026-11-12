from __future__ import annotations

import unittest

from chemistry_bot.tutor_graph.decision_logging import build_decision_log_events
from chemistry_bot.tutor_graph.schemas import QuestionClassification, RoutingDecision


class DecisionLoggingTests(unittest.TestCase):
    def _classification(self, *, safety: str = "safe", main_type: str | None = None):
        return QuestionClassification(
            main_type=main_type or (
                "experiment_request_caution" if safety == "unsafe" else "theory_learning_open"
            ),
            needs_course_rag=safety != "unsafe",
            needs_user_documents=False,
            needs_wikipedia=False,
            needs_diagnostic_question=False,
            safety_level=safety,
            reason="main_type derived by test policy",
        )

    def test_normal_turn_logs_no_model_confidence(self) -> None:
        classification = self._classification()
        routing = RoutingDecision(
            route="theory",
            reason="Conceptual question.",
            classification_main_type=classification.main_type,
        )
        events = build_decision_log_events(
            classification=classification,
            routing=routing,
            debug={"classification_source": "llm_label"},
        )
        self.assertEqual([e.decision_type for e in events], ["classification", "routing"])
        self.assertTrue(all(e.confidence is None for e in events))
        self.assertEqual(events[0].decision["classifier_contract"], "main_type_only")

    def test_turn_relation_gets_its_own_row_without_confidence(self) -> None:
        classification = self._classification().model_copy(
            update={"turn_relation": "new_question"}
        )
        routing = RoutingDecision(
            route="theory",
            reason="Conceptual question.",
            classification_main_type=classification.main_type,
        )
        events = build_decision_log_events(
            classification=classification,
            routing=routing,
            debug={
                "turn_relation_source": "llm_label",
                "turn_relation_model_name": "phi4-mini-4k:latest",
                "turn_relation_raw_output": "new_question",
                "turn_relation_allowed_labels": ["student_answer", "new_question", "unclear"],
                "turn_relation_context": "answer_to_check_question",
                "turn_relation_state_constrained": True,
            },
        )
        self.assertEqual(events[-1].decision_type, "turn_relation")
        self.assertIsNone(events[-1].confidence)
        self.assertEqual(events[-1].decision["relation"], "new_question")
        self.assertEqual(
            events[-1].decision["allowed_labels"],
            ["student_answer", "new_question", "unclear"],
        )
        self.assertTrue(events[-1].decision["state_constrained"])

    def test_safety_override_gets_its_own_row(self) -> None:
        classification = self._classification(safety="unsafe")
        routing = RoutingDecision(
            route="safety_redirect",
            reason="Unsafe request.",
            classification_main_type=classification.main_type,
            safety_override=True,
        )
        events = build_decision_log_events(
            classification=classification,
            routing=routing,
            debug={"classification_derivation": {"changes": ["safety -> unsafe"]}},
        )
        self.assertEqual(
            [e.decision_type for e in events],
            ["classification", "routing", "safety_override"],
        )

    def test_unclear_label_logs_clarification_fallback(self) -> None:
        classification = self._classification(main_type="unclear")
        routing = RoutingDecision(
            route="clarification",
            reason="Unclear label.",
            classification_main_type="unclear",
            clarification_fallback=True,
        )
        events = build_decision_log_events(classification=classification, routing=routing)
        self.assertEqual(events[-1].decision_type, "clarification_fallback")
        self.assertEqual(events[-1].decision["trigger"], "explicit_unclear_label")


if __name__ == "__main__":
    unittest.main()
