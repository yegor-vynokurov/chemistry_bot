from __future__ import annotations

import unittest

from chemistry_bot.integrations.anythingllm_probe.rag_policy import should_call_user_document_rag
from chemistry_bot.tutor_graph.schemas import QuestionClassification, RoutingDecision


def classification(*, user_documents: bool, safety: str = "safe") -> QuestionClassification:
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


def route(name: str, *, safety_override: bool = False) -> RoutingDecision:
    return RoutingDecision(
        route=name,
        reason="test route",
        classification_main_type="theory_learning_open",
        safety_override=safety_override,
    )


class RoutingPreviewPolicyTests(unittest.TestCase):
    def test_classifier_policy_calls_rag_for_explicit_user_documents(self) -> None:
        result, _ = should_call_user_document_rag(
            classification=classification(user_documents=True),
            routing=route("theory"),
            rag_tool_available=True,
            policy="classifier",
        )
        self.assertTrue(result)

    def test_course_rag_flag_alone_does_not_open_user_book(self) -> None:
        result, _ = should_call_user_document_rag(
            classification=classification(user_documents=False),
            routing=route("theory"),
            rag_tool_available=True,
            policy="classifier",
        )
        self.assertFalse(result)


    def test_dev_override_flag_uses_same_protected_policy_gate(self) -> None:
        forced = classification(user_documents=True)
        forced = forced.model_copy(update={"needs_documents_dev": True})
        result, reason = should_call_user_document_rag(
            classification=forced,
            routing=route("theory"),
            rag_tool_available=True,
            policy="classifier",
        )
        self.assertTrue(result)
        self.assertIn("developer override", reason.lower())

    def test_safety_route_never_calls_user_documents(self) -> None:
        result, _ = should_call_user_document_rag(
            classification=classification(user_documents=True, safety="unsafe"),
            routing=route("safety_redirect", safety_override=True),
            rag_tool_available=True,
            policy="always",
        )
        self.assertFalse(result)

    def test_missing_tool_disables_rag(self) -> None:
        result, reason = should_call_user_document_rag(
            classification=classification(user_documents=True),
            routing=route("theory"),
            rag_tool_available=False,
            policy="classifier",
        )
        self.assertFalse(result)
        self.assertIn("did not expose", reason)


if __name__ == "__main__":
    unittest.main()
