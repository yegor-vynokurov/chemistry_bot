"""Deterministic routing for the standalone chemistry tutor graph.

I0.3.1A removes model self-confidence from routing. Clarification is selected
for the explicit ``unclear`` label, not for a number invented by the classifier.
"""

from __future__ import annotations

from .schemas import QuestionClassification, RoutingDecision


def select_tutor_route(
    classification: QuestionClassification,
    *,
    confidence_threshold: float | None = None,
) -> RoutingDecision:
    """Map an operational classification to one route.

    ``confidence_threshold`` is accepted only for backward compatibility and is
    deliberately ignored.
    """
    del confidence_threshold

    if classification.safety_level == "unsafe":
        return RoutingDecision(
            route="safety_redirect",
            reason=(
                "The semantic safety gate detected both a hazardous target and "
                "procedural intent."
            ),
            classification_main_type=classification.main_type,
            safety_override=True,
        )

    if classification.main_type == "experiment_request_caution":
        return RoutingDecision(
            route="reaction_illustration",
            reason=(
                "The message asks for a practical chemistry procedure, but the "
                "safety gate did not mark it unsafe; keep the response conceptual."
            ),
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "unclear":
        return RoutingDecision(
            route="clarification",
            reason="The classifier emitted the explicit unclear label.",
            classification_main_type=classification.main_type,
            clarification_fallback=True,
        )

    if classification.main_type == "reference_closed":
        return RoutingDecision(
            route="reference",
            reason="Closed factual request: give a concise direct answer.",
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "theory_learning_open":
        route = "diagnostic" if classification.needs_diagnostic_question else "theory"
        return RoutingDecision(
            route=route,
            reason=(
                "Programmatic dialogue policy requested one diagnostic question."
                if route == "diagnostic"
                else "Conceptual learning request: explain the idea."
            ),
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "student_answer_to_assess":
        return RoutingDecision(
            route="student_answer",
            reason="The turn-relation classifier identified a student answer.",
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "reaction_illustration_request":
        return RoutingDecision(
            route="reaction_illustration",
            reason="Conceptual reaction illustration requested without a lab protocol.",
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "calculation_request":
        return RoutingDecision(
            route="calculation",
            reason="Calculation or equation task: use calculation guidance.",
            classification_main_type=classification.main_type,
        )

    if classification.main_type == "off_topic":
        return RoutingDecision(
            route="off_topic",
            reason="The message is outside the chemistry tutor scope.",
            classification_main_type=classification.main_type,
        )

    return RoutingDecision(
        route="clarification",
        reason="No explicit route matched; use a clarification fallback.",
        classification_main_type=classification.main_type,
        clarification_fallback=True,
    )
