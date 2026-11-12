"""Pure helpers for building durable tutor decision-log events.

I0.3.1A records deterministic signals instead of model-generated confidence.
The SQLite confidence column is left nullable for schema compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import QuestionClassification, RoutingDecision


@dataclass(frozen=True, slots=True)
class DecisionLogEvent:
    node_name: str
    decision_type: str
    decision: dict[str, Any]
    confidence: float | None
    reason: str | None


def build_decision_log_events(
    *,
    classification: QuestionClassification,
    routing: RoutingDecision,
    debug: dict[str, Any] | None = None,
) -> list[DecisionLogEvent]:
    debug = dict(debug or {})
    derivation = debug.get("classification_derivation")
    if not isinstance(derivation, dict):
        derivation = debug.get("classification_calibration")
    if not isinstance(derivation, dict):
        derivation = {}

    classification_payload = {
        "classification": classification.model_dump(),
        "classifier_contract": "main_type_only",
        "classifier_source": debug.get("classification_source"),
        "classifier_model_name": debug.get("classifier_model_name"),
        "classifier_error": debug.get("classification_error"),
        "classifier_raw_output": debug.get("classifier_raw_output"),
        "turn_relation": classification.turn_relation,
        "turn_relation_source": debug.get("turn_relation_source"),
        "turn_relation_model_name": debug.get("turn_relation_model_name"),
        "turn_relation_num_ctx": debug.get("turn_relation_num_ctx"),
        "turn_relation_error": debug.get("turn_relation_error"),
        "turn_relation_raw_output": debug.get("turn_relation_raw_output"),
        "turn_relation_allowed_labels": debug.get("turn_relation_allowed_labels"),
        "turn_relation_context": debug.get("turn_relation_context"),
        "turn_relation_state_constrained": debug.get("turn_relation_state_constrained"),
        "semantic_safety_gate": debug.get("semantic_safety_gate"),
        "derivation_changes": derivation.get("changes", []),
        "derivation_signals": derivation.get("signals", {}),
    }
    events = [
        DecisionLogEvent(
            node_name="classify_question",
            decision_type="classification",
            decision=classification_payload,
            confidence=None,
            reason=classification.reason,
        ),
        DecisionLogEvent(
            node_name="select_route",
            decision_type="routing",
            decision={
                "routing": routing.model_dump(),
                "classification_main_type": classification.main_type,
                "classification_safety_level": classification.safety_level,
                "needs_course_rag": classification.needs_course_rag,
                "needs_user_documents": classification.needs_user_documents,
                "needs_wikipedia": classification.needs_wikipedia,
            },
            confidence=None,
            reason=routing.reason,
        ),
    ]

    if classification.turn_relation is not None:
        events.append(
            DecisionLogEvent(
                node_name="classify_turn_relation",
                decision_type="turn_relation",
                decision={
                    "relation": classification.turn_relation,
                    "source": debug.get("turn_relation_source"),
                    "model_name": debug.get("turn_relation_model_name"),
                    "num_ctx": debug.get("turn_relation_num_ctx"),
                    "error": debug.get("turn_relation_error"),
                    "raw_output": debug.get("turn_relation_raw_output"),
                    "allowed_labels": debug.get("turn_relation_allowed_labels"),
                    "relation_context": debug.get("turn_relation_context"),
                    "state_constrained": debug.get("turn_relation_state_constrained"),
                },
                confidence=None,
                reason="State-constrained one-label classification for the awaited tutor reply.",
            )
        )

    if routing.safety_override:
        events.append(
            DecisionLogEvent(
                node_name="select_route",
                decision_type="safety_override",
                decision={
                    "route": routing.route,
                    "main_type": classification.main_type,
                    "safety_level": classification.safety_level,
                    "derivation_changes": derivation.get("changes", []),
                    "semantic_safety_gate": debug.get("semantic_safety_gate"),
                },
                confidence=None,
                reason="Deterministic safety policy overrode normal answer generation.",
            )
        )

    if routing.clarification_fallback and routing.route == "clarification":
        events.append(
            DecisionLogEvent(
                node_name="select_route",
                decision_type="clarification_fallback",
                decision={
                    "route": routing.route,
                    "main_type": classification.main_type,
                    "trigger": "explicit_unclear_label",
                },
                confidence=None,
                reason="The explicit unclear label triggered a clarification question.",
            )
        )

    return events
