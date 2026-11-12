"""Run the thin classifier before an AnythingLLM tool call.

The small model returns one label only. Python derives safety, source routing,
diagnostic behaviour, and audit reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from chemistry_bot.tutor_graph.graph import (
    DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_CLASSIFIER_PROMPT,
    _classifier_input_payload,
)
from chemistry_bot.tutor_graph.routing import select_tutor_route
from chemistry_bot.tutor_graph.semantic_safety import SemanticSafetyGate
from chemistry_bot.tutor_graph.schemas import QuestionClassification, RoutingDecision
from chemistry_bot.tutor_graph.turn_relation import (
    DEFAULT_TURN_RELATION_PROMPT,
    allowed_turn_relations,
    build_turn_relation_payload,
    parse_turn_relation_label,
    rule_based_turn_relation,
    should_classify_turn_relation,
    turn_relation_context,
    turn_relation_input_text,
)
from chemistry_bot.tutor_graph.thin_classifier import (
    derive_question_classification,
    parse_main_type_label,
    rule_based_main_type,
)

from .rag_policy import UserDocumentPolicy, should_call_user_document_rag


@dataclass(frozen=True, slots=True)
class RoutingPreview:
    classification: QuestionClassification
    routing: RoutingDecision
    debug: dict[str, Any]
    should_call_user_documents: bool
    user_document_policy: UserDocumentPolicy
    reason: str
    needs_documents_dev: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.model_dump(),
            "routing": self.routing.model_dump(),
            "debug": self.debug,
            "should_call_user_documents": self.should_call_user_documents,
            "user_document_policy": self.user_document_policy,
            "reason": self.reason,
            "needs_documents_dev": self.needs_documents_dev,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoutingPreview":
        policy = str(value.get("user_document_policy", "classifier"))
        if policy not in {"classifier", "always", "never"}:
            policy = "classifier"
        return cls(
            classification=QuestionClassification.model_validate(value["classification"]),
            routing=RoutingDecision.model_validate(value["routing"]),
            debug=dict(value.get("debug", {})),
            should_call_user_documents=bool(value.get("should_call_user_documents")),
            user_document_policy=policy,  # type: ignore[arg-type]
            reason=str(value.get("reason", "Restored from interaction log.")),
            needs_documents_dev=bool(value.get("needs_documents_dev", False)),
        )


class AnythingLLMRoutingPreview:
    """Reusable one-label classifier/router for the API bridge."""

    def __init__(
        self,
        *,
        classifier_enabled: bool = True,
        classifier_model_name: str = DEFAULT_CLASSIFIER_MODEL,
        classifier_temperature: float = 0.0,
        classifier_prompt: str = DEFAULT_CLASSIFIER_PROMPT,
        confidence_threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        fallback_to_rules: bool = True,
        turn_relation_enabled: bool = True,
        turn_relation_model_name: str = DEFAULT_CLASSIFIER_MODEL,
        turn_relation_temperature: float = 0.0,
        turn_relation_num_ctx: int = 2048,
        turn_relation_prompt: str = DEFAULT_TURN_RELATION_PROMPT,
        turn_relation_fallback_to_rules: bool = True,
        semantic_safety_enabled: bool = True,
        safety_embedding_model_name: str = "nomic-embed-text",
        safety_hazard_threshold: float = 0.60,
        safety_procedure_threshold: float = 0.60,
        safety_fuzzy_threshold: float = 86.0,
        user_document_policy: UserDocumentPolicy = "classifier",
        needs_documents_dev: bool = False,
    ) -> None:
        if user_document_policy not in {"classifier", "always", "never"}:
            raise ValueError(f"Unsupported user_document_policy: {user_document_policy!r}")
        self.classifier_enabled = classifier_enabled
        self.classifier_model_name = classifier_model_name
        self.classifier_temperature = classifier_temperature
        self.classifier_prompt = classifier_prompt
        self.deprecated_confidence_threshold = confidence_threshold
        self.fallback_to_rules = fallback_to_rules
        self.turn_relation_enabled = bool(turn_relation_enabled)
        self.turn_relation_model_name = turn_relation_model_name
        self.turn_relation_temperature = turn_relation_temperature
        self.turn_relation_num_ctx = max(512, int(turn_relation_num_ctx))
        self.turn_relation_prompt = turn_relation_prompt
        self.turn_relation_fallback_to_rules = turn_relation_fallback_to_rules
        self.user_document_policy = user_document_policy
        self.needs_documents_dev = bool(needs_documents_dev)
        self._classifier_model = (
            ChatOllama(model=classifier_model_name, temperature=classifier_temperature)
            if classifier_enabled
            else None
        )
        self._turn_relation_model = (
            ChatOllama(
                model=turn_relation_model_name,
                temperature=turn_relation_temperature,
                num_ctx=self.turn_relation_num_ctx,
                num_predict=16,
            )
            if turn_relation_enabled
            else None
        )
        self._safety_gate = SemanticSafetyGate(
            enabled=semantic_safety_enabled,
            embedding_model_name=safety_embedding_model_name,
            hazard_threshold=safety_hazard_threshold,
            procedure_threshold=safety_procedure_threshold,
            fuzzy_threshold=safety_fuzzy_threshold,
        )

    def classify(
        self,
        *,
        user_text: str,
        session_state: dict[str, Any] | None = None,
        rag_tool_available: bool,
    ) -> RoutingPreview:
        state = dict(session_state or {})
        payload = _classifier_input_payload(user_text, {})
        source = "rules_disabled"
        error_text: str | None = None
        raw_output: str | None = None

        if self._classifier_model is None:
            thin = rule_based_main_type(user_text)
        else:
            try:
                response = self._classifier_model.invoke(
                    [
                        SystemMessage(content=self.classifier_prompt),
                        HumanMessage(
                            content=(
                                "Return one chemistry-content label for this isolated current turn.\n"
                                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
                            )
                        ),
                    ]
                )
                raw_output = str(response.content)
                thin = parse_main_type_label(raw_output)
                source = "llm_label"
            except Exception as error:
                if not self.fallback_to_rules:
                    raise
                error_text = str(error)
                source = "rules_after_classifier_error"
                thin = rule_based_main_type(user_text)

        relation = None
        relation_source = "not_needed"
        relation_error: str | None = None
        relation_raw_output: str | None = None
        relation_payload: dict[str, Any] | None = None
        relation_allowed_labels = None
        relation_context = None
        if should_classify_turn_relation(state):
            relation_allowed_labels = allowed_turn_relations(state)
            relation_context = turn_relation_context(state)
            relation_payload = build_turn_relation_payload(user_text, state)
            if self._turn_relation_model is None:
                relation = rule_based_turn_relation(user_text, state).relation
                relation_source = "rules_disabled"
            else:
                try:
                    response = self._turn_relation_model.invoke(
                        [
                            SystemMessage(content=self.turn_relation_prompt),
                            HumanMessage(content=turn_relation_input_text(user_text, state)),
                        ]
                    )
                    relation_raw_output = str(response.content)
                    relation = parse_turn_relation_label(
                        relation_raw_output,
                        allowed_labels=relation_allowed_labels,
                    ).relation
                    relation_source = "llm_label"
                except Exception as error:
                    if not self.turn_relation_fallback_to_rules:
                        raise
                    relation_error = str(error)
                    relation = rule_based_turn_relation(user_text, state).relation
                    relation_source = "rules_after_classifier_error"

        safety_assessment = self._safety_gate.evaluate(user_text)
        classification, derivation = derive_question_classification(
            thin,
            user_text=user_text,
            source=source,
            needs_documents_dev=self.needs_documents_dev,
            turn_relation=relation,
            session_state=state,
            safety_assessment=safety_assessment,
        )
        routing = select_tutor_route(classification)
        debug = {
            "classification_node": "anythingllm_routing_preview",
            "classifier_contract": "main_type_only",
            "classification_source": source,
            "classifier_model_name": self.classifier_model_name if self.classifier_enabled else None,
            "classification_error": error_text,
            "classifier_raw_output": raw_output,
            "classifier_input_payload": payload,
            "turn_relation_contract": "one_state_allowed_label_when_reply_awaited",
            "turn_relation_source": relation_source,
            "turn_relation_model_name": (
                self.turn_relation_model_name if relation_payload is not None else None
            ),
            "turn_relation_num_ctx": (
                self.turn_relation_num_ctx if relation_payload is not None else None
            ),
            "turn_relation_error": relation_error,
            "turn_relation_raw_output": relation_raw_output,
            "turn_relation_input_payload": relation_payload,
            "turn_relation_allowed_labels": (
                list(relation_allowed_labels) if relation_allowed_labels is not None else None
            ),
            "turn_relation_context": relation_context,
            "turn_relation_state_constrained": relation_payload is not None,
            "semantic_safety_gate": safety_assessment.as_dict(),
            "classification_derivation": derivation,
            "needs_documents_dev": self.needs_documents_dev,
            "routing_node": "anythingllm_routing_preview",
            "selected_route": routing.route,
            "routing_reason": routing.reason,
            "routing_clarification_fallback": routing.clarification_fallback,
            "routing_safety_override": routing.safety_override,
        }

        should_call, reason = should_call_user_document_rag(
            classification=classification,
            routing=routing,
            rag_tool_available=rag_tool_available,
            policy=self.user_document_policy,
        )
        return RoutingPreview(
            classification=classification,
            routing=routing,
            debug=debug,
            should_call_user_documents=should_call,
            user_document_policy=self.user_document_policy,
            reason=reason,
            needs_documents_dev=self.needs_documents_dev,
        )


__all__ = [
    "AnythingLLMRoutingPreview",
    "RoutingPreview",
    "UserDocumentPolicy",
    "should_call_user_document_rag",
]
