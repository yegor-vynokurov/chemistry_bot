"""Bridge AnythingLLM requests to the durable LangGraph chemistry tutor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from chemistry_bot.tutor_graph.decision_logging import build_decision_log_events
from chemistry_bot.tutor_graph.graph import TutorGraphConfig, build_tutor_graph
from chemistry_bot.tutor_graph.schemas import (
    QuestionClassification,
    RoutingDecision,
    TutorAnswer,
)
from chemistry_bot.tutor_graph.storage import SQLiteTutorStore

from .context_adapter import AdaptedRAGContext
from .conversation_resolver import (
    AnythingLLMConversationResolver,
    ConversationResolution,
    ParsedConversation,
)
from .routing_preview import RoutingPreview


DEFAULT_CONTEXT_TURNS = 12


@dataclass(frozen=True, slots=True)
class TutorBridgeResult:
    answer: TutorAnswer
    classification: QuestionClassification
    routing: RoutingDecision
    graph_debug: dict[str, Any]
    graph_error: str | None
    session_state_before: dict[str, Any]
    session_state_after: dict[str, Any]
    persisted_turn_ids: dict[str, Any]
    decision_log: dict[str, Any]
    external_context: dict[str, Any] | None

    @property
    def response_text(self) -> str:
        return self.answer.to_cli_text()

    def as_dict(self, *, include_context_text: bool = True) -> dict[str, Any]:
        external_context = self.external_context
        if isinstance(external_context, dict) and not include_context_text:
            external_context = dict(external_context)
            external_context.pop("prompt_text", None)
            chunks = external_context.get("chunks")
            if isinstance(chunks, list):
                external_context["chunks"] = [
                    {**chunk, "text": "[omitted]"} if isinstance(chunk, dict) else chunk
                    for chunk in chunks
                ]
        return {
            "answer": self.answer.model_dump(),
            "classification": self.classification.model_dump(),
            "routing": self.routing.model_dump(),
            "graph_debug": self.graph_debug,
            "graph_error": self.graph_error,
            "session_state_before": self.session_state_before,
            "session_state_after": self.session_state_after,
            "persisted_turn_ids": self.persisted_turn_ids,
            "decision_log": self.decision_log,
            "external_context": external_context,
        }


def _parsed_history_to_messages(parsed: ParsedConversation) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in parsed.history:
        if item.role == "student":
            messages.append(HumanMessage(content=item.text))
        elif item.role == "tutor":
            messages.append(AIMessage(content=item.text))
    return messages


def _derive_session_state_patch(
    answer: TutorAnswer,
    *,
    classification: QuestionClassification,
    routing: RoutingDecision,
    external_context: AdaptedRAGContext | None,
    context_turn_limit: int,
    loaded_history_turns: int,
) -> dict[str, Any]:
    next_question = answer.next_question.strip() if answer.next_question else None
    active_concept_id = answer.concept_ids[0] if answer.concept_ids else None
    return {
        "awaiting_student_answer": bool(next_question),
        "last_bot_question": next_question,
        "active_concept_id": active_concept_id,
        "last_tutor_move": answer.tutor_move,
        "last_concept_ids": answer.concept_ids,
        "last_route": routing.route,
        "last_route_reason": routing.reason,
        "last_question_classification": classification.main_type,
        "last_turn_relation": classification.turn_relation,
        "last_classification_safety_level": classification.safety_level,
        "last_needs_course_rag": classification.needs_course_rag,
        "last_needs_user_documents": classification.needs_user_documents,
        "last_needs_documents_dev": classification.needs_documents_dev,
        "last_needs_wikipedia": classification.needs_wikipedia,
        "last_needs_diagnostic_question": classification.needs_diagnostic_question,
        "last_context_source": (
            "anythingllm_document" if external_context is not None else "conversation_history"
        ),
        "last_context_turn_limit": context_turn_limit,
        "last_loaded_history_turns": loaded_history_turns,
        "last_anythingllm_rag_relevance": (
            external_context.relevance if external_context is not None else None
        ),
        "last_anythingllm_rag_score": (
            external_context.relevance_score if external_context is not None else None
        ),
        "last_anythingllm_sources": (
            list(external_context.source_documents) if external_context is not None else []
        ),
    }


class TutorGraphBridge:
    """Invoke the graph and persist the same audit trail as the CLI tutor."""

    def __init__(
        self,
        *,
        store: SQLiteTutorStore,
        conversation_resolver: AnythingLLMConversationResolver,
        graph_config: TutorGraphConfig,
        context_turns: int = DEFAULT_CONTEXT_TURNS,
    ) -> None:
        self.store = store
        self.conversation_resolver = conversation_resolver
        self.graph_config = graph_config
        self.context_turns = max(1, int(context_turns))
        self._graphs = {graph_config.model_name: build_tutor_graph(graph_config)}
        self._graph_lock = Lock()

    def _graph_for_model(self, model_name: str):
        """Return a compiled tutor graph for one validated model tag.

        Only ``TutorGraphConfig.model_name`` varies; classifier, turn relation,
        follow-up, safety and embedding roles remain fixed. A lock prevents
        duplicate graph construction when simultaneous requests select the
        same model.
        """
        with self._graph_lock:
            graph = self._graphs.get(model_name)
            if graph is None:
                model_config = replace(self.graph_config, model_name=model_name)
                graph = build_tutor_graph(model_config)
                self._graphs[model_name] = graph
            return graph

    def invoke(
        self,
        *,
        resolution: ConversationResolution,
        parsed: ParsedConversation,
        routing_preview: RoutingPreview,
        request_id: str,
        request_fingerprint: str,
        model_name: str | None = None,
        external_context: AdaptedRAGContext | None = None,
        interaction_id: str | None = None,
    ) -> TutorBridgeResult:
        """Run the selected tutor graph and persist its conversation audit.

        ``model_name`` must already be resolved against the active backend by
        the request handler. It affects only the tutor generation model; the
        shared session store and fixed support-model roles are unchanged.
        """
        selected_model = (model_name or self.graph_config.model_name).strip()
        if not selected_model:
            raise ValueError("The selected tutor model must be a non-empty tag.")
        session_state_before = self.store.get_session_state(resolution.session_id)
        history_messages = _parsed_history_to_messages(parsed)

        external_payload: dict[str, Any] | None = None
        if external_context is not None:
            external_payload = external_context.as_dict(include_text=True)
            external_payload["prompt_text"] = external_context.to_system_message_text()

        graph_state = {
            "student_id": resolution.student_id,
            "session_id": resolution.session_id,
            "user_text": parsed.current_user_text,
            "messages": history_messages,
            "session_state": session_state_before,
            "external_context": external_payload,
            "classification": routing_preview.classification,
            "routing": None,
            "response": None,
            "error": None,
            "debug": {
                **routing_preview.debug,
                "integration_stage": "I0.3.1B.4",
                "selected_tutor_model": selected_model,
                "anythingllm_interaction_id": interaction_id,
                "context_source": (
                    "anythingllm_document" if external_context is not None else "conversation_history"
                ),
                "history_turn_count": len(parsed.history),
                "external_context_relevance": (
                    external_context.relevance if external_context is not None else None
                ),
                "external_context_score": (
                    external_context.relevance_score if external_context is not None else None
                ),
            },
        }
        raw_result = self._graph_for_model(selected_model).invoke(graph_state)
        answer = raw_result.get("response")
        classification = raw_result.get("classification")
        routing = raw_result.get("routing")
        if not isinstance(answer, TutorAnswer):
            raise RuntimeError("LangGraph returned no valid TutorAnswer.")
        if not isinstance(classification, QuestionClassification):
            raise RuntimeError("LangGraph returned no valid QuestionClassification.")
        if not isinstance(routing, RoutingDecision):
            raise RuntimeError("LangGraph returned no valid RoutingDecision.")

        graph_debug = dict(raw_result.get("debug", {}))
        graph_error = raw_result.get("error")
        state_patch = _derive_session_state_patch(
            answer,
            classification=classification,
            routing=routing,
            external_context=external_context,
            context_turn_limit=self.context_turns,
            loaded_history_turns=len(parsed.history),
        )

        context_metadata = (
            external_context.as_dict(include_text=False)
            if external_context is not None
            else None
        )
        persisted_turn_ids = self.conversation_resolver.append_exchange(
            resolution=resolution,
            parsed=parsed,
            response_text=answer.to_cli_text(),
            request_id=request_id,
            request_fingerprint=request_fingerprint,
            integration_stage="I0.3.1B.4",
            extra_metadata={
                "anythingllm_interaction_id": interaction_id,
            },
            response_json=answer.model_dump(),
            tutor_metadata={
                "source": "langgraph_tutor",
                "classification": classification.model_dump(),
                "routing": routing.model_dump(),
                "graph_error": graph_error,
                "debug": graph_debug,
                "session_state_before": session_state_before,
                "session_state_patch": state_patch,
                "anythingllm_context": context_metadata,
            },
        )

        decision_ids: list[int] = []
        decision_types: list[str] = []
        decision_error: str | None = None
        try:
            for event in build_decision_log_events(
                classification=classification,
                routing=routing,
                debug=graph_debug,
            ):
                decision_ids.append(
                    self.store.save_decision(
                        student_id=resolution.student_id,
                        session_id=resolution.session_id,
                        node_name=event.node_name,
                        decision_type=event.decision_type,
                        decision=event.decision,
                        confidence=event.confidence,
                        reason=event.reason,
                        turn_id=int(persisted_turn_ids["student"]),
                    )
                )
                decision_types.append(event.decision_type)
        except Exception as error:  # Audit failure must not hide a tutor answer.
            decision_error = str(error)

        state_patch.update(
            {
                "last_student_turn_id": persisted_turn_ids.get("student"),
                "last_tutor_turn_id": persisted_turn_ids.get("tutor"),
            }
        )
        session_state_after = self.store.update_session_state(
            resolution.session_id,
            state_patch,
        )
        decision_log = {
            "ok": decision_error is None,
            "decision_ids": decision_ids,
            "decision_types": decision_types,
            "error": decision_error,
        }

        return TutorBridgeResult(
            answer=answer,
            classification=classification,
            routing=routing,
            graph_debug=graph_debug,
            graph_error=str(graph_error) if graph_error is not None else None,
            session_state_before=session_state_before,
            session_state_after=session_state_after,
            persisted_turn_ids=persisted_turn_ids,
            decision_log=decision_log,
            external_context=external_payload,
        )


__all__ = ["DEFAULT_CONTEXT_TURNS", "TutorBridgeResult", "TutorGraphBridge"]
