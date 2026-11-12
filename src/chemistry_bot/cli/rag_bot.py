"""Main RAG-enabled CLI bot for the current Chemistry Bot baseline.

This module owns the student-facing chat loop, structured answer schema,
Prompt Garden integration, bot-side textbook retrieval orchestration,
controlled Wikipedia enrichment, and prompt-debug logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Any
import logging

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


from pydantic import BaseModel, Field, model_validator

from chemistry_bot.promptops.garden import PromptGarden
from chemistry_bot.retrievers.wiki_context import (
    WikipediaContextBuilder,
    WikiContextConfig,
)
from chemistry_bot.retrieval.introchem_rag import (
    ChromaTheoryRAG,
    RAGConfig,
    RAGHit,
    RAG_POLICY_SYSTEM_TEMPLATE,
    RAG_REFERENCE_USER_TEMPLATE,
)

import json


LOG_PATH = Path("logs") / "chat_session.log"
PROMPT_DEBUG_LOG_PATH = Path("logs") / "prompt_debug.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


WIKI_REFERENCE_USER_TEMPLATE = """
Wikipedia reference context:
{wiki_context}

Wikipedia context usage rules:
- Treat Wikipedia only as extra background for definitions, history, and reaction illustrations.
- Textbook RAG context has priority over Wikipedia.
- Do not turn Wikipedia snippets into a lab protocol, synthesis method, recipe, or home experiment.
- Do not add exact masses, volumes, concentrations, temperatures, heating times, apparatus setup, or step-by-step operations.
- If Wikipedia and textbook RAG disagree, prefer the textbook RAG context and mention uncertainty.
- source_ids must include only textbook RAG source labels such as S1 or S2, never Wiki labels.
"""


logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)


class ExperimentIngredient(BaseModel):
    name: str = Field(min_length=1)
    formula: str | None = None
    amount: str = Field(min_length=1)
    is_solution: bool
    concentration: str | None = None
    purpose: str = Field(min_length=1)

    @model_validator(mode="after")
    def solution_requires_concentration(self):
        if self.is_solution and not self.concentration:
            raise ValueError("A solution ingredient must include its concentration.")
        return self


class NoExperiment(BaseModel):
    kind: Literal["none"]
    reason: str = Field(min_length=1)


class ClarificationRequest(BaseModel):
    kind: Literal["clarification"]
    reason: str = Field(min_length=1)
    questions: list[str] = Field(min_length=1, max_length=3)


class ExperimentPlan(BaseModel):
    kind: Literal["experiment"]
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    ingredients: list[ExperimentIngredient] = Field(min_length=1)
    equipment: list[str] = Field(min_length=1)
    protective_equipment: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    safety_notes: list[str] = Field(min_length=1)
    disposal: str = Field(min_length=1)
    adult_supervision_required: bool


ExperimentChoice = Annotated[
    NoExperiment | ClarificationRequest | ExperimentPlan,
    Field(discriminator="kind"),
]


class AnswerProfile(BaseModel):
    request_type: Literal["theory", "experiment", "mixed"]
    short_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list, max_length=2)
    experiment: ExperimentChoice
    certainty: Literal["high", "medium", "low"]
    source_ids: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Labels of retrieved sources actually used, "
            "for example ['S1', 'S3']. Empty when no source was used."
        ),
    )


@dataclass
class StudentContext:
    """Student-profile variables materialized into the active prompt combo."""

    language: str = "English"
    learner_profile: str = "9th grade student"
    learning_preferences: str = "interesting but factually precise explanation"
    school_context: str = "good private school"
    learning_goal: str = "understand chemistry concepts safely"
    constraints: str = "avoid unnecessary complexity; stay respectful and concrete"
    protocol_context: str = "No verified experimental protocol was retrieved."
    notes: str = ""

    def to_prompt_dict(self) -> dict[str, str]:
        return asdict(self)

    def legacy_variables(self) -> dict[str, str]:
        return {
            "language": self.language,
            "school_class": self.learner_profile,
            "explanation_level": self.learning_preferences,
            "type_of_school": self.school_context,
            "protocol_context": self.protocol_context,
            "notes": self.notes,
        }


@dataclass
class TeacherContext:
    """Teacher-profile variables that shape explanation style and safety tone."""

    teacher_profile: str = (
        "an experienced chemistry teacher with university-level knowledge"
    )

    personality_traits: str = (
        "patient, intellectually curious, calm, attentive, and honest"
    )

    tone: str = (
        "warm, clear, respectful, and encouraging without sounding childish"
    )

    teaching_style: str = (
        "explain from simple ideas to more complex ones; "
        "use short analogies and concrete chemistry examples"
    )

    correction_style: str = (
        "correct false assumptions directly but gently; "
        "explain what was wrong and provide the accurate model"
    )

    language_style: str = (
        "use plain English suitable for a ninth-grade student; "
        "avoid slang, excessive enthusiasm, and unnecessary jargon"
    )

    safety_stance: str = (
        "be strictly cautious with experiments; "
        "never invent quantities, concentrations, risks, or disposal rules"
    )

    response_preferences: str = (
        "prioritize scientific accuracy over entertainment; "
        "state uncertainty plainly"
    )

    notes: str = ""

    def to_prompt_dict(self) -> dict[str, str]:
        return asdict(self)

    def legacy_variables(self) -> dict[str, str]:
        return {
            "level_of_teacher": self.teacher_profile,
            "teacher_personality": self.personality_traits,
            "teacher_tone": self.tone,
            "personality_of_teacher": self.teaching_style,
            "correction_style": self.correction_style,
            "language_style": self.language_style,
            "safety_stance": self.safety_stance,
            "response_preferences": self.response_preferences,
            "teacher_notes": self.notes,
        }


class CliBot:
    """Interactive chemistry bot with Prompt Garden and optional RAG support."""

    def __init__(
        self,
        model_name: str,
        garden_root: str | Path = "prompt_garden",
        combo_id: str = "combo_000001",
        fewshot_id: str | None = None,
        max_history_messages: int = 12,
        materialize_context: bool = True,
        rag_config: RAGConfig | None = None,
        enable_wikipedia_context: bool = True,
        wikipedia_lang: str = "en",
        wikipedia_top_k_results: int = 2,
        wikipedia_doc_chars: int = 1600,
        show_prompt: bool = False,
        log_prompts: bool = True,
    ) -> None:
        self.model_name = model_name.strip()
        self.base_combo_id = combo_id
        self.combo_id = combo_id

        self.fewshot_id = fewshot_id
        self.selected_fewshot_example_ids: list[str] = []

        self.materialize_context = materialize_context

        self.garden = PromptGarden(garden_root)
        self.garden.init()

        self.chat_model = ChatOllama(model=self.model_name, temperature=0)
        self.structured_model = self.chat_model.with_structured_output(
            AnswerProfile,
            method="json_schema",
            include_raw=True,
        )

        self.store: dict[str, InMemoryChatMessageHistory] = {}
        self.max_history_messages = max_history_messages

        self.student_context = StudentContext()
        self.teacher_context = TeacherContext()
        self.last_input_data: dict[str, Any] | None = None
        self.last_raw_response_text: str | None = None
        self.last_parsed_answer: dict[str, Any] | None = None
        self.last_parsing_error: str | None = None
        self.last_invoke_error: str | None = None

        self.rag = ChromaTheoryRAG(
            rag_config or RAGConfig(enabled=False)
        )
        self.last_rag_hits: list[RAGHit] = []
        self.last_rag_error: str | None = None

        self.wikipedia = WikipediaContextBuilder(
            WikiContextConfig(
                enabled=enable_wikipedia_context,
                lang=wikipedia_lang,
                top_k_results=wikipedia_top_k_results,
                doc_content_chars_max=wikipedia_doc_chars,
            )
        )
        self.force_wiki_next_turn = False
        self.last_wiki_context: str = ""
        self.last_wiki_reason: str | None = None

        self.show_prompt = show_prompt
        self.log_prompts = log_prompts
        self.prompt_debug_log_path = PROMPT_DEBUG_LOG_PATH
        self.last_prompt_debug: dict[str, Any] | None = None

        self.combined_system_prompt: str = ""
        self.combined_user_prompt: str = ""
        self.prompt: ChatPromptTemplate | None = None
        self.chain = None

        if materialize_context:
            self.configure_contexts(
                context=self.student_context,
                teacher_context=self.teacher_context,
            )
        else:
            self._load_chain_from_combo(self.combo_id)

    def configure_contexts(
        self,
        context: StudentContext,
        teacher_context: TeacherContext,
    ) -> None:
        self.student_context = context
        self.teacher_context = teacher_context

        contextual_combo = self.garden.get_or_create_context_combo(
            base_combo_id=self.base_combo_id,
            student_context=context.to_prompt_dict(),
            teacher_context=teacher_context.to_prompt_dict(),
            runtime_placeholders=(
                "question",
                "rag_context",
                "local_context",
                "wiki_context",
            ),
        )

        self.combo_id = contextual_combo["id"]
        self._load_chain_from_combo(self.combo_id)

    def refresh_context_combo(self) -> None:
        self.configure_contexts(
            context=self.student_context,
            teacher_context=self.teacher_context,
        )
        print(f"Бот: контекст обновлён. Активная combo: {self.combo_id}")


    def _load_fewshot_messages(
        self,
    ) -> list[HumanMessage | AIMessage]:
        if self.fewshot_id is None:
            self.selected_fewshot_example_ids = []
            return []

        node = self.garden.get_node(self.fewshot_id)

        if node["type"] != "fewshot":
            raise ValueError(
                f"Prompt {self.fewshot_id!r} has type "
                f"{node['type']!r}, expected 'fewshot'."
            )

        fewshot_text = self.garden.read_prompt(
            self.fewshot_id
        )

        return self._build_fewshot_messages(
            fewshot_text
        )


    def _load_chain_from_combo(
        self,
        combo_id: str,
    ) -> None:
        combo_prompts = self.garden.read_combo_prompts(
            combo_id
        )

        if (
            "system" not in combo_prompts
            or "user" not in combo_prompts
        ):
            raise ValueError(
                "Combo must include at least "
                "'system' and 'user' prompts."
            )

        self.system_prompt = combo_prompts["system"]
        self.user_prompt = combo_prompts["user"]

        # Для few-shot примеров используем исходный user prompt,
        # а не его contextual/materialized вариант.
        base_combo_prompts = self.garden.read_combo_prompts(
            self.base_combo_id
        )

        if "user" not in base_combo_prompts:
            raise ValueError(
                "Base combo must include a user prompt."
            )

        self.example_user_prompt = base_combo_prompts["user"]

        fewshot_messages = self._load_fewshot_messages()

        self.combined_system_prompt = (
            self.system_prompt
            + "\n\n"
            + RAG_POLICY_SYSTEM_TEMPLATE
        )
        self.combined_user_prompt = (
            self.user_prompt
            + "\n\n"
            + RAG_REFERENCE_USER_TEMPLATE
            + "\n\n"
            + WIKI_REFERENCE_USER_TEMPLATE
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.combined_system_prompt),
                *fewshot_messages,
                MessagesPlaceholder(
                    variable_name="history"
                ),
                ("human", self.combined_user_prompt),
            ]
        )

        self.chain = self.prompt | self.structured_model


    def _build_prompt_variables(
        self,
        question: str,
        overrides: dict | None = None,
    ) -> dict:
        values = {
            "question": question,
            "rag_context": (
                "RAG context has not been prepared for this turn."
            ),
            "local_context": (
                "Local textbook context has not been prepared for this turn."
            ),
            "wiki_context": (
                "No Wikipedia reference context was retrieved for this turn."
            ),
            **self.student_context.legacy_variables(),
            **self.teacher_context.legacy_variables(),
        }

        if overrides:
            values.update(overrides)

        return values
    
    def _render_example_human_message(
        self,
        example_input: dict,
    ) -> str:
        overrides = {
            key: value
            for key, value in example_input.items()
            if key != "question"
        }

        variables = self._build_prompt_variables(
            question=example_input["question"],
            overrides=overrides,
        )

        return self.example_user_prompt.format(
            **variables
        )
    

    def _build_fewshot_messages(
        self,
        fewshot_text: str,
    ) -> list[HumanMessage | AIMessage]:
        try:
            examples = json.loads(fewshot_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Few-shot prompt {self.fewshot_id!r} "
                "must contain valid JSON."
            ) from error

        if not isinstance(examples, list):
            raise ValueError(
                "Few-shot prompt must contain "
                "a JSON list of examples."
            )

        if not examples:
            raise ValueError(
                "Few-shot prompt contains no examples."
            )

        messages: list[HumanMessage | AIMessage] = []
        self.selected_fewshot_example_ids = []

        for index, example in enumerate(
            examples,
            start=1,
        ):
            if not isinstance(example, dict):
                raise ValueError(
                    f"Few-shot example #{index} "
                    "must be a JSON object."
                )

            if "input" not in example:
                raise ValueError(
                    f"Few-shot example #{index} "
                    "has no 'input' field."
                )

            if "answer" not in example:
                raise ValueError(
                    f"Few-shot example #{index} "
                    "has no 'answer' field."
                )

            example_input = example["input"]

            if "question" not in example_input:
                raise ValueError(
                    f"Few-shot example #{index} "
                    "has no input.question."
                )

            validated_answer = AnswerProfile.model_validate(
                example["answer"]
            )

            human_content = (
                self._render_example_human_message(
                    example_input
                )
            )

            assistant_content = (
                validated_answer.model_dump_json(
                    indent=2
                )
            )

            messages.append(
                HumanMessage(content=human_content)
            )

            messages.append(
                AIMessage(content=assistant_content)
            )

            example_id = example.get(
                "id",
                f"example_{index}",
            )

            self.selected_fewshot_example_ids.append(
                example_id
            )

        return messages

    
    def _retrieve_rag_context(
        self,
        user_text: str,
    ) -> tuple[str, list[RAGHit]]:
        self.last_rag_error = None

        if not self.rag.enabled:
            self.last_rag_hits = []
            return self.rag.format_context([]), []

        try:
            rag_context, hits = self.rag.retrieve_context(
                user_text
            )
            self.last_rag_hits = hits
            return rag_context, hits
        except Exception as error:
            self.last_rag_hits = []
            self.last_rag_error = str(error)
            logging.exception("RAG retrieval error")

            if not self.rag.config.fail_open:
                raise

            rag_context = (
                "RAG retrieval failed for this turn. "
                "No textbook passages are available. "
                "Use general knowledge cautiously, state uncertainty, "
                "and return an empty source_ids list. "
                f"Technical error: {error}"
            )
            return rag_context, []


    def _retrieve_wiki_context(
        self,
        user_text: str,
    ) -> tuple[str, str]:
        """Return Wikipedia context for this turn plus a short reason."""

        force = self.force_wiki_next_turn
        self.force_wiki_next_turn = False

        allowed, reason = self.wikipedia.should_retrieve(
            user_text,
            force=force,
        )
        self.last_wiki_reason = reason

        if not allowed:
            context = (
                "No Wikipedia reference context was retrieved for this turn. "
                f"Reason: {reason}"
            )
            self.last_wiki_context = context
            return context, reason

        context = self.wikipedia.build(
            user_text,
            force=force,
        ).strip()

        if not context:
            context = (
                "Wikipedia reference context was requested, but no usable "
                "article snippets were returned. "
                f"Reason: {reason}"
            )

        self.last_wiki_context = context
        return context, reason

    def print_wiki_status(self) -> None:
        print("\nНастройки Wikipedia:")
        print(f"  {self.wikipedia.status()}")
        print(f"  last_reason: {self.last_wiki_reason}")
        print(f"  force_next_turn: {self.force_wiki_next_turn}")
        print()

    def preview_wiki_search(self, query: str) -> None:
        if not query.strip():
            print("Бот: после /wiki-search нужен поисковый запрос.")
            return

        context, reason = self._retrieve_wiki_context(query)
        print("\nПредпросмотр Wikipedia:")
        print(f"reason: {reason}")
        print(context)
        print()

    def _format_prompt_messages(
        self,
        input_data: dict[str, Any],
        session_id: str,
    ) -> str:
        if self.prompt is None:
            return "Prompt is not configured."

        try:
            history_messages = list(
                self.get_session_history(session_id).messages
            )
            rendered_messages = self.prompt.format_messages(
                **input_data,
                history=history_messages,
            )
        except Exception as error:  # noqa: BLE001 - debug should not break chat
            logging.exception("Prompt rendering for debug failed")
            return (
                "Prompt rendering failed. "
                f"Technical error: {error}\n\n"
                "System template:\n"
                f"{self.combined_system_prompt}\n\n"
                "User template:\n"
                f"{self.combined_user_prompt}"
            )

        parts: list[str] = []
        for index, message in enumerate(rendered_messages, start=1):
            role = getattr(message, "type", message.__class__.__name__)
            content = str(getattr(message, "content", ""))
            parts.append(
                f"--- {index}. {role.upper()} MESSAGE ---\n{content}"
            )

        return "\n\n".join(parts)

    def _build_prompt_debug(
        self,
        *,
        question: str,
        session_id: str,
        input_data: dict[str, Any],
        rag_hits: list[RAGHit],
        wiki_reason: str,
    ) -> dict[str, Any]:
        prompt_body = self._format_prompt_messages(
            input_data=input_data,
            session_id=session_id,
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "model": self.model_name,
            "combo_id": self.combo_id,
            "base_combo_id": self.base_combo_id,
            "fewshot_id": self.fewshot_id,
            "question": question,
            "prompt_body": prompt_body,
            "system_prompt_body": self.combined_system_prompt,
            "user_prompt_body": self.combined_user_prompt,
            "rag_context": input_data.get("rag_context", ""),
            "local_context": input_data.get("local_context", ""),
            "wiki_context": input_data.get("wiki_context", ""),
            "protocol_context": input_data.get("protocol_context", ""),
            "rag": {
                "enabled": self.rag.enabled,
                "error": self.last_rag_error,
                "sources": [
                    hit.to_log_dict()
                    for hit in rag_hits
                ],
            },
            "wikipedia": {
                "status": self.wikipedia.status(),
                "reason": wiki_reason,
                "force_next_turn_after_this_call": self.force_wiki_next_turn,
            },
            "prompt_uses_rag_context_placeholder": (
                "{rag_context}" in self.combined_user_prompt
                or "{rag_context}" in self.combined_system_prompt
            ),
            "prompt_uses_local_context_placeholder": (
                "{local_context}" in self.combined_user_prompt
                or "{local_context}" in self.combined_system_prompt
            ),
            "prompt_uses_wiki_context_placeholder": (
                "{wiki_context}" in self.combined_user_prompt
                or "{wiki_context}" in self.combined_system_prompt
            ),
        }

    def _write_prompt_debug(self, prompt_debug: dict[str, Any]) -> None:
        try:
            self.prompt_debug_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with self.prompt_debug_log_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        prompt_debug,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:  # noqa: BLE001 - logging must not break chat
            logging.exception("Could not write prompt debug log")

    def _answer_payload_for_console(
        self,
        answer: AnswerProfile,
        prompt_debug: dict[str, Any],
    ) -> dict[str, Any]:
        payload = answer.model_dump()
        if self.show_prompt:
            payload["_debug"] = {
                "prompt_body": prompt_debug.get("prompt_body", ""),
                "rag_context": prompt_debug.get("rag_context", ""),
                "local_context": prompt_debug.get("local_context", ""),
                "wiki_context": prompt_debug.get("wiki_context", ""),
                "protocol_context": prompt_debug.get("protocol_context", ""),
                "rag": prompt_debug.get("rag", {}),
                "wikipedia": prompt_debug.get("wikipedia", {}),
                "prompt_uses_rag_context_placeholder": prompt_debug.get(
                    "prompt_uses_rag_context_placeholder"
                ),
                "prompt_uses_local_context_placeholder": prompt_debug.get(
                    "prompt_uses_local_context_placeholder"
                ),
                "prompt_uses_wiki_context_placeholder": prompt_debug.get(
                    "prompt_uses_wiki_context_placeholder"
                ),
            }
        return payload

    def print_last_prompt_debug(self) -> None:
        if not self.last_prompt_debug:
            print("Бот: prompt debug ещё не создавался.")
            return

        print("\nПоследний prompt debug:")
        print(
            json.dumps(
                self.last_prompt_debug,
                indent=2,
                ensure_ascii=False,
            )
        )
        print()

    def _validate_answer_source_ids(
        self,
        answer: AnswerProfile,
        hits: list[RAGHit],
    ) -> list[str]:
        allowed = {hit.label for hit in hits}
        valid: list[str] = []
        invalid: list[str] = []

        for source_id in answer.source_ids:
            normalized = source_id.strip().upper()
            if normalized in allowed and normalized not in valid:
                valid.append(normalized)
            elif normalized not in allowed:
                invalid.append(source_id)

        answer.source_ids = valid

        if invalid:
            logging.warning(
                "Model returned unknown RAG source ids: %s",
                invalid,
            )

        return invalid

    def print_rag_status(self) -> None:
        print("\nНастройки RAG:")
        for key, value in self.rag.status().items():
            print(f"  {key}: {value}")
        if self.last_rag_error:
            print(f"  last_error: {self.last_rag_error}")
        print()

    def print_last_sources(self) -> None:
        print("\nПоследний поиск RAG:")
        print(self.rag.format_sources(self.last_rag_hits))
        print()

    def preview_rag_search(self, query: str) -> None:
        if not query.strip():
            print("Бот: после /rag-search нужен поисковый запрос.")
            return

        try:
            _, hits = self.rag.retrieve_context(query)
            self.last_rag_hits = hits
            self.last_rag_error = None
            self.print_last_sources()
        except Exception as error:
            self.last_rag_hits = []
            self.last_rag_error = str(error)
            print(f"Бот: ошибка RAG-поиска: {error}")

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def trim_history(self, session_id: str) -> None:
        history = self.get_session_history(session_id)
        messages = history.messages
        if len(messages) <= self.max_history_messages:
            return
        recent_messages = messages[-self.max_history_messages:]
        while recent_messages and recent_messages[0].type != "human":
            recent_messages = recent_messages[1:]
        history.clear()
        history.add_messages(recent_messages)

    def _append_turn_to_history(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Store a completed turn without RunnableWithMessageHistory.

        RunnableWithMessageHistory is deprecated in recent LangChain versions.
        The bot only needs simple in-memory chat history, so we update the
        InMemoryChatMessageHistory object directly after a model response.
        """

        history = self.get_session_history(session_id)
        history.add_messages(
            [
                HumanMessage(content=user_text),
                AIMessage(content=assistant_text),
            ]
        )
        self.trim_history(session_id)

    def _invoke_model_with_history(
        self,
        input_data: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Invoke the chain with explicit history injection.

        This replaces RunnableWithMessageHistory while keeping the same
        behavior: previous messages are passed into the prompt through the
        {history} MessagesPlaceholder, then the raw assistant output is saved
        back to the in-memory history.
        """

        if self.chain is None:
            raise RuntimeError("Chain is not configured.")

        history_messages = list(
            self.get_session_history(session_id).messages
        )
        chain_input = {
            **input_data,
            "history": history_messages,
        }

        response = self.chain.invoke(chain_input)

        raw_answer = response.get("raw")
        raw_content = getattr(raw_answer, "content", None)

        if raw_content is None:
            parsed_answer = response.get("parsed")
            if parsed_answer is not None:
                raw_content = parsed_answer.model_dump_json(
                    indent=2,
                )

        if raw_content is not None:
            self._append_turn_to_history(
                session_id=session_id,
                user_text=str(input_data.get("question", "")),
                assistant_text=str(raw_content),
            )

        return response

    def invoke_once(
        self,
        user_text: str,
        session_id: str,
        experiment_id: str | None = None,
        silent: bool = False,
        record_run: bool = True,
        show_prompt: bool | None = None,
        log_prompt: bool | None = None,
    ) -> AnswerProfile | None:
        if self.chain is None:
            raise RuntimeError("Chain is not configured.")

        self.trim_history(session_id)

        original_show_prompt = self.show_prompt
        original_log_prompts = self.log_prompts
        if show_prompt is not None:
            self.show_prompt = show_prompt
        if log_prompt is not None:
            self.log_prompts = log_prompt

        rag_context, rag_hits = self._retrieve_rag_context(
            user_text
        )
        wiki_context, wiki_reason = self._retrieve_wiki_context(
            user_text
        )

        input_data = self._build_prompt_variables(
            question=user_text,
            overrides={
                "rag_context": rag_context,
                "local_context": rag_context,
                "wiki_context": wiki_context,
            },
        )

        prompt_debug = self._build_prompt_debug(
            question=user_text,
            session_id=session_id,
            input_data=input_data,
            rag_hits=rag_hits,
            wiki_reason=wiki_reason,
        )
        self.last_prompt_debug = prompt_debug

        if self.log_prompts:
            self._write_prompt_debug(prompt_debug)

        run_input_data = {
            **input_data,
            "rag": prompt_debug["rag"],
            "wikipedia": prompt_debug["wikipedia"],
            "prompt_debug": prompt_debug,
        }
        self.last_input_data = run_input_data
        self.last_raw_response_text = None
        self.last_parsed_answer = None
        self.last_parsing_error = None
        self.last_invoke_error = None

        try:
            response = self._invoke_model_with_history(
                input_data=input_data,
                session_id=session_id,
            )

            parsing_error = response["parsing_error"]
            raw_answer = response["raw"]
            parsed_answer = response["parsed"]
            if parsing_error is not None or parsed_answer is None:
                self.last_raw_response_text = str(raw_answer.content)
                self.last_parsing_error = str(parsing_error)
                if not silent:
                    print("\nОшибка Pydantic:")
                    print(parsing_error)
                    print("\nСырой ответ модели:")
                    print(raw_answer.content)
                    print()

                if record_run:
                    self.garden.add_run(
                        combo_id=self.combo_id,
                        experiment_id=experiment_id,
                        task="chemistry_cli_bot",
                        model=self.model_name,
                        input_data=run_input_data,
                        output_data=raw_answer.content,
                        validation_ok=False,
                        error=str(parsing_error),
                    )
                self.show_prompt = original_show_prompt
                self.log_prompts = original_log_prompts
                return None

            answer: AnswerProfile = parsed_answer
            self.last_raw_response_text = str(raw_answer.content)
            self.last_parsed_answer = answer.model_dump()
            invalid_source_ids = self._validate_answer_source_ids(
                answer,
                rag_hits,
            )

            if not silent:
                print("\nБот:")
                print(
                    json.dumps(
                        self._answer_payload_for_console(
                            answer,
                            prompt_debug,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                if invalid_source_ids:
                    print(
                        "\nПредупреждение: модель указала "
                        "неизвестные source_ids, они удалены: "
                        + ", ".join(invalid_source_ids)
                    )

                if answer.source_ids:
                    used = [
                        hit
                        for hit in rag_hits
                        if hit.label in answer.source_ids
                    ]
                    print("\nИспользованные источники RAG:")
                    print(self.rag.format_sources(used))
                elif rag_hits:
                    print(
                        "\nRAG нашёл источники, но модель "
                        "не отметила их как использованные. "
                        "Команда /sources покажет выдачу."
                    )
                print()

            if record_run:
                self.garden.add_run(
                    combo_id=self.combo_id,
                    experiment_id=experiment_id,
                    task="chemistry_cli_bot",
                    model=self.model_name,
                    input_data=run_input_data,
                    output_data=answer.model_dump(),
                    validation_ok=True,
                )
            logging.info("User: %s", user_text)
            logging.info("Bot: %s", answer.model_dump_json())
            self.show_prompt = original_show_prompt
            self.log_prompts = original_log_prompts
            return answer

        except Exception as error:
            self.last_invoke_error = str(error)
            if not silent:
                print("\nОшибка обращения к модели. Подробности записаны в лог.\n")
            logging.exception("Model invocation error")
            if record_run:
                self.garden.add_run(
                    combo_id=self.combo_id,
                    experiment_id=experiment_id,
                    task="chemistry_cli_bot",
                    model=self.model_name,
                    input_data=run_input_data,
                    output_data=None,
                    validation_ok=False,
                    error=str(error),
                )
            self.show_prompt = original_show_prompt
            self.log_prompts = original_log_prompts
            return None

    def print_settings(self) -> None:
        print("\nТекущие настройки ученика:")
        for key, value in self.student_context.to_prompt_dict().items():
            print(f"  {key}: {value}")

        print("\nТекущие настройки учителя:")
        for key, value in self.teacher_context.to_prompt_dict().items():
            print(f"  {key}: {value}")

        print("\nRAG:")
        rag_status = self.rag.status()
        print(f"  enabled: {rag_status['enabled']}")
        print(f"  collection: {rag_status['collection_name']}")
        print(f"  embedding_model: {rag_status['embedding_model']}")
        print(f"  top_k: {rag_status['top_k']}")

        print("\nWikipedia:")
        print(f"  {self.wikipedia.status()}")
        print(f"  last_reason: {self.last_wiki_reason}")

        print("\nPrompt debug:")
        print(f"  show_prompt: {self.show_prompt}")
        print(f"  log_prompts: {self.log_prompts}")
        print(f"  log_path: {self.prompt_debug_log_path}")
        print()

    def print_combo(self) -> None:
        print("\nБазовая combo-связка:")
        self.garden.show_combo(self.base_combo_id)
        print("\nАктивная combo-связка:")
        self.garden.show_combo(self.combo_id)
        print()

    def __call__(
        self,
        session_id: str,
        context: StudentContext | None = None,
        teacher_context: TeacherContext | None = None,
    ) -> None:
        if context is not None or teacher_context is not None:
            if self.materialize_context:
                self.configure_contexts(
                    context=context or self.student_context,
                    teacher_context=teacher_context or self.teacher_context,
                )
            else:
                self.student_context = context or self.student_context
                self.teacher_context = teacher_context or self.teacher_context

        print("Локальный учитель химии")
        print("Команды:")
        print("  /combo                    показать combo-связки")
        print("  /settings                 показать настройки")
        print("  /reset                    очистить историю")
        print("  /exit                     выйти")
        print("  /protocol <контекст>      изменить protocol_context")
        print("  /teacher-style <текст>    изменить стиль учителя")
        print("  /constraints <текст>      изменить ограничения ученика")
        print("  /rag                      показать состояние RAG")
        print("  /rag-on                   включить RAG")
        print("  /rag-off                  выключить RAG")
        print("  /rag-k <1..10>            изменить число источников")
        print("  /rag-search <запрос>      проверить поиск без вызова LLM")
        print("  /sources                  показать последний поиск RAG")
        print("  /wiki status              показать состояние Wikipedia")
        print("  /wiki on                  включить Wikipedia-контекст")
        print("  /wiki off                 выключить Wikipedia-контекст")
        print("  /wiki once                принудительно использовать Wikipedia в следующем вопросе")
        print("  /wiki auto                включать Wikipedia только по эвристике")
        print("  /wiki always              добавлять Wikipedia к каждому вопросу")
        print("  /wiki lang <en|ru|uk>     изменить язык Wikipedia")
        print("  /wiki-search <запрос>     проверить Wikipedia без вызова LLM")
        print("  /show-prompt on/off       печатать финальный prompt вместе с ответом")
        print("  /log-prompts on/off       писать prompt debug в logs/prompt_debug.jsonl")
        print("  /prompt last              показать последний prompt debug")

        self.print_settings()
        self.print_combo()

        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break

            if not user_text:
                continue

            command = user_text.lower()

            if command in {"/exit", "выход", "стоп", "конец"}:
                print("Бот: До свидания!")
                break

            if command in {"/reset", "сброс"}:
                self.store.pop(session_id, None)
                print("Бот: Контекст диалога очищен.")
                continue

            if command == "/settings":
                self.print_settings()
                continue

            if command == "/combo":
                self.print_combo()
                continue

            if command == "/rag":
                self.print_rag_status()
                continue

            if command == "/rag-on":
                try:
                    self.rag.set_enabled(True)
                    print("Бот: RAG включён.")
                except Exception as error:
                    print(f"Бот: не удалось включить RAG: {error}")
                continue

            if command == "/rag-off":
                self.rag.set_enabled(False)
                print("Бот: RAG выключен.")
                continue

            if command == "/sources":
                self.print_last_sources()
                continue

            if command in {"/wiki", "/wiki status"}:
                self.print_wiki_status()
                continue

            if command == "/wiki on":
                self.wikipedia.set_enabled(True)
                print("Бот: Wikipedia-контекст включён.")
                continue

            if command == "/wiki off":
                self.wikipedia.set_enabled(False)
                print("Бот: Wikipedia-контекст выключен.")
                continue

            if command == "/wiki once":
                self.force_wiki_next_turn = True
                print("Бот: Wikipedia будет использована для следующего вопроса.")
                continue

            if command == "/wiki auto":
                self.wikipedia.config.auto_mode = True
                print("Бот: Wikipedia работает в auto-режиме.")
                continue

            if command == "/wiki always":
                self.wikipedia.config.auto_mode = False
                print("Бот: Wikipedia будет добавляться к каждому вопросу, если нет safety-блокировки.")
                continue

            if command.startswith("/wiki lang "):
                lang = user_text[len("/wiki lang "):].strip()
                self.wikipedia.set_language(lang)
                print(f"Бот: язык Wikipedia изменён: {self.wikipedia.config.lang}")
                continue

            if command.startswith("/wiki-search "):
                query = user_text[len("/wiki-search "):].strip()
                self.preview_wiki_search(query)
                continue

            if command in {"/show-prompt", "/show-prompt status"}:
                print(f"Бот: show_prompt = {self.show_prompt}")
                continue

            if command == "/show-prompt on":
                self.show_prompt = True
                print("Бот: финальный prompt будет печататься вместе с ответом.")
                continue

            if command == "/show-prompt off":
                self.show_prompt = False
                print("Бот: финальный prompt больше не будет печататься вместе с ответом.")
                continue

            if command in {"/log-prompts", "/log-prompts status"}:
                print(f"Бот: log_prompts = {self.log_prompts}; path = {self.prompt_debug_log_path}")
                continue

            if command == "/log-prompts on":
                self.log_prompts = True
                print(f"Бот: prompt debug будет писаться в {self.prompt_debug_log_path}.")
                continue

            if command == "/log-prompts off":
                self.log_prompts = False
                print("Бот: запись prompt debug выключена.")
                continue

            if command == "/prompt last":
                self.print_last_prompt_debug()
                continue

            if command.startswith("/rag-k "):
                value_text = user_text[len("/rag-k "):].strip()
                try:
                    self.rag.set_top_k(int(value_text))
                    print(
                        "Бот: число источников RAG изменено: "
                        f"{self.rag.config.top_k}"
                    )
                except (TypeError, ValueError) as error:
                    print(f"Бот: неверное значение /rag-k: {error}")
                continue

            if command.startswith("/rag-search "):
                query = user_text[len("/rag-search "):].strip()
                self.preview_rag_search(query)
                continue

            if command.startswith("/protocol "):
                self.student_context.protocol_context = user_text[len("/protocol "):].strip()
                if self.materialize_context:
                    self.refresh_context_combo()
                continue

            if command.startswith("/teacher-style "):
                self.teacher_context.teaching_style = user_text[len("/teacher-style "):].strip()
                if self.materialize_context:
                    self.refresh_context_combo()
                continue

            if command.startswith("/constraints "):
                self.student_context.constraints = user_text[len("/constraints "):].strip()
                if self.materialize_context:
                    self.refresh_context_combo()
                continue

            self.invoke_once(user_text=user_text, session_id=session_id)


if __name__ == "__main__":
    teacher = TeacherContext(
        teacher_profile=(
            "an experienced chemistry teacher "
            "with university-level knowledge"
        ),
        personality_traits=(
            "patient, curious, calm, observant, and honest"
        ),
        tone=(
            "warm and encouraging, but never childish or overly enthusiastic"
        ),
        teaching_style=(
            "build explanations step by step; "
            "use one concrete example before introducing terminology"
        ),
        correction_style=(
            "correct misconceptions directly but gently"
        ),
        language_style=(
            "plain English for a ninth-grade student; "
            "no slang and no unnecessary jargon"
        ),
    )
    bot = CliBot(
        model_name="gemma4:12b",
        garden_root="prompt_garden",
        combo_id="combo_000014",
        fewshot_id="fsh_000002", # or None
        max_history_messages=12,
        materialize_context=True,
        enable_wikipedia_context=True,
        wikipedia_lang="en",
        wikipedia_top_k_results=2,
        wikipedia_doc_chars=1600,
        show_prompt=False,
        log_prompts=True,
        rag_config=RAGConfig(
            enabled=True,
            db_path="data/indexes/introductory_chemistry_chroma",
            collection_name="introchem_theory_v1",
            # None means: read the exact values from the index manifest.
            embedding_model=None,
            ollama_base_url=None,
            # None means: use the exact chunks path stored in the manifest.
            chunks_path=None,
            top_k=4,
            candidate_k=12,
            max_per_section=2,
            max_context_chars=6500,
            filter_default_retrieval=True,
            fail_open=True,
        ),
    )
    bot(session_id="user_123", context=StudentContext(), teacher_context=teacher)
