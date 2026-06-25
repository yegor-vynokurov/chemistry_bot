from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal
import logging

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


from pydantic import BaseModel, Field, model_validator

from prompt_garden import PromptGarden

import json



logging.basicConfig(
    filename="chat_session.log",
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


@dataclass
class StudentContext:
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
    def __init__(
        self,
        model_name: str,
        garden_root: str | Path = "prompt_garden",
        combo_id: str = "combo_000001",
        fewshot_id: str | None = None,
        max_history_messages: int = 12,
        materialize_context: bool = True,
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

        self.prompt: ChatPromptTemplate | None = None
        self.chain = None
        self.chain_with_history = None

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
            runtime_placeholders=("question",),
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

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                *fewshot_messages,
                MessagesPlaceholder(
                    variable_name="history"
                ),
                ("human", self.user_prompt),
            ]
        )

        self.chain = self.prompt | self.structured_model

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="raw",
        )


    def _build_prompt_variables(
        self,
        question: str,
        overrides: dict | None = None,
    ) -> dict:
        values = {
            "question": question,
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

    def invoke_once(
        self,
        user_text: str,
        session_id: str,
        experiment_id: str | None = None,
        silent: bool = False,
    ) -> AnswerProfile | None:
        if self.chain_with_history is None:
            raise RuntimeError("Chain is not configured.")

        self.trim_history(session_id)

        input_data = self._build_prompt_variables(question=user_text)


        try:
            response = self.chain_with_history.invoke(
                input_data,
                config={"configurable": {"session_id": session_id}},
            )

            parsing_error = response["parsing_error"]
            raw_answer = response["raw"]
            parsed_answer = response["parsed"]

            if parsing_error is not None or parsed_answer is None:
                if not silent:
                    print("\nОшибка Pydantic:")
                    print(parsing_error)
                    print("\nСырой ответ модели:")
                    print(raw_answer.content)
                    print()

                self.garden.add_run(
                    combo_id=self.combo_id,
                    experiment_id=experiment_id,
                    task="chemistry_cli_bot",
                    model=self.model_name,
                    input_data=input_data,
                    output_data=raw_answer.content,
                    validation_ok=False,
                    error=str(parsing_error),
                )
                return None

            answer: AnswerProfile = parsed_answer

            if not silent:
                print("\nБот:")
                print(answer.model_dump_json(indent=2))
                print()

            self.garden.add_run(
                combo_id=self.combo_id,
                experiment_id=experiment_id,
                task="chemistry_cli_bot",
                model=self.model_name,
                input_data=input_data,
                output_data=answer.model_dump(),
                validation_ok=True,
            )
            logging.info("User: %s", user_text)
            logging.info("Bot: %s", answer.model_dump_json())
            return answer

        except Exception as error:
            if not silent:
                print("\nОшибка обращения к модели. Подробности записаны в лог.\n")
            logging.exception("Model invocation error")
            self.garden.add_run(
                combo_id=self.combo_id,
                experiment_id=experiment_id,
                task="chemistry_cli_bot",
                model=self.model_name,
                input_data=input_data,
                output_data=None,
                validation_ok=False,
                error=str(error),
            )
            return None

    def print_settings(self) -> None:
        print("\nТекущие настройки ученика:")
        for key, value in self.student_context.to_prompt_dict().items():
            print(f"  {key}: {value}")

        print("\nТекущие настройки учителя:")
        for key, value in self.teacher_context.to_prompt_dict().items():
            print(f"  {key}: {value}")
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
    )
    bot(session_id="user_123", context=StudentContext(), teacher_context=teacher)
