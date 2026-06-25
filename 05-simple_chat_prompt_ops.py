from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
import logging

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field, model_validator

from prompt_garden import PromptGarden


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
    type_of_school: str = "good private school"
    school_class: str = "9 class"
    explanation_level: str = "interesting but factually precise"
    language: str = "English"
    protocol_context: str = "No verified experimental protocol was retrieved."
    notes: str | None = '' # some notes about specific student e.g. student is fool and stupid or student is a clever guy

@dataclass
class TeacherContext:
    level_of_teacher: str = "strong high university like Sorbonna" # education level of the teacher e.g.primary school, academic etc
    type_of_humour: str = "like to joke" # type of humour
    notes: str | None = '' # some notes about personality of the teacher or specific demands to the explanation


class CliBot:
    def __init__(
        self,
        model_name: str,
        garden_root: str | Path = "prompt_garden",
        combo_id: str = "combo_000001",
        max_history_messages: int = 12,
    ) -> None:
        self.model_name = model_name.strip()
        self.combo_id = combo_id

        self.garden = PromptGarden(garden_root)
        self.garden.init()

        combo_prompts = self.garden.read_combo_prompts(combo_id)

        if "system" not in combo_prompts or "user" not in combo_prompts:
            raise ValueError("Combo must include at least 'system' and 'user' prompts.")

        self.system_prompt = combo_prompts["system"]
        self.user_prompt = combo_prompts["user"]

        self.chat_model = ChatOllama(model=self.model_name, temperature=0)
        self.structured_model = self.chat_model.with_structured_output(
            AnswerProfile,
            method="json_schema",
            include_raw=True,
        )

        self.store: dict[str, InMemoryChatMessageHistory] = {}
        self.max_history_messages = max_history_messages

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
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

    @staticmethod
    def print_settings(context: StudentContext) -> None:
        print("\nТекущие настройки:")
        print(f"  Тип школы:           {context.type_of_school}")
        print(f"  Класс:               {context.school_class}")
        print(f"  Уровень объяснения:  {context.explanation_level}")
        print(f"  Язык:                {context.language}")
        print(f"  Protocol context:    {context.protocol_context}\n")

    def print_combo(self) -> None:
        print("\nТекущая combo-связка:")
        self.garden.show_combo(self.combo_id)
        print()

    @staticmethod
    def print_answer(answer: AnswerProfile) -> None:
        print("\nБот:")
        print(answer.model_dump_json(indent=2))
        print()

        if isinstance(answer.experiment, ClarificationRequest):
            print("Бот просит уточнение:")
            for question in answer.experiment.questions:
                print(f"  - {question}")
            print()
        elif isinstance(answer.experiment, ExperimentPlan):
            print("План эксперимента:")
            for ingredient in answer.experiment.ingredients:
                concentration = ingredient.concentration or "не применяется"
                print(f"  - {ingredient.name}: {ingredient.amount}; концентрация: {concentration}")
            print()

    def invoke_once(self, user_text: str, session_id: str, context: StudentContext, teacher_context: TeacherContext) -> AnswerProfile | None:
        self.trim_history(session_id)

        input_data = {
            "question": user_text,
            "school_class": context.school_class,
            "explanation_level": context.explanation_level,
            "language": context.language,
            "type_of_school": context.type_of_school,
            "notes": context.notes, 
            "level_of_teacher": teacher_context.level_of_teacher, 
            "personality_of_teacher": teacher_context.type_of_humour,
            "teacher_notes": teacher_context.notes,
            "protocol_context": context.protocol_context,
        }

        try:
            response = self.chain_with_history.invoke(
                input_data,
                config={"configurable": {"session_id": session_id}},
            )

            parsing_error = response["parsing_error"]
            raw_answer = response["raw"]
            parsed_answer = response["parsed"]

            if parsing_error is not None or parsed_answer is None:
                print("\nОшибка Pydantic:")
                print(parsing_error)
                print("\nСырой ответ модели:")
                print(raw_answer.content)
                print()

                self.garden.add_run(
                    combo_id=self.combo_id,
                    task="chemistry_cli_bot",
                    model=self.model_name,
                    input_data=input_data,
                    output_data=raw_answer.content,
                    validation_ok=False,
                    error=str(parsing_error),
                )
                return None

            answer: AnswerProfile = parsed_answer
            self.print_answer(answer)
            self.garden.add_run(
                combo_id=self.combo_id,
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
            print("\nОшибка обращения к модели. Подробности записаны в лог.\n")
            logging.exception("Model invocation error")
            self.garden.add_run(
                combo_id=self.combo_id,
                task="chemistry_cli_bot",
                model=self.model_name,
                input_data=input_data,
                output_data=None,
                validation_ok=False,
                error=str(error),
            )
            return None

    def __call__(self, session_id: str, context: StudentContext | None = None, teacher_context: TeacherContext | None = None) -> None:
        if context is None:
            context = StudentContext()
        if teacher_context is None:
            teacher_context = TeacherContext()

        print("Локальный учитель химии")
        print("Команды:")
        print("  /class <класс>       изменить класс")
        print("  /level <уровень>     изменить уровень объяснения")
        print("  /lang <язык>         изменить язык")
        print("  /protocol <контекст> изменить protocol_context")
        print("  /combo               показать текущую combo-связку")
        print("  /settings            показать настройки")
        print("  /reset               очистить историю")
        print("  /exit                выйти")

        self.print_settings(context)
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
                self.print_settings(context)
                continue

            if command == "/combo":
                self.print_combo()
                continue

            if command.startswith("/class "):
                context.school_class = user_text[len("/class "):].strip()
                print(f"Бот: Класс изменён на: {context.school_class}")
                continue

            if command.startswith("/level "):
                context.explanation_level = user_text[len("/level "):].strip()
                print(f"Бот: Уровень объяснения изменён на: {context.explanation_level}")
                continue

            if command.startswith("/lang "):
                context.language = user_text[len("/lang "):].strip()
                print(f"Бот: Язык изменён на: {context.language}")
                continue

            if command.startswith("/protocol "):
                context.protocol_context = user_text[len("/protocol "):].strip()
                print("Бот: protocol_context обновлён.")
                continue

            self.invoke_once(user_text=user_text, session_id=session_id, context=context, teacher_context=teacher_context)


if __name__ == "__main__":
    bot = CliBot(
        model_name="llama3.1:8b-instruct-q8_0",
        garden_root="prompt_garden",
        combo_id="combo_000001",
        max_history_messages=12,
    )
    bot(session_id="user_123", context=StudentContext(), teacher_context=TeacherContext())
