from dataclasses import dataclass
from typing import Annotated, Literal
import logging

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field, model_validator


logging.basicConfig(
    filename="chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)


SYSTEM_PROMPT = """
You are a chemistry teacher in a {type_of_school}.
Answer in {language}.

Main rules:
1. Factual accuracy is more important than a beautiful explanation.
2. Do not invent colors, odors, aggregate states, reaction conditions,
   reagent concentrations, amounts, or experimental procedures.
3. Correct false assumptions in the student's question.
4. If reliable information is insufficient, say so clearly.
5. Before answering, check formulas, equation coefficients,
   conservation of atoms, aggregate states, and reaction conditions.
6. Propose an experiment only when it is age-appropriate and can be
   described with exact substances, amounts, concentrations, equipment,
   protective equipment, steps, risks, and disposal instructions.
7. If you cannot provide a safe and precise experiment, choose
   experiment.kind='none'. Never guess amounts or concentrations.
8. Follow the Pydantic response schema. Do not add text outside it.
"""

HUMAN_PROMPT = """
Process the following chemistry request.

Student class: {school_class}
Explanation level: {explanation_level}
Question: {question}

Intent rules:

1. First determine request_type:
   - theory;
   - experiment;
   - mixed.

2. Follow the user's literal intent.

3. Never interpret "experiment with a substance" as
   "synthesize or manufacture the substance" unless the
   user explicitly asks how to prepare or synthesize it.

4. If the user explicitly requests an experiment:
   - set request_type="experiment";
   - focus the answer on an experiment rather than giving
     a long general lecture;
   - examples may be an empty list.

5. If the requested experimental goal is ambiguous,
   return experiment.kind="clarification".

6. Ask for clarification when any of these are unknown:
   - what property should be demonstrated;
   - whether the setting is home or laboratory;
   - whether a teacher or trained adult supervises;
   - what equipment is available.

7. Return experiment.kind="experiment" only when a precise,
   age-appropriate and safe protocol is available in the
   verified protocol context below.

8. Never invent amounts, concentrations, temperatures,
   risks or disposal instructions.

9. For oxidizers:
   - do not propose combustion;
   - do not mix with fuels or reducing agents;
   - do not propose pyrotechnic, propellant or explosive uses;
   - do not heat to dryness.

Verified protocol context:
{protocol_context}
"""


class ExperimentIngredient(BaseModel):
    name: str = Field(
        min_length=1,
        description="Name of the reagent or substance.",
    )
    formula: str | None = Field(
        default=None,
        description="Chemical formula, or null when it is not applicable.",
    )
    amount: str = Field(
        min_length=1,
        description="Exact amount with units, for example '10 mL' or '2 g'.",
    )
    is_solution: bool = Field(
        description="True when the ingredient is used as a solution."
    )
    concentration: str | None = Field(
        default=None,
        description=(
            "Exact concentration with units, for example '3% w/v' or "
            "'0.10 mol/L'. Use null for a pure substance or when "
            "concentration is not applicable."
        ),
    )
    purpose: str = Field(
        min_length=1,
        description="Role of this ingredient in the experiment.",
    )

    @model_validator(mode="after")
    def solution_requires_concentration(self):
        if self.is_solution and not self.concentration:
            raise ValueError(
                "A solution ingredient must include its concentration."
            )
        return self


class NoExperiment(BaseModel):
    kind: Literal["none"] = Field(
        description="Use the exact value 'none'."
    )
    reason: str = Field(
        min_length=1,
        description=(
            "Why a real physical experiment is unnecessary, unsafe, "
            "or cannot be specified reliably."
        ),
    )


class ExperimentPlan(BaseModel):
    kind: Literal["experiment"] = Field(
        description="Use the exact value 'experiment'."
    )
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    ingredients: list[ExperimentIngredient] = Field(
        min_length=1,
        description="Reagents with exact amounts and concentrations.",
    )
    equipment: list[str] = Field(
        min_length=1,
        description="Laboratory or household equipment required.",
    )
    protective_equipment: list[str] = Field(
        min_length=1,
        description="Required personal protective equipment.",
    )
    steps: list[str] = Field(
        min_length=1,
        description="Ordered physical actions of the experiment.",
    )
    safety_notes: list[str] = Field(
        min_length=1,
        description="Concrete risks and safety precautions.",
    )
    disposal: str = Field(
        min_length=1,
        description="How to dispose of the resulting materials safely.",
    )
    adult_supervision_required: bool = Field(
        description="Whether an adult or teacher must supervise the activity."
    )

class ClarificationRequest(BaseModel):
    kind: Literal["clarification"] = Field(
        description=(
            "Use the exact value 'clarification' when "
            "the experiment request is too broad or important "
            "safety information is missing."
        )
    )

    reason: str = Field(
        min_length=1,
        description="Why clarification is required.",
    )

    questions: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "One to three concise questions needed before "
            "an experiment can be proposed."
        ),
    )


ExperimentChoice = Annotated[
    NoExperiment
    | ClarificationRequest
    | ExperimentPlan,
    Field(discriminator="kind"),
]


class AnswerProfile(BaseModel):
    request_type: Literal[
        "theory",
        "experiment",
        "mixed",
    ] = Field(
        description=(
            "The main intent of the user's request."
        )
    )

    short_answer: str = Field(
        min_length=1,
        description="Direct response to the user's request.",
    )

    explanation: str = Field(
        min_length=1,
        description=(
            "Relevant chemistry background. Keep this brief "
            "when request_type is 'experiment'."
        ),
    )

    examples: list[str] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "Optional chemical examples. Return an empty list "
            "when examples are not useful."
        ),
    )

    experiment: ExperimentChoice

    certainty: Literal[
        "high",
        "medium",
        "low",
    ]

@dataclass
class StudentContext:
    type_of_school: str = "good private school"
    school_class: str = "6 class"
    explanation_level: str = "simple"
    language: str = "English"


class CliBot:
    def __init__(
        self,
        model_name: str,
        max_history_messages: int = 12,
        protocol_context: str = "No verified experimental protocol was retrieved.",
    ) -> None:
        self.protocol_context = protocol_context
        self.chat_model = ChatOllama(
            model=model_name.strip(),
            temperature=0,
        )

        self.structured_model = self.chat_model.with_structured_output(
            AnswerProfile,
            method="json_schema",
            include_raw=True,
        )

        self.store: dict[str, InMemoryChatMessageHistory] = {}
        self.max_history_messages = max_history_messages

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    HUMAN_PROMPT,
                ),
            ]
        )

        self.chain = self.prompt | self.structured_model

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            # include_raw=True returns a dictionary. The raw field contains
            # the AIMessage that can be stored in chat history.
            output_messages_key="raw",
        )

    def get_session_history(
        self,
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def trim_history(self, session_id: str) -> None:
        history = self.get_session_history(session_id)
        messages = history.messages

        if len(messages) <= self.max_history_messages:
            return

        recent_messages = messages[-self.max_history_messages :]

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
        print(f"  Язык:                {context.language}\n")

    @staticmethod
    def print_answer(answer: AnswerProfile) -> None:
        print("\nБот:")
        print(answer.model_dump_json(indent=2))
        print()

        # This branch also demonstrates that Pydantic created
        # one of the two concrete experiment classes.
        if isinstance(answer.experiment, ExperimentPlan):
            print("План эксперимента:")
            for ingredient in answer.experiment.ingredients:
                concentration = ingredient.concentration or "не применяется"
                print(
                    f"  - {ingredient.name}: {ingredient.amount}; "
                    f"концентрация: {concentration}"
                )
            print()

    def __call__(
        self,
        session_id: str,
        context: StudentContext | None = None,
    ) -> None:
        if context is None:
            context = StudentContext()

        print("Локальный учитель химии")
        print("Команды:")
        print("  /class <класс>     изменить класс")
        print("  /level <уровень>   изменить уровень объяснения")
        print("  /lang <язык>       изменить язык")
        print("  /settings          показать настройки")
        print("  /reset             очистить историю")
        print("  /exit              выйти")

        self.print_settings(context)

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

            if command.startswith("/class "):
                new_class = user_text[len("/class ") :].strip()
                if new_class:
                    context.school_class = new_class
                    print(f"Бот: Класс изменён на: {new_class}")
                continue

            if command.startswith("/level "):
                new_level = user_text[len("/level ") :].strip()
                if new_level:
                    context.explanation_level = new_level
                    print(f"Бот: Уровень объяснения изменён на: {new_level}")
                continue

            if command.startswith("/lang "):
                new_language = user_text[len("/lang ") :].strip()
                if new_language:
                    context.language = new_language
                    print(f"Бот: Язык изменён на: {new_language}")
                continue

            self.trim_history(session_id)

            try:
                response = self.chain_with_history.invoke(
                    {
                        "question": user_text,
                        "school_class": context.school_class,
                        "explanation_level": context.explanation_level,
                        "language": context.language,
                        "type_of_school": context.type_of_school,
                        "protocol_context": protocol_context,
                    },
                    config={
                        "configurable": {
                            "session_id": session_id,
                        }
                    },
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

                    logging.error(
                        "Structured output error: %s; raw=%s",
                        parsing_error,
                        raw_answer.content,
                    )
                    continue

                answer: AnswerProfile = parsed_answer
                self.print_answer(answer)

                logging.info("User: %s", user_text)
                logging.info("Bot: %s", answer.model_dump_json())

            except Exception:
                print("\nОшибка обращения к модели. Подробности записаны в лог.\n")
                logging.exception("Model invocation error")


if __name__ == "__main__":

    protocol_context = (
        "No verified experimental protocol was retrieved."
    )
    
    bot = CliBot(
        model_name="llama3.1:8b-instruct-q8_0",
        max_history_messages=12,
        protocol_context = protocol_context,
    )

    bot(
        session_id="user_123",
        context=StudentContext(
            school_class="9 class",
            explanation_level="interesting but factually precise",
            language="English",
        ),
        
    )
