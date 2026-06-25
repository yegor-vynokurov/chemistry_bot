from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


from typing import Literal

from pydantic import BaseModel, Field, model_validator


from typing import Literal

from pydantic import BaseModel, Field


from typing import Annotated, Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class NoExperiment(BaseModel):
    kind: Literal["none"] = Field(
        description=(
            "Always use the exact value 'none' when no real "
            "physical experiment is proposed."
        )
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Why a physical experiment is unnecessary "
            "or inappropriate."
        ),
    )


class ExperimentPlan(BaseModel):
    kind: Literal["experiment"] = Field(
        description=(
            "Always use the exact value 'experiment' when "
            "a real physical experiment is proposed."
        )
    )

    title: str = Field(
        min_length=1,
        description="Short title of the real experiment.",
    )

    goal: str = Field(
        min_length=1,
        description="Educational goal of the experiment.",
    )

    materials: list[str] = Field(
        min_length=1,
        description="Materials required for the experiment.",
    )

    steps: list[str] = Field(
        min_length=1,
        description="Ordered physical actions of the experiment.",
    )

    safety_notes: list[str] = Field(
        description=(
            "Concrete safety precautions for the experiment."
        ),
    )


ExperimentChoice = Annotated[
    NoExperiment | ExperimentPlan,
    Field(discriminator="kind"),
]


class AnswerProfile(BaseModel):
    short_answer: str = Field(
        description=(
            "Direct definition of the requested concept."
        )
    )

    explanation: str = Field(
        description=(
            "Detailed theory. Do not place examples "
            "or experiments in this field."
        )
    )

    examples: list[str] = Field(
        min_length=1,
        max_length=2,
        description=(
            "One or two chemical examples illustrating "
            "the concept."
        ),
    )

    experiment: ExperimentChoice = Field(
        description=(
            "Return a NoExperiment object with kind='none' "
            "when no experiment is needed. Return an "
            "ExperimentPlan object with kind='experiment' "
            "only for a real physical activity."
        ),
    )

    certainty: Literal[
        "high",
        "medium",
        "low",
    ]

    


llm = ChatOllama(
    model="llama3.1:8b-instruct-q8_0",
    temperature=0,
)

structured_llm = llm.with_structured_output(
    AnswerProfile,
    method="json_schema",
    include_raw=True,
)

response = structured_llm.invoke(
    """
    Explain what a covalent bond is for a ninth-grade student.

    Follow these semantic rules:

    - short_answer must directly define a covalent bond;
    - explanation must explain sharing of valence electrons;
    - examples must contain one or two chemical examples;
    - do not place examples inside explanation;
    - experiment must always be an object;
    - when no real physical experiment is required, use:
      {
        "kind": "none",
        "reason": "..."
      }
    - only use kind="experiment" for a real physical activity;
    - never create an empty ExperimentPlan;
    - do not place additional theory in safety_notes.
    """
)

if response["parsing_error"] is not None:
    print("Ошибка Pydantic:")
    print(response["parsing_error"])

    print("\nСырой ответ:")
    print(response["raw"].content)

else:
    answer: AnswerProfile = response["parsed"]

    print(type(answer))
    print(answer.model_dump_json(indent=2))