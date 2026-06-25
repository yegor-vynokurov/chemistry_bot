from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser # make str answer without tech data
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.middleware import before_model
from langchain.agents import AgentState

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langchain.agents.middleware import dynamic_prompt, ModelRequest

import inspect
import langchain



from typing import Any

from dataclasses import dataclass




model = ChatOllama(
    model="llama3.1:8b-instruct-q8_0",
    temperature=1,
)


@before_model
def trim_messages(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """
    Оставляет только последние 9 сообщений.

    В простом диалоге без инструментов это примерно:
    четыре предыдущие пары user/assistant
    плюс новый вопрос пользователя.
    """
    messages = state["messages"]

    if len(messages) <= 9:
        return None

    recent_messages = messages[-9:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages,
        ]
    }

@dataclass
class StudentContext:
    level: str
    language: str = "русский"


@dynamic_prompt
def chemistry_prompt(request: ModelRequest) -> str:
    context = request.runtime.context

    return (
        "Ты учитель по химии. "
        f"Уровень ученика: {context.level}. "
        f"Язык ответа: {context.language}. "
        "Объясняй просто, последовательно и с примерами. "
        "Используй эмодзи для наглядности."
    )


agent = create_agent(model = model, 
                     checkpointer=InMemorySaver(), # here we may use sql checkpointers
                     tools=[], # разнообразные инструменты, связанные с наблюдениями и действиями

                     middleware=[trim_messages, chemistry_prompt],  # Добавляем middleware, если хотим сохранять часть сообщений

                     # if we want to add context
                     system_prompt="Ты - учитель по химии. "
                                    "Объясняй концепции просто и с примерами. "
                                    "Используй эмодзи для наглядности."
                     )


if __name__ == '__main__':

    # print("LangChain version:", langchain.__version__)
    # print(inspect.signature(create_agent))
    config = {
        "configurable": {
            "thread_id": "conversation_01",
        }
    }

    current_context = StudentContext(
        level="6 класс",
        language="русский",
    )

    print("Локальный учитель химии")
    print()
    print("Команды:")
    print("  /level <уровень>    изменить уровень")
    print("  /lang <язык>        изменить язык")
    print("  /context            показать текущие настройки")
    print("  /exit               завершить программу")
    print()

    while True:
        user_input = input("Вы: ").strip()

        if not user_input:
            continue

        # ----------------------------------------------------
        # Выход
        # ----------------------------------------------------

        if user_input.lower() in {"/exit", "exit", "quit", "выход"}:
            print("Работа завершена.")
            break

        # ----------------------------------------------------
        # Изменение уровня
        # ----------------------------------------------------

        if user_input.startswith("/level "):
            new_level = user_input[len("/level "):].strip()

            if not new_level:
                print("Укажите уровень, например: /level 8 класс")
                continue

            current_context.level = new_level

            print(f"Уровень изменён: {current_context.level}")
            print()
            continue

        # ----------------------------------------------------
        # Изменение языка
        # ----------------------------------------------------

        if user_input.startswith("/lang "):
            new_language = user_input[len("/lang "):].strip()

            if not new_language:
                print("Укажите язык, например: /lang украинский")
                continue

            current_context.language = new_language

            print(f"Язык изменён: {current_context.language}")
            print()
            continue

        # ----------------------------------------------------
        # Просмотр контекста
        # ----------------------------------------------------

        if user_input == "/context":
            print()
            print("Текущий контекст:")
            print(f"  уровень: {current_context.level}")
            print(f"  язык:    {current_context.language}")
            print()
            continue

        # ----------------------------------------------------
        # Обычный вопрос пользователя
        # ----------------------------------------------------

        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
                context=current_context,
            )

            answer = response["messages"][-1].content

            print()
            print("Учитель:", answer)
            print()

        except Exception as error:
            print()
            print(f"Ошибка: {error}")
            print()