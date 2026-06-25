# ollama run gemma4:12b
'''
phi4-mini:latest       78fad5d182a7    2.5 GB    7 hours ago
qwen3.5:9b-q4_K_M      6488c96fa5fa    6.6 GB    47 hours ago
gemma4:12b             4eb23ef187e2    7.6 GB    6 days ago
'''

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

from typing import Any




model = ChatOllama(
    model="llama3.1:8b-instruct-q8_0",
    temperature=1,
)

# response = llm.invoke(
#     "Коротко объясни, что такое YAML, на русском языке."
# )

# print(response.content)

# prompt = ChatPromptTemplate.from_template("Объясни простыми словами, что такое {question} на русском языке")

# chain = RunnableSequence(first=prompt, last=model)

# result = chain.invoke({"question": "логиты (соотношения шансов) в машинном обучении"})
# print(result.content)



# prompt = ChatPromptTemplate.from_template("Переведи на английский {text}")

# # chain = RunnableSequence(first=prompt, last=model)

# # result = chain.invoke({"text": "Корабли лавировали, лавировали да не вылавировали"})
# # print(result.content)
# # The ships had drifted, drifted but didn't drift

# basic_chain = prompt | model | StrOutputParser()

# result = basic_chain.invoke({"text": "Корабли лавировали, лавировали да не вылавировали"})


# prompt1 = PromptTemplate.from_template('Плюсы {topic}')
# prompt2 = PromptTemplate.from_template('Минусы {topic}')
# parralell_chain = RunnableParallel(
#     pros = prompt1 | model | StrOutputParser(),
#     cons = prompt2 | model | StrOutputParser()
# )
# result = parralell_chain.invoke({"topic": "отдых в Турции"})

# prompt1 = ChatPromptTemplate.from_template("максимально возможно грубо ответить на {question}")
# prompt2 = ChatPromptTemplate.from_template("максимально вежливо ответить на {question}")

# def is_respect(text):
#     return 'Пожалуйста' in text

# branch = RunnableBranch(
#     (is_respect, prompt1 | model), 
#     prompt2 | model
# )

# chain = {'question': RunnablePassthrough()} | branch | StrOutputParser()


# # result = chain.invoke({'question': 'Пожалуйста скажите, как пройти в библиотеку'})
# result = chain.invoke({'question': 'Как пройти в библиотеку'})

######################################
# AGENTS



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

agent = create_agent(model = model, 
                     checkpointer=InMemorySaver(), # here we may use sql checkpointers
                     tools=[], # разнообразные инструменты, связанные с наблюдениями и действиями

                     middleware=[trim_messages],  # Добавляем middleware, если хотим сохранять часть сообщений

                     # if we want to add context
                     system_message="Ты дружелюбный помощник по математике. "
                                    "Объясняй концепции просто и с примерами. "
                                    "Используй эмодзи для наглядности."
                     )

# # SQLite (для production)
# from langgraph.checkpoint.sqlite import SqliteSaver
# checkpointer = SqliteSaver.from_conn_string("database.db")

# # PostgreSQL (для масштабирования)
# from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# pip install langgraph-checkpoint-sqlite langgraph-checkpoint-postgres

conf1 = {'configurable': {'thread_id': 'conversation_01'}}

question1 = 'Существует прессованная дымовая шашка на основе производного парафина. Что это за производное?'
response1 = agent.invoke({'messages': [{'role': 'user', 'content': question1}]}, conf1) # Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'

print("пользователь 1:", question1)
print("Бот:", response1["messages"][-1].content)

# question2 = 'Какой активатор горения можно использовать в прессованной дымовой шашке?'
# response2 = agent.invoke({'messages': [{'role': 'user', 'content': question2}]}, conf1)

# print("пользователь 1:", question2)
# print("Бот:", response2["messages"][-1].content)

question3 = 'Напомни, что мы говорили о дымовой шашке только что'
response3 = agent.invoke({'messages': [{'role': 'user', 'content': question3}]}, conf1)

print("пользователь 1:", question3)
print("Бот:", response3["messages"][-1].content)

conf2 = {'configurable': {'thread_id': 'conversation_02'}}

question4 = 'Напомни, что мы говорили о дымовой шашке только что'
response4 = agent.invoke({'messages': {'role': 'user', 'content': question4}}, conf2)

print("пользователь 2:", question4)
print("Бот:", response4["messages"][-1].content)

# print(result)

# Получить состояние потока
state = agent.get_state(conf1)
print("История сообщений:")
for msg in state.values["messages"]:
    print(f"{msg.type}: {msg.content}")
