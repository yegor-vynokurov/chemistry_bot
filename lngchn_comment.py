from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate
import os 
# import config 

# api_key = config.chat_gpt_api_key
# os.environ['OPENAI_API_KEY'] = api_key
# # Создаем чат-модель (ChatGPT) с нужными параметрами
# chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

chat = ChatOllama(
    model="llama3.1:8b-instruct-q8_0",
    temperature=1,
)

store = {}  # Создаем пустой словарь для хранения истории сообщений
# функция для получения логов по session_id
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    print(session_id)
    # если session_id нет в словаре логов, создаем новую историю сообщений (ChatMessageHistory)
    if session_id not in store:  
        store[session_id] = ChatMessageHistory()
    # возвращаем логи по session_id
    return store[session_id]  

 

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты асистент, который хорошо разбирается в {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | chat

 

chain_with_history = RunnableWithMessageHistory(
    chain,
    # добавляем нашу функцию для сохранения истории переписки
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.invoke(
    {"ability": "математика", "question": "что значит косинус"},
    config={"configurable": {"session_id": "foo"}}
))