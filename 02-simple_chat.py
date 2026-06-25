from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import logging

logging.basicConfig(
    filename="chat_session.log", 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


SYSTEM_PROMPT = '''
Ты — учитель школьной химии.

Главные правила:

1. Фактическая точность важнее красивого объяснения.
2. Не выдумывай цвет, запах, агрегатное состояние,
   условия реакции или технику эксперимента.
3. Если данных недостаточно, прямо скажи:
   "У меня недостаточно надёжной информации".
4. Не соглашайся с ложной предпосылкой вопроса.
5. Перед ответом проверь:
   - формулы веществ;
   - коэффициенты в уравнении;
   - сохранение числа атомов;
   - агрегатные состояния;
   - условия реакции.
6. Для химического опыта обязательно укажи:
   - концентрации;
   - средства защиты;
   - основные опасности.
7. Не предлагай смешивать вещества без понятной
   учебной цели и подтверждённой безопасной процедуры.
8. Когда предоставлен контекст из учебника,
   отвечай только на его основании.
9. В конце укажи:
   "Уверенность: высокая / средняя / низкая".'''

@dataclass
class StudentContext:
    """Настройки текущего ученика."""

    school_class: str = "7 класс"
    explanation_level: str = "простой"
    language: str = "русский"


class CliBot:
    def __init__(
        self,
        model_name: str,
        max_history_messages: int = 12,
    ):
        # Локальная модель Ollama
        self.chat_model = ChatOllama(
            model=model_name.strip(),
            temperature=0.1,
        )

        # Истории разговоров:
        # session_id -> InMemoryChatMessageHistory
        self.store: dict[str, InMemoryChatMessageHistory] = {}

        self.max_history_messages = max_history_messages

        # Динамические параметры находятся прямо
        # в системном сообщении.
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                SYSTEM_PROMPT
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.chat_model

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_session_history(
        self,
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        """Получить или создать историю конкретной сессии."""

        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()

        return self.store[session_id]

    def trim_history(self, session_id: str) -> None:
        """
        Ограничить количество предыдущих сообщений.

        Для простого чата без инструментов сообщения обычно
        чередуются: human, ai, human, ai.
        """

        history = self.get_session_history(session_id)
        messages = history.messages

        if len(messages) <= self.max_history_messages:
            return

        recent_messages = messages[-self.max_history_messages:]

        # Желательно начинать историю с сообщения пользователя,
        # а не с ответа модели.
        while (
            recent_messages
            and recent_messages[0].type != "human"
        ):
            recent_messages = recent_messages[1:]

        history.clear()
        history.add_messages(recent_messages)

    @staticmethod
    def print_settings(context: StudentContext) -> None:
        print()
        print("Текущие настройки:")
        print(f"  Класс:               {context.school_class}")
        print(f"  Уровень объяснения:  {context.explanation_level}")
        print(f"  Язык:                {context.language}")
        print()

    def __call__(
        self,
        session_id: str,
        context: StudentContext | None = None,
    ) -> None:
        if context is None:
            context = StudentContext()

        print("Локальный учитель химии")
        print()
        print("Команды:")
        print("  /class <класс>     изменить класс")
        print("  /level <уровень>   изменить уровень объяснения")
        print("  /lang <язык>       изменить язык")
        print("  /settings          показать настройки")
        print("  /reset             очистить историю")
        print("  /exit              выйти")
        print()

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

            # --------------------------------------------
            # Завершение программы
            # --------------------------------------------

            if command in {
                "/exit",
                "выход",
                "стоп",
                "конец",
            }:
                print("Бот: До свидания!")
                break

            # --------------------------------------------
            # Очистка истории
            # --------------------------------------------

            if command in {"/reset", "сброс"}:
                self.store.pop(session_id, None)
                print("Бот: Контекст диалога очищен.")
                continue

            # --------------------------------------------
            # Просмотр настроек
            # --------------------------------------------

            if command == "/settings":
                self.print_settings(context)
                continue

            # --------------------------------------------
            # Изменение класса
            # --------------------------------------------

            if command.startswith("/class "):
                new_class = user_text[len("/class "):].strip()

                if not new_class:
                    print("Бот: Укажите класс.")
                    continue

                context.school_class = new_class
                print(f"Бот: Класс изменён на: {new_class}")
                continue

            # --------------------------------------------
            # Изменение уровня объяснения
            # --------------------------------------------

            if command.startswith("/level "):
                new_level = user_text[len("/level "):].strip()

                if not new_level:
                    print("Бот: Укажите уровень объяснения.")
                    continue

                context.explanation_level = new_level
                print(
                    "Бот: Уровень объяснения изменён на: "
                    f"{new_level}"
                )
                continue

            # --------------------------------------------
            # Изменение языка
            # --------------------------------------------

            if command.startswith("/lang "):
                new_language = user_text[len("/lang "):].strip()

                if not new_language:
                    print("Бот: Укажите язык.")
                    continue

                context.language = new_language
                print(f"Бот: Язык изменён на: {new_language}")
                continue

            # Ограничиваем историю перед новым запросом.
            self.trim_history(session_id)

            try:
                response = self.chain_with_history.invoke(
                    {
                        "question": user_text,
                        "school_class": context.school_class,
                        "explanation_level": (
                            context.explanation_level
                        ),
                        "language": context.language,
                    },
                    {
                        "configurable": {
                            "session_id": session_id,
                        }
                    },
                )

                print()
                print("Бот:", response.content)
                print()

            except Exception as error:
                print()
                print(f"Ошибка обращения к модели: {error}")
                logging.error(f"Error: {error}")
                print()

            logging.info(f"User: {user_text}")
            logging.info(f"Bot: {response}")




if __name__ == "__main__":
    bot = CliBot(
        model_name="llama3.1:8b-instruct-q8_0",
        max_history_messages=12,
    )

    bot(
        session_id="user_123",
        context=StudentContext(
            school_class="5 class",
            explanation_level="простой и наглядный",
            language="english",
        ),
    )