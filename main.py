import os
import time
from dotenv import load_dotenv
load_dotenv()
import sys
import argparse


import openai

print(openai.__version__)


from openai import OpenAI


api_key = os.getenv("api")



from openai import OpenAI
import os
import requests

# api_key = os.getenv("api")

class ChooseAndAnswer:
    def __init__(self, question, api_key = os.getenv("api"), first_stop = False):
        self.first_stop = first_stop
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key)
        self.question = question
        self.preffered_models = [
            "meta-llama/llama-3-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-7b-it:free"]
        self.answer = {}
        self.max_tokens = 1000
        self.timeout = 20

    def __get_free_models(self, add_preffered = True, verbose = False):
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        models = resp.json()["data"]
        models = [m["id"] for m in models if ":free" in m["id"]]
        if add_preffered:
            models = self.preffered_models + models
            models = list(dict.fromkeys(models))
        if verbose:
            print('models:', models)
        return models

    def __try_llm(self, model, retries = 2, sleep = 10):
        for _ in range(retries):
            try:
                print(f"Trying: {model}")

                answer = self.__get_answer(model, mode = 'chat')
                return answer

            except Exception:
                time.sleep(sleep)

    def __get_answer(self, model, mode = 'chat'): # other variant 'responses'
        if mode == 'chat':
            completion = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Ты дружелюбный ассистент. Отвечай кратко."},
                            {"role": "user", "content": f"{self.question}"}],
                        max_tokens=self.max_tokens,
                        timeout=self.timeout  # важно!
                        # Примеры полезных параметров:
                        # temperature=0.7,
                    )
            answer = completion.choices[0].message.content
        elif mode == 'responses':
            ans = self.client.responses.create(
                        model=model,  # название модели
                        instructions="Ты дружелюбный ассистент. Отвечай кратко и по делу.", # Краткая инструкция ассистенту (необязательно)
                        input=self.question, # Сам запрос (строка или массив «контента»)
                        # Примеры полезных параметров:
                        # temperature=0.7,
                        max_output_tokens=self.max_tokens,
                        timeout=self.timeout,  # чтобы не вешать приложение при долгом ответе
                    )
            answer = ans.output_text.strip()
        else:
            raise ValueError(f'Unknown argument, was inputed: {mode}, but possible is chat or responses')

        return answer

    def ask_llm(self):
        models = self.__get_free_models(verbose=True)

        for model in models:
            try:
                ans = self.__try_llm(model)
            except Exception as e:
                print(f'Model was failed in: {e}')

            if ans and len(ans) > 30:
                self.answer.update({model: ans})

            # early stop
            if self.answer and self.first_stop:
                return self.answer

        if not self.answer:
            return "No free models responded"
        else:
            return self.answer


def main():
    # 2) Считываем аргументы запуска    
    parser = argparse.ArgumentParser(description="Первый запрос к LLM")
    parser.add_argument(
        "-q", "--query",
        default="Привет, бот! Назови столицу Франции.",
        help="Текст запроса к модели (по умолчанию — простой вопрос)"
    )
    args = parser.parse_args()

    llm = ChooseAndAnswer(question=args.query, first_stop = True)
    response = llm.ask_llm()
    print(response)
    return 0



if __name__ == '__main__':
    # python main.py -q "Приведи пример интерпретации логитов как соотношения шансов для логистической регрессии для предсказания, например, цены на земельный участок"
    # question = 'Как логиты связаны с соотношением шансов? Откуда берутся значения шансов, чтобы посчитать логиты? Если мы не используем шансы для расчёта - то откуда знаем, что логиты - это соотношение шансов?'
    sys.exit(main())
    
    

    