# Prompt Garden Step By Step

Этот файл описывает практический рабочий цикл Prompt Garden в текущем состоянии проекта.

Главная идея сейчас такая:

1. `Streamlit` — основная панель управления, базового authoring и анализа.
2. `runner` — запуск экспериментов из терминала.
3. `notebook` — power-user поверхность для глубоких и нестандартных изменений.

## 1. Запустить панель управления

Открой Streamlit-приложение:

```powershell
.\.venv\Scripts\streamlit.exe run apps\prompt_garden_review.py
```

После запуска у тебя будет одна точка входа с двумя верхними вкладками:

- `Control`
- `Analysis`

Если нужно, проверь путь в боковой панели в поле `Prompt Garden Root`.

## 2. Сначала посмотреть на текущее состояние workspace

Перейди в `Control`.

Сейчас внутри `Control` есть основные разделы:

- `Workspace`
- `Prompt Workspace`
- `Combo Explorer`
- `Experiments`
- `Cleanup`
- `Review Scopes`

Хороший стартовый порядок такой:

1. открыть `Workspace` и понять общий объём prompts, combos, experiments и review scopes
2. открыть `Prompt Workspace` и вспомнить, какие prompt-ветки уже есть
3. открыть `Combo Explorer` и посмотреть, какие связки уже собраны
4. открыть `Experiments` и проверить существующие эксперименты и их состав

Это помогает не создавать лишние объекты, если нужный prompt или combo уже существует.

## 3. Изучить существующие prompts в Streamlit

Во вкладке `Control -> Prompt Workspace` можно:

- inspect `Prompt Text` first
- then read `Usage & Results`
- use `Archive Prompt` for routine retirement
- keep `Delete Prompt` inside the guarded `Danger Zone`

- искать prompts по `id`, названию, типу, `tree_id`, `branch` и `tags`
- читать полный текст prompt
- смотреть lineage
- смотреть зависимые combos
- смотреть, в каких экспериментах prompt уже используется

Рекомендуемый порядок:

1. найти нужный `prompt tree`
2. открыть нужный `prompt node`
3. прочитать текст
4. проверить lineage и usage
5. решить, нужен ли новый root prompt, branch или вообще ничего менять не нужно

## 4. Создать новый root prompt в Streamlit

Базовое создание prompt теперь можно делать прямо в Streamlit.

Порядок:

1. открыть `Control -> Prompt Workspace`
2. нажать `Create Root Prompt`
3. внизу страницы откроется общий блок `Authoring Workspace`
4. заполнить поля:
   - `Prompt Type`
   - `Tree ID`
   - `Title`
   - `Branch`
   - `Tags`
   - `Prompt Text`
5. нажать `Register Prompt`

После сохранения:

- authoring-блок закроется
- данные обновятся автоматически
- новый prompt будет сразу выбран в `Prompt Workspace`

## 5. Сделать branch от существующего prompt в Streamlit

Базовое ветвление prompt тоже теперь поддерживается в Streamlit.

Порядок:

1. открыть `Control -> Prompt Workspace`
2. выбрать существующий prompt
3. под карточкой prompt нажать `Branch Prompt`
4. в `Authoring Workspace` отредактировать:
   - `Title`
   - `Branch`
   - `Tags`
   - `Prompt Text`
5. нажать `Register Prompt`

Что важно:

- `Parent Prompt ID`, `Prompt Type` и `Tree ID` в branch-режиме фиксированы
- если родитель архивирован или у него есть проблемы с файлом, Streamlit покажет предупреждение
- после сохранения новый branch сразу выбирается в explorer

## 6. Создать combo в Streamlit

Базовое создание combo теперь тоже делается в Streamlit.

Порядок:

1. открыть `Control -> Combo Explorer`
2. нажать `Create Combo`
3. внизу страницы откроется `Authoring Workspace`
4. заполнить:
   - `Title`
   - `Tags`
   - `Notes`
   - `System Prompt`
   - `User Prompt`
   - при необходимости `Few-Shot Prompt`
5. проверить блок `Prompt Set Preview`
6. нажать `Register Combo`

Что важно:

- `system` и `user` обязательны
- `fewshot` опционален
- архивные prompts скрыты по умолчанию, но их можно показать через `Show Archived Prompts`
- если такой же prompt-role набор уже существует, сохранение будет заблокировано
- после сохранения новый combo сразу выбирается в `Combo Explorer`

## 7. Когда всё ещё нужен notebook

Notebook по-прежнему нужен для операций, которые пока не вынесены в Streamlit:

- прямое редактирование существующего prompt
- смена parent у уже существующего prompt
- редактирование состава уже существующего combo
- более глубокая few-shot подготовка и структурная ручная работа
- bulk-операции и более сложная авторская подготовка

Основной notebook:

- `prompt_garden/control/prompt_garden_experiments_control.ipynb`

После изменений из notebook:

1. вернись в Streamlit
2. нажми `Reload Cached Data`
3. заново открой нужный explorer

## 8. Создать или обновить experiment в Streamlit

Создание и редактирование экспериментов делается в `Control -> Experiments`.

Там можно:

- создать новый experiment
- найти experiment по имени или `id`
- обновить metadata
- прикреплять и откреплять combos
- смотреть composition
- генерировать команду запуска

Минимальный рабочий порядок:

1. открыть `Control -> Experiments`
2. создать experiment
3. заполнить минимум:
   - `name`
   - `goal`
   - `hypothesis`
   - `tags`
   - `status`
4. сохранить experiment
5. выбрать его в списке

## 9. Прикрепить combos к experiment

После выбора experiment в том же разделе:

1. открыть блок управления составом
2. выбрать нужные combos
3. прикрепить их к experiment
4. проверить composition summary

После этого полезно вручную пройтись по составу:

- какие combos участвуют
- какие prompt ids входят в каждое combo
- нет ли шумных, устаревших или слишком похожих вариантов

## 10. Сгенерировать команду запуска

Во вкладке `Control -> Experiments -> Command Builder` можно собрать команду для запуска.

Там задаются, например:

- `model`
- `bot variant`
- few-shot режим
- использование `RAG`
- case set
- фильтры по combo
- фильтры по case
- `run mode`

Рекомендуемый порядок:

1. выбрать experiment
2. открыть `Command Builder`
3. настроить параметры
4. посмотреть dry-run preview
5. убедиться, что число combos и cases соответствует ожиданиям
6. скопировать итоговую команду

## 11. Запустить experiment через runner

Запуск делается из терминала, а не внутри Streamlit.

Базовый вид команды:

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_experiment.py --experiment-id exp_000007 --model phi4-mini
```

Полезные режимы:

- `--dry-run`
- `--run-mode missing`
- `--run-mode failed`
- `--only-combo ...`
- `--only-case-id ...`

Рекомендуемый порядок:

1. сначала сделать `dry-run`
2. потом запустить реальный прогон
3. дождаться записи raw и normalized artifacts

## 12. Открыть результаты в Analysis

После завершения запуска перейди в верхнюю вкладку `Analysis`.

Здесь можно:

- выбрать experiment scope по имени или `id`
- посмотреть overview
- открыть ответы по одному
- посмотреть summary metrics
- посмотреть similarity
- оставить notes
- зафинализировать experiment

Если результаты только что появились:

1. нажми `Reload Cached Data`
2. открой `Analysis`
3. выбери нужный scope

## 13. Просмотреть ответы вручную

Для первого прохода используй `Analysis -> Answer Browser`.

Практический порядок:

1. выбрать experiment scope
2. открыть конкретный answer record
3. прочитать:
   - `question`
   - `short answer`
   - `explanation`
   - `score`
   - `combo`
   - `model`
4. переключаться между ответами разных combos
5. сравнивать их глазами на одном и том же кейсе

## 14. Зафиксировать выводы и завершить experiment

Когда общая картина уже понятна, используй `Analysis -> Experiment Notes`.

Там имеет смысл:

1. записать краткий итог эксперимента
2. отметить, какие combos сработали лучше
3. отметить, какие prompts стоит доработать
4. проставить финальный subject score
5. выполнить finalization, если эксперимент завершён

Хороший формат заметок:

- что проверяли
- что подтвердилось
- что не подтвердилось
- какие 1-2 следующих шага

## 15. Почистить workspace, если он разрастается

Для prompt-части теперь используй `Control -> Prompt Workspace`, а `Control -> Cleanup` оставь для combo и experiment cleanup.

Там можно:

- archive combo
- archive experiment
- удалить combo или experiment, если это безопасно

Рекомендуемый порядок:

1. сначала смотреть dependency preview
2. сначала архивировать, а не удалять
3. удалять только то, что точно не нужно и не связано с артефактами

## Короткая памятка

Используй `Streamlit`, когда нужно:

- понять текущее состояние workspace
- создать root prompt
- сделать branch от prompt
- создать combo
- создать или обновить experiment
- прикрепить combos
- сгенерировать команду запуска
- изучить результаты
- оставить notes
- сделать cleanup

Используй `runner`, когда нужно:

- реально выполнить experiment
- сделать rerun
- сузить запуск до нужных combos или cases

Используй `notebook`, когда нужно:

- напрямую переписать существующий prompt
- менять parent у существующего prompt
- перестраивать уже существующее combo
- делать более глубокую few-shot и structural authoring работу

## Минимальный практический сценарий

Если совсем кратко, то рабочий путь теперь такой:

1. открыть `apps/prompt_garden_review.py`
2. в `Control` найти нужные prompts и combos
3. при необходимости прямо в Streamlit создать root prompt, branch или combo
4. если нужна более глубокая правка, уйти в notebook
5. в `Experiments` собрать или обновить experiment
6. в `Command Builder` сгенерировать команду
7. запустить `scripts/run_prompt_experiment.py`
8. вернуться в `Analysis`
9. просмотреть ответы и метрики
10. записать итог и решить, какой следующий эксперимент делать
