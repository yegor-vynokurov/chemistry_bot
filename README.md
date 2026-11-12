# Chemistry Bot

English-language local-LLM chemistry assistant for middle-school and high-school students.

final goal: self learned bot with saved the data about every student and its conversations with llm.

ВНИМАНИЕ, ВАЖНО: перевести промпт гарден на скл базу. Будет проще отслеживать индексы и эксперименты. Не нужно хранить весь системный или юзер-промпт, достаточно хранить индексы соответствующих частей, и если надо - программно воспроизвести всё целиком. Так мы будем видеть, какие промпты используются часто, какие наследуются (содежит идентификатор родителя), как будут ветвиться. Если он рут, то его индекс будет 0. Если нужно собрать статистику, то запрос к базе покажет: в какой беседе какой промпт использовался, из чего состоял, как соотносится. Так же и юзер промпт - его не надо хранить в базе целиком. Вопросы учеников - отдельно, в соответствующей таблице. Вспомогательные штуки типа класса и так далее - тоже отдельно, в профиле ученика. Так же и личность учителя тоже должна быть в соответстуующей таблице.


команда для тьютор бота с заданным классификатором; основной тьютор - модель по умолчанию, шоу джсон - это для дебага
python -m chemistry_bot.tutor_graph.app --classifier-model phi4-mini-4k:latest --show-json



важно: возможность подключиить чат гпт или другого агента по кей, подключить, проверить, тратит ли токены

ВАЖНО!!!!!!!!!!!! пусть из энисинг ллм текст от раг приходит в любом случае. Не нужно городить огород, пользователь захотел добавить - получил. Чтобы не было - он добавил, мы обрезали, не нужно.
Нужно менять количество фрагментов, которые подаются - указать в ридми, как это делать.






This repository combines:

- a school-focused chemistry chat bot
- a textbook-grounded RAG pipeline
- a Prompt Garden subsystem for prompt versioning, combo management, experiment tracking, and operator review




 указать PYTHONPATH:

cd D:\documents\chemistry_bot
$env:PYTHONPATH = "src"
python -m chemistry_bot.cli.legacy_bot

## Current Public Baseline

The current working baseline is:

- `src/chemistry_bot/cli/rag_bot.py` as the main RAG-enabled bot path
- `src/chemistry_bot/retrieval/introchem_rag.py` as the bot-facing retrieval integration layer
- `scripts/run_chemistry_bot_rag.py` as the recommended public launcher
- `prompt_garden/` as a top-level subsystem rather than hidden runtime data

The older `src/chemistry_bot/cli/legacy_bot.py` variant is still kept intentionally as a parallel experimental lineage and is not presented as the main path.

## Audience And Scope

This bot is designed for:

- middle-school students
- high-school students

It is not positioned as a university, graduate, or specialist chemistry tutor.

The current verified corpus and prompt design are English-first.

## Quickstart

Run the current bot:

```powershell
.\.venv\Scripts\python.exe scripts\run_chemistry_bot_rag.py
```

Rebuild the textbook-to-RAG pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py --dry-run
```

Run the core smoke tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_smoke_tests.py
```

Launch the Prompt Garden control panel:

```powershell
.\.venv\Scripts\streamlit.exe run apps\prompt_garden_review.py
```

Expected smoke-test success signal:
`unittest` ends with `OK`.

## Repository Map

- `scripts/`
  Stable public entry points for bot launch, retrieval inspection, corpus rebuild, and smoke tests.

- `docs/`
  Human-facing documentation for repository layout, data sources, testing, and the RAG pipeline.

- `tests/`
  Lightweight confidence checks for retrieval connectivity and end-to-end QA.

- `src/`
  Low-level pipeline scripts plus the package-oriented scaffold in `src/chemistry_bot/`.

- `prompt_garden/`
  Prompt texts, prompt lineage, curated baseline assets, experiment records, the Streamlit control panel, and notebook authoring assets.

- `config/`
  Lightweight tracked retrieval fixtures and small reproducibility-oriented assets.

- `data/`
  Local corpus inputs and generated normalized, chunked, and indexed artifacts.

## Corpus

The current verified RAG workflow is tuned for:

- `Introductory Chemistry, 1st Canadian Edition`

The parser and retrieval pipeline are intentionally specialized for this corpus.
Source, provenance, and redistribution notes are documented in [docs/data-sources.md](docs/data-sources.md).

## Prompt Garden

Prompt Garden is a first-class part of the project.

It is used for:

- storing prompt texts
- tracking prompt-node history
- generating and evaluating system/user combos
- recording experiment metadata and results
- managing experiments and review from a Streamlit control panel
- keeping the notebook for prompt authoring, branching, and power-user edits

The main operator surface lives at `apps/prompt_garden_review.py`.
The notebook fallback for authoring work lives at `prompt_garden/control/prompt_garden_experiments_control.ipynb`.
The tracked public baseline lives under `prompt_garden/curated/` and currently exposes `combo_000014`, `sys_000006`, `usr_000006`, and `fsh_000002`.

The prompt-centric operator flow now centers on `Control -> Prompt Workspace`:

- filter and select prompts in one place
- read `Prompt Text` before acting
- inspect `Usage & Results` to see combo/model performance
- choose `Branch Prompt`, `Archive Prompt`, or the guarded delete `Danger Zone`

`Control -> Cleanup` is now reserved for combo and experiment cleanup rather than prompt browsing.

## Current Architecture Notes

- `src/chemistry_bot/` is now the package-oriented scaffold for the intended long-term layout.
- The active bot, retrieval, and prompt-operations implementation modules now live inside that package.
- Public scripts, smoke tests, and the Prompt Garden notebook now import through that package path.
- The repository still favors structural clarity and reproducibility over premature deep refactoring.

## More Documentation

- Repository layout: [docs/repository-layout.md](docs/repository-layout.md)
- RAG pipeline details: [docs/rag-pipeline.md](docs/rag-pipeline.md)
- Testing workflow: [docs/testing.md](docs/testing.md)
- Data directory notes: [data/README.md](data/README.md)
- Config and fixtures: [config/README.md](config/README.md)
- Source layout notes: [src/README.md](src/README.md)
- Test directory notes: [tests/README.md](tests/README.md)
- Prompt Garden workspace: [prompt_garden/README.md](prompt_garden/README.md)
- Curated Prompt Garden baseline: [prompt_garden/curated/README.md](prompt_garden/curated/README.md)
- Stable script entry points: [scripts/README.md](scripts/README.md)





last worked command
python -m chemistry_bot.tutor_graph.app --classifier-model phi4-mini-4k:latest --show-json


python -m chemistry_bot.tutor_graph.app `
  --classifier-model phi4-mini-4k:latest `
  --show-json

/classification





python -m chemistry_bot.integrations.anythingllm_probe `
  --tutor-model gemma4:12b `
  --classifier-model phi4-mini-4k:latest `
  --user-document-policy classifier `
  --no-needs-documents-dev `
  --tutor-db "data/tutor_graph/tutor.db" `
  --student-id "anythingllm-desktop-local"


обязательно читать книги в энисинг ллм и добавлять их в ответ: (--needs-documents-dev)

python -m chemistry_bot.integrations.anythingllm_probe `
  --tutor-model gemma4:12b `
  --classifier-model phi4-mini-4k:latest `
  --user-document-policy classifier `
  --needs-documents-dev `
  --tutor-db "data/tutor_graph/tutor.db" `
  --student-id "anythingllm-desktop-local"



  3 вариант, тест

  python -m chemistry_bot.integrations.anythingllm_probe `
  --tutor-model gemma4:12b `
  --classifier-model phi4-mini-4k:latest `
  --turn-relation-model phi4-mini-4k:latest `
  --turn-relation-num-ctx 2048 `
  --followup-model phi4-mini-4k:latest `
  --safety-embedding-model embeddinggemma:latest `
  --user-document-policy classifier `
  --no-needs-documents-dev `
  --tutor-db "data/tutor_graph/tutor.db" `
  --student-id "anythingllm-desktop-local"

python -m chemistry_bot.integrations.anythingllm_probe `
  --tutor-model gemma4:12b `
  --tutor-num-ctx 8192 `
  --tutor-num-predict 1536 `
  --classifier-model phi4-mini-4k:latest `
  --turn-relation-model phi4-mini-4k:latest `
  --turn-relation-num-ctx 2048 `
  --followup-model phi4-mini-4k:latest `
  --safety-embedding-model embeddinggemma:latest `
  --user-document-policy classifier `
  --no-needs-documents-dev `
  --tutor-db "data/tutor_graph/tutor.db" `
  --student-id "anythingllm-desktop-local"

python -m chemistry_bot.integrations.anythingllm_probe `
  --tutor-model gemma4:12b `
  --tutor-num-ctx 8192 `
  --tutor-num-predict 1536 `
  --classifier-model phi4-mini-4k:latest `
  --turn-relation-model phi4-mini-4k:latest `
  --turn-relation-num-ctx 2048 `
  --followup-model phi4-mini-4k:latest `
  --safety-embedding-model embeddinggemma:latest `
  --rag-relevance-embedding-model embeddinggemma:latest `
  --rag-relevance-medium-threshold 0.45 `
  --rag-relevance-high-threshold 0.58 `
  --rag-relevance-top-k 2 `
  --user-document-policy classifier `
  --no-needs-documents-dev `
  --tutor-db "data/tutor_graph/tutor.db" `
  --student-id "anythingllm-desktop-local"






агент пишет:

  Отдельного канонического файла продуктовой политики в проекте нет. Поэтому я записал принятую политику как рабочий baseline, не меняя .planning/entry_point/project_policy.md: он регулирует управление проектом, а не поведение тьютора. Сроки хранения данных, согласие, доступ, удаление и экспорт, а также выбор между полной передачей фрагментов RAG и фильтрацией по релевантности остаются открытыми.

вот и прокол: цели мы вынесли, замысел вынесли, а политика где-то потерялась. Нужно проанализировать и найти и положить где-то рядом с целями.

нужно различать: конкретная политика по отношению к инструментам и методическим материалам .планнинг,
и политика по отношению к проекту, с которым мы работаем. Вот она где-то рядом с целями должна находиться, и агент должен помнить, что она там и что её нужно использовать.

после записи цели слеующий шаг не предложил. спрашиваю сама.



далее, я дала агенту добро на конкретные действия в проекте. Я думала, что он сначала сделает оформление чейнджа, как мы договаривались: сначала документы как понял, что это значит, потом дефинишн, потом план, потом одобрение, потом исполнение. А тут агент, как я поняла, пошёл сразу скакать и работать.

также, боюсь, и с тестами может быть: вместо планирования и размышления - сразу в работу.

тут я бы всё же свободу чуть-чуть подограничила. Но давай не "так надо, дальше не иди", а
"для оптимизации разработки следуй обязательным шагам
1. три начальных документа - сверка с пользователем.
2. дефинишн (или дефинишн плюс план, если это допустимо) - и сверка с пользователем.
3. реадинесс и финальная сверка.
4. работа.

что-то вроде этого.
