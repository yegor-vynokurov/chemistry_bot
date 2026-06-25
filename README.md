# Chemistry Bot

English-language local-LLM chemistry assistant for middle-school and high-school students.

This repository combines:

- a school-focused chemistry chat bot
- a textbook-grounded RAG pipeline
- a Prompt Garden subsystem for prompt versioning, combo management, and experiment tracking

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
  Prompt texts, prompt lineage, curated baseline assets, experiment records, and the notebook control surface.

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
- managing prompt evolution through a notebook control surface

The active notebook lives at `prompt_garden/control/prompt_garden_experiments_control.ipynb`.
The tracked public baseline lives under `prompt_garden/curated/` and currently exposes `combo_000014`, `sys_000006`, `usr_000006`, and `fsh_000002`.

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
- Repository roadmap: [REPOSITORY_STRUCTURE_PLAN.md](REPOSITORY_STRUCTURE_PLAN.md)
- Quality criteria: [PROJECT_QUALITY_CRITERIA.md](PROJECT_QUALITY_CRITERIA.md)
