# Testing

This document describes the current smoke-test paths for the repository.

## Goal

The current test goal is not broad coverage.
It is fast confidence that the main school-level QA workflow and Prompt Garden tooling still work.

## Main Smoke-Test Commands

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_smoke_tests.py
```

Prompt Garden tooling:

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_garden_smoke_tests.py
```

Expected success signal for the core smoke tests:

- the command finishes without an exception
- `unittest` ends with `OK`
- the script prints `OK: core retrieval and QA smoke tests passed.`

Expected success signal for the Prompt Garden smoke tests:

- the command finishes without an exception
- `unittest` ends with `OK`
- the script prints `OK: Prompt Garden smoke tests passed.`

## What It Checks

The current smoke-test suite covers these paths:

1. Retrieval-only smoke test.
   It opens the existing Chroma index and runs a small set of tracked fixture queries from `config/introchem_retrieval_queries.jsonl`.

2. End-to-end bot smoke test.
   It runs `CliBot.invoke_once(...)` through the current RAG-enabled path with:
   - `phi4-mini`
   - `combo_000014`
   - `fsh_000002`

3. Prompt Garden tooling smoke tests.
   They load a tiny tracked fixture workspace under `tests/fixtures/prompt_garden_smoke/` and verify:
   - normalized artifact loading into review rows
   - control-panel and analysis-loader bundle assembly
   - diff and segment-alignment helpers
   - runner dry-run planning and command generation
   - cleanup safety and dependency previews
   - Streamlit control-panel import and scope loading

## Preconditions

Before running the smoke tests, make sure:

- Ollama is running locally
- `phi4-mini` is available locally
- the Introductory Chemistry Chroma index already exists under `data/indexes/introductory_chemistry_chroma`
- the environment includes both the base dependencies and the RAG dependencies

For Prompt Garden smoke tests specifically:

- no live model run is required
- no local RAG index is required
- `streamlit` must be installed because the control panel app is imported directly

## Current Scope Boundary

These tests are intentionally narrow.

Included now:

- retrieval connectivity
- structured-answer bot invocation
- lightweight fixture-based sanity checks
- Prompt Garden control-panel, review, and runner smoke coverage through a tracked fixture workspace

Not included yet:

- broader regression coverage
- notebook automation
