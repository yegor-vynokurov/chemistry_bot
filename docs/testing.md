# Testing

This document describes the current smoke-test path for the repository.

## Goal

The current test goal is not broad coverage.
It is fast confidence that the main school-level QA workflow still works.

## Main Smoke-Test Command

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_smoke_tests.py
```

Expected success signal:

- the command finishes without an exception
- `unittest` ends with `OK`
- the script prints `OK: core retrieval and QA smoke tests passed.`

## What It Checks

The current smoke-test suite covers two paths:

1. Retrieval-only smoke test.
   It opens the existing Chroma index and runs a small set of tracked fixture queries from `config/introchem_retrieval_queries.jsonl`.

2. End-to-end bot smoke test.
   It runs `CliBot.invoke_once(...)` through the current RAG-enabled path with:
   - `phi4-mini`
   - `combo_000014`
   - `fsh_000002`

## Preconditions

Before running the smoke tests, make sure:

- Ollama is running locally
- `phi4-mini` is available locally
- the Introductory Chemistry Chroma index already exists under `data/indexes/introductory_chemistry_chroma`
- the environment includes both the base dependencies and the RAG dependencies

## Current Scope Boundary

These tests are intentionally narrow.

Included now:

- retrieval connectivity
- structured-answer bot invocation
- lightweight fixture-based sanity checks

Not included yet:

- Prompt Garden automated tests
- broader regression coverage
- notebook automation
