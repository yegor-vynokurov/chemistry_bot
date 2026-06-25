# Tests

This directory contains lightweight confidence checks for the current public baseline.

## Current Scope

The current tests are smoke tests, not broad regression coverage.

They currently verify:

- retrieval connectivity for the Introductory Chemistry index
- end-to-end structured QA through the RAG-enabled bot path
- Prompt Garden loader, comparison, runner dry-run, and Streamlit review smoke paths through a tiny tracked fixture workspace

## Main Command

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_smoke_tests.py
```

Prompt Garden smoke tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_garden_smoke_tests.py
```

For fuller workflow notes, see [../docs/testing.md](../docs/testing.md).
