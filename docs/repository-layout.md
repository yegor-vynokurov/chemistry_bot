# Repository Layout

This document describes the current near-final repository layout.

The goal of this stage is not to finish every refactor.
The goal is to make the structure predictable, public-facing, and easy to evolve without another large reset.

## Layout Principles

The repository currently separates:

- public documentation in `docs/`
- stable entry-point wrappers in `scripts/`
- smoke tests in `tests/`
- package-oriented source scaffolding in `src/chemistry_bot/`
- Prompt Garden as a visible top-level subsystem in `prompt_garden/`
- local or generated corpus artifacts in `data/`

## Current Top-Level Roles

- `README.md`
  Canonical public entry point for the repository.

- `docs/`
  Supporting architecture, data, pipeline, and testing documentation.

- `scripts/`
  User-facing commands for running the bot, inspecting retrieval, rebuilding the corpus pipeline, and running smoke tests.

- `tests/`
  Lightweight repository confidence checks for retrieval connectivity and end-to-end QA.

- `src/`
  Current low-level retrieval pipeline scripts plus the new `src/chemistry_bot/` package scaffold.

- `prompt_garden/`
  Top-level prompt-management subsystem with the Streamlit control panel, notebook authoring surface, prompt lineage, curated baselines, and experiment history.

- `config/`
  Lightweight tracked fixtures for retrieval checks and future small configuration docs.

- `data/`
  Local corpus inputs and generated outputs. This remains mostly ignored and reproducible by command.

## Package Scaffold

The package-oriented scaffold now lives under:

- `src/chemistry_bot/cli/`
- `src/chemistry_bot/promptops/`
- `src/chemistry_bot/retrieval/`
- `src/chemistry_bot/schemas/`
- `src/chemistry_bot/utils/`

Current role of this scaffold:

- provide stable package import paths
- mirror the intended long-term architecture
- reduce future disruptive moves
- keep the implementation close to the final intended structure without breaking the current public scripts

The active bot, retrieval, and prompt-management implementation modules now live inside `src/chemistry_bot/`.
The old root-level implementation files have been migrated out of the repository root.

At this stage, scripts and tests activate the scaffold through explicit `src` path bootstrapping rather than through a packaged installation workflow.

## Prompt Garden Placement

`prompt_garden/` remains at the repository top level by design.

It is not treated as disposable runtime data because it represents:

- prompt source texts
- prompt version history
- combo and node relationships
- experiment records
- the Streamlit control panel and notebook authoring surface used for prompt operations

## Data Flow Placement

The current Introductory Chemistry flow is organized around these locations:

- `data/raw/introductory_chemistry/`
  Local source textbook files.

- `data/normalized/introductory_chemistry/`
  Parsed chapter outputs.

- `data/rag/introductory_chemistry/`
  Chunked retrieval artifacts and retrieval-review outputs.

- `data/indexes/introductory_chemistry_chroma/`
  Local Chroma persistence directory and manifest.

- `config/introchem_retrieval_queries.jsonl`
  Lightweight tracked retrieval fixtures.

The shared path map for this workflow now lives in:

- `src/chemistry_bot/retrieval/layout.py`

That module is the current source of truth for the main corpus paths used by scripts and smoke tests.
