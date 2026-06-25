# Source Layout

This directory currently contains two layers:

- low-level pipeline scripts in `src/*.py`
- the package-oriented scaffold in `src/chemistry_bot/`

## Current Intent

The repository now runs from a cleaner long-term package layout while still keeping the low-level pipeline scripts explicit.

That means:

- `src/chemistry_bot/` is the architectural target surface
- public scripts and smoke tests already import through that scaffold
- the main bot, retrieval, and prompt-operations implementation modules now live inside that package

## Low-Level Pipeline Scripts

The low-level textbook pipeline remains explicit here:

- `parse_introchem_xhtml_v3.py`
- `build_introchem_rag_chunks_v3.py`
- `introchem_vector_search.py`

These scripts remain useful because they show the current implementation details behind the higher-level wrapper commands in `scripts/`.
