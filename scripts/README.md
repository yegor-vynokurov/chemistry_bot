# Scripts

This folder contains stable, user-facing entry points that wrap the current experimental modules.

At the current cleanup stage, these scripts provide a cleaner public interface without forcing a large internal refactor.

They now sit alongside a package-oriented scaffold in `src/chemistry_bot/`, while continuing to protect the currently working execution path.

Current scripts:

- `run_chemistry_bot_rag.py`
  Recommended launcher for the current RAG-enabled chemistry bot.

- `inspect_introchem_rag.py`
  Quick retrieval inspection tool for the Introductory Chemistry index.

- `build_introchem_rag_bundle.py`
  One public wrapper command for the current textbook -> normalized chapters -> RAG chunks -> Chroma index workflow.

- `run_core_smoke_tests.py`
  One-command smoke-test runner for retrieval connectivity and end-to-end QA.
