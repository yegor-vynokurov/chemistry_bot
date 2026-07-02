# Scripts

This folder contains stable, user-facing entry points that wrap the current experimental modules.

At the current cleanup stage, these scripts provide a cleaner public interface without forcing a large internal refactor.

They now sit alongside a package-oriented scaffold in `src/chemistry_bot/`, while continuing to protect the currently working execution path.

Current scripts:

- `run_prompt_experiment.py`
  Prompt Garden experiment runner that executes selected combos and cases, supports resume modes, writes raw plus normalized run artifacts, and emits derived summary reports for review tooling.

- `run_chemistry_bot_rag.py`
  Recommended launcher for the current RAG-enabled chemistry bot.

- `inspect_introchem_rag.py`
  Quick retrieval inspection tool for the Introductory Chemistry index.

- `build_introchem_rag_bundle.py`
  One public wrapper command for the current textbook -> normalized chapters -> RAG chunks -> Chroma index workflow.

- `run_core_smoke_tests.py`
  One-command smoke-test runner for retrieval connectivity and end-to-end QA.

- `run_prompt_garden_smoke_tests.py`
  One-command smoke-test runner for Prompt Garden fixtures, control and analysis loaders, diff helpers, runner command planning, prompt similarity, and cleanup safety flows.

Related interactive surface:

- `apps/prompt_garden_review.py`
  Streamlit control and analysis app for Prompt Garden workspace browsing, experiment management, runner command generation, answer review, prompt similarity, notes, and cleanup.
