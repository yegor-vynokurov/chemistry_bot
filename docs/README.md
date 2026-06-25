# Documentation

This folder contains supporting project documentation that should not crowd the repository root.

Current documents:

- [prompt-garden-review.md](prompt-garden-review.md)
  Supported notebook -> runner -> Streamlit operator workflow for Prompt Garden, including the daily loop and rebuild guidance.

- [prompt-garden-contracts.md](prompt-garden-contracts.md)
  Stable file and metadata contracts for Prompt Garden authoring, runs, normalized review artifacts, summary reports, and answer-normalization rules.

- [rag-pipeline.md](rag-pipeline.md)
  Low-level textbook-to-RAG pipeline commands and notes for the current implementation.

- [data-sources.md](data-sources.md)
  Corpus provenance, official source links, and current license/redistribution notes.

- [testing.md](testing.md)
  Core smoke-test workflow for retrieval connectivity and the main QA path.

- [repository-layout.md](repository-layout.md)
  Current near-final repository layout and the role of the new package scaffold.

Related public entry points:

- [../scripts/README.md](../scripts/README.md)
  Recommended launch and inspection scripts for the current experimental architecture.

- [../apps/prompt_garden_review.py](../apps/prompt_garden_review.py)
  Dedicated Streamlit review surface for Prompt Garden experiments.

This folder is now the main home for human-facing repository documentation outside the root README.
