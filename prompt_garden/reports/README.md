# Prompt Garden Reports

This directory is reserved for derived review outputs.

Examples:

- experiment summaries
- baseline-vs-challenger comparison bundles
- compact exports for reviewer decisions

Current Phase 4 output:

- `reports/<scope>/summary__<timestamp>.json`

That summary report is derived from normalized artifacts and currently contains:

- top-level summary metrics
- combo-level summary rows
- case-level summary rows
- filter and execution context used to build the report

Reports are derived artifacts.
They should be rebuildable from normalized runs and kept local unless a specific report becomes a curated portfolio artifact.
