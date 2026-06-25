# Raw Run Artifacts

This directory is reserved for raw Prompt Garden execution records.

Expected path pattern:

- `raw/<scope>/<run_id>__<combo_id>__<case_or_task_slug>.json`

These files should preserve:

- what inputs were sent
- which combo and model were used
- what raw output came back
- whether validation succeeded
- timing and error metadata
- a stable `schema_version`
- repo-relative `artifact_paths`
- the executed `prompt_snapshot`, including combo lineage and few-shot linkage

These artifacts are generated and should not be tracked by default.
