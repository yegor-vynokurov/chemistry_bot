# Normalized Review Artifacts

This directory is reserved for normalized Prompt Garden review records.

Expected path pattern:

- `normalized/<scope>/<run_id>__<combo_id>__<case_or_task_slug>.json`

These files should contain:

- parsed answer fields
- normalized text blocks for comparison
- compact review metrics
- tags and flags useful for filtering
- canonical `comparison_text` and `comparison_hash`
- prompt lineage metadata via `prompt_snapshot`
- stable repo-relative `artifact_paths`

These artifacts are rebuildable from raw runs and should not be tracked by default.
