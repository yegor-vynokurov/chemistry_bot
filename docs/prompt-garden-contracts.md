# Prompt Garden Contracts

## Purpose

This document freezes the Phase 1 data and file contracts for Prompt Garden.

It defines:

- the source-of-truth objects used by the subsystem
- where each artifact type lives
- the minimal required fields for each artifact
- stable naming rules for future runner and review outputs
- which Prompt Garden artifacts are tracked in Git and which remain local or generated

This contract is intentionally aligned with the current implementation in `src/chemistry_bot/promptops/garden.py` and with the future direction described in `PROMPT_GARDEN_IMPROVEMENT_PLAN.md`.

## Contract Status

Status:
Approved for the current cleanup pass on June 25, 2026.

Working rule:

- new Prompt Garden tooling should follow these contracts unless we intentionally version and revise them
- future phases may extend the schema, but should avoid silently breaking these minimal fields
- backward-compatible compatibility layers are acceptable while notebook and runner workflows overlap

## Directory Contract

Prompt Garden now has four artifact zones.

### 1. Authoring And Registry Zone

These files define prompt lineage and experiment intent.

- `prompt_garden/prompts/`
- `prompt_garden/registry/`
- `prompt_garden/experiments/`
- `prompt_garden/cases/`

### 2. Raw Execution Zone

These files capture what the model actually received and returned.

- `prompt_garden/runs/raw/`

### 3. Normalized Review Zone

These files convert raw executions into review-friendly answer records.

- `prompt_garden/runs/normalized/`

### 4. Derived Analysis Zone

These files are rebuilt from raw or normalized artifacts.

- `prompt_garden/reports/`
- `prompt_garden/cache/`
- `prompt_garden/cache/embeddings/`

## Source-Of-Truth Objects

Prompt Garden should treat the following as its main logical objects.

### Prompt Node

Primary storage:

- prompt text file: `prompt_garden/prompts/<prompt_type>/<prompt_id>.md`
- metadata row: `prompt_garden/registry/nodes.jsonl`

Minimal required fields in `nodes.jsonl`:

- `id`
- `type`
- `tree_id`
- `path`
- `parent_id`
- `branch`
- `title`
- `created_at`
- `tags`

Recommended fields:

- `metadata`
- `stats`

Notes:

- the prompt text file and the node row together form the source-of-truth prompt object
- `path` should stay relative to the Prompt Garden root
- `metadata.content_hash` should be treated as the stable content fingerprint when available

### Lineage Edge

Primary storage:

- `prompt_garden/registry/edges.jsonl`

Minimal required fields:

- `from`
- `to`
- `kind`
- `created_at`
- `metadata`

Notes:

- this object is required to preserve prompt lineage, context variants, and experiment-to-combo relations
- even though the Phase 1 scope focuses on prompts, combos, experiments, case sets, runs, and reports, edges remain a mandatory supporting contract

### Combo

Primary storage:

- `prompt_garden/registry/combos.jsonl`

Minimal required fields:

- `id`
- `title`
- `prompt_ids`
- `status`
- `test_status`
- `notes`
- `tags`
- `created_at`

Recommended fields:

- `score`
- `metadata`
- `stats`

Notes:

- `prompt_ids` must map runtime roles such as `system`, `user`, `fewshot`, or other supported roles to existing prompt node IDs
- `metadata.combo_key` is the preferred stable deduplication fingerprint

### Experiment Node

Primary storage:

- `prompt_garden/registry/experiment_nodes.jsonl`

Minimal required fields:

- `id`
- `type`
- `name`
- `goal`
- `hypothesis`
- `notes`
- `tags`
- `status`
- `created_at`
- `path`

Recommended fields:

- `updated_at`

Notes:

- the experiment node is the compact index row for discovery and filtering
- `type` should stay equal to `experiment`

### Experiment Object

Primary storage:

- `prompt_garden/experiments/<experiment_id>.json`

Minimal required fields:

- all required experiment-node fields
- `combo_ids`
- `results`
- `summary`
- `subjective_summary`
- `metadata`

Recommended fields:

- `updated_at`
- `final_result_text`
- `final_subject_score`

Notes:

- this is the current detailed source-of-truth object for experiment state
- `metadata.schema_version` should be present
- `results` may continue to hold compact per-combo outcomes during the transition to the future runner

### Compact Experiment Result Row

Primary storage:

- `prompt_garden/registry/experiments.jsonl`

Minimal required fields:

- `experiment_id`
- `combo_id`
- `score`
- `result_text`
- `metrics`
- `created_at`

Recommended fields:

- `subject_score`

Notes:

- this file remains a compatibility layer and compact summary index
- it is not the long-term replacement for normalized review artifacts

### Case Set

Primary storage:

- `prompt_garden/cases/<case_set_id>.json`

Minimal required top-level fields:

- `id`
- `name`
- `description`
- `audience`
- `language`
- `created_at`
- `tags`
- `cases`

Minimal required fields for each case item:

- `id`
- `question`

Recommended case fields:

- `expected_request_type`
- `expected_experiment_kind`
- `forbidden_phrases`
- `forbid_concentration_patterns`
- `protocol_context`
- `notes`
- `metadata`

Notes:

- one file should represent one named case set
- early case sets may be migrated from `DEFAULT_CHEMISTRY_CASES` in `src/chemistry_bot/promptops/eval.py`
- this contract freezes the intended file location even before that migration happens

### Raw Run Artifact

Primary storage:

- `prompt_garden/runs/raw/<scope>/<run_file>.json`

Minimal required fields:

- `id`
- `combo_id`
- `experiment_id`
- `task`
- `model`
- `input_data`
- `output_data`
- `validation_ok`
- `error`
- `metrics`
- `created_at`

Recommended fields:

- `case_set_id`
- `case_id`
- `scope`
- `schema_version`
- `prompt_snapshot`
- `artifact_paths`
- `request_params`
- `timings`
- `runner_version`
- `execution`
- `rag_context`

Notes:

- this contract is intentionally compatible with the current `add_run(...)` shape in `garden.py`
- `input_data` should contain the question or task inputs needed to reproduce the call
- `output_data` should preserve the raw model output or parsed object exactly as seen at execution time
- `schema_version` is now `prompt-garden-raw-run-v1`
- `artifact_paths.raw` and `artifact_paths.normalized` should store stable repo-relative references
- `prompt_snapshot` should preserve the executed combo state, prompt lineage, and optional few-shot linkage at execution time

### Normalized Review Artifact

Primary storage:

- `prompt_garden/runs/normalized/<scope>/<normalized_file>.json`

Minimal required fields:

- `id`
- `raw_run_id`
- `combo_id`
- `experiment_id`
- `model`
- `task`
- `created_at`
- `question`
- `parsed_answer`
- `normalized_text_blocks`
- `metrics`
- `tags`

Recommended fields:

- `case_set_id`
- `case_id`
- `scope`
- `schema_version`
- `source_ids`
- `source_usage`
- `request_type`
- `experiment_kind`
- `answer_lengths`
- `review_flags`
- `artifact_paths`
- `prompt_snapshot`

Notes:

- normalized artifacts are review-first projections of raw runs
- they should never be the only preserved copy of a run
- `normalized_text_blocks` should support field-level and paragraph-level comparison without re-parsing notebook output
- `schema_version` is now `prompt-garden-normalized-review-v1`
- `normalized_text_blocks.normalization_version` is now `prompt-review-normalization-v1`
- `normalized_text_blocks` should keep a stable field order for `short_answer`, `explanation`, `examples`, `experiment_reason`, `experiment_questions`, and `raw_output`
- `normalized_text_blocks.comparison_text` is the canonical concatenated text for future diff and embedding workflows
- `comparison_hash` should be derived from that canonical comparison text
- `prompt_snapshot` should include both the base combo and the active combo when contextual prompt materialization changes the executed prompt IDs

### Derived Report Artifact

Primary storage:

- `prompt_garden/reports/<scope>/<report_file>`

Minimal required fields for JSON reports:

- `id`
- `report_kind`
- `scope`
- `created_at`
- `source_run_count`
- `source_normalized_count`
- `artifacts`

Recommended fields:

- `baseline_combo_id`
- `summary_metrics`
- `filters`
- `context`
- `review_notes`

Notes:

- reports are derived artifacts, not the primary system of record
- `artifacts` should list the normalized or raw files used to build the report
- the current summary-report schema version is `prompt-garden-summary-report-v1`
- the current summary report format includes top-level `summary_metrics`, `combo_rows`, and `case_rows`

## Stable Answer Normalization Rules

Prompt Garden now has explicit review-text normalization rules.

Current rules:

- convert `None` values to an empty string
- normalize line endings to LF
- trim trailing whitespace per line
- collapse repeated spaces and tabs inside a line
- collapse three or more blank lines into a double line break
- trim outer whitespace
- split paragraphs on blank lines
- preserve both the original text and the normalized text for each review block

These rules apply to:

- `short_answer`
- `explanation`
- `examples`
- `experiment.reason`
- `experiment.questions`
- raw output text

The normalized review artifact should store:

- per-field normalized block payloads
- paragraph lists for future paragraph-level comparison
- a canonical concatenated `comparison_text`
- a deterministic `comparison_hash`

## Naming Rules

### Identifier Rules

The current ID rules remain the baseline:

- prompt nodes: `<prefix>_<6-digit number>` such as `sys_000006`
- combos: `combo_<6-digit number>`
- experiments: `exp_<6-digit number>`
- runs: `run_<6-digit number>`

New case-set IDs should use stable lowercase slugs, for example:

- `school_core_safety_v1`
- `theory_basics_grade9_v1`

### File Rules

Use ASCII filenames only.

Use these file rules:

- prompt text: `<prompt_id>.md`
- experiment object: `<experiment_id>.json`
- case set: `<case_set_id>.json`
- raw run file: `<run_id>__<combo_id>__<case_or_task_slug>.json`
- normalized run file: `<run_id>__<combo_id>__<case_or_task_slug>.json`
- report file: `<report_kind>__<timestamp>.json` or `<report_kind>__<timestamp>.md`
- embedding cache file: `<scope>__<embedding_model>__<content_kind>.json`

### Scope Rules

For raw runs, normalized runs, and reports, `<scope>` should normally be:

- the experiment ID, such as `exp_000007`, when the run belongs to an experiment
- `_adhoc`, when the run is exploratory and not attached to an experiment

### Timestamp Rules

Use filesystem-safe timestamps in filenames:

- `YYYYMMDDTHHMMSS`

Example:

- `summary__20260625T184500.json`

Avoid colons in filenames because the project is developed on Windows.

## Git Tracking Policy

### Tracked In Git

The repository should track:

- `prompt_garden/README.md`
- `prompt_garden/control/prompt_garden_experiments_control.ipynb`
- `prompt_garden/curated/`
- contract and placeholder documentation for `prompt_garden/cases/`, `prompt_garden/runs/`, `prompt_garden/reports/`, and `prompt_garden/cache/`
- curated case sets when we intentionally promote them into the public repository story

### Currently Local Or Selectively Ignored

The repository should currently keep local or selectively ignored:

- `prompt_garden/prompts/`
- `prompt_garden/registry/`
- `prompt_garden/experiments/`
- raw run outputs under `prompt_garden/runs/raw/`
- normalized run outputs under `prompt_garden/runs/normalized/`
- generated reports under `prompt_garden/reports/`
- embedding and analysis caches under `prompt_garden/cache/`

### Rule Of Thumb

Track:

- curated exemplars
- contracts
- notebooks that define the supported workflow
- compact artifacts that strengthen the portfolio story

Do not track by default:

- bulky execution history
- rebuildable review outputs
- caches
- transient experiment noise

## Compatibility Note

The current notebook and `garden.py` implementation may continue to write compact results into:

- `prompt_garden/registry/experiments.jsonl`
- `prompt_garden/registry/runs.jsonl`
- `prompt_garden/experiments/<experiment_id>.json`

That is acceptable during the transition.

Future runner and review tooling should gradually add:

- file-based raw runs under `prompt_garden/runs/raw/`
- file-based normalized review artifacts under `prompt_garden/runs/normalized/`
- derived report bundles under `prompt_garden/reports/`

without breaking the ability to read older experiment history.
