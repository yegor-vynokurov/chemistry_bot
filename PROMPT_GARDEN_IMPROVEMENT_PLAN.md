# Prompt Garden Improvement Plan

## Purpose

This document describes how Prompt Garden should evolve from a notebook-heavy local workflow into a cleaner and more scalable subsystem for prompt authoring, experiment execution, and answer review.

The main direction is:

- keep the notebook as the authoring and registration surface
- move experiment execution into reproducible scripts
- move comparison and review into a Streamlit application

This is an implementation roadmap, not just a concept note. It should be used as a step-by-step plan that we can execute and mark off over time.

## Current Baseline

Verified repository state on June 25, 2026:

- the tracked control notebook is `prompt_garden/control/prompt_garden_experiments_control.ipynb`
- the main prompt-operations code is `src/chemistry_bot/promptops/garden.py`
- the current evaluation helper is `src/chemistry_bot/promptops/eval.py`
- the workspace already stores prompts, lineage, combos, experiments, and runs under `prompt_garden/`
- the current workflow can register prompt nodes, generate combos, attach combos to experiments, and record experiment results

The current weakness is not the existence of Prompt Garden. The weakness is that authoring, execution, logging, and review are still too tightly mixed together.

## Problems To Solve

- the notebook currently does too many jobs at once
- answer comparison becomes unreliable when an experiment returns many outputs
- run history is hard to inspect because raw outputs and review-oriented artifacts are not clearly separated
- manual review is too dependent on visual scanning and reviewer stamina
- there is no dedicated interface for side-by-side comparison, filtering, or baseline-vs-challenger review
- embeddings and similarity analysis are not yet used as reviewer aids for grouping near-duplicates and spotting outliers
- rerunning only selected combos, cases, or failed items should be easier

## Target Operating Model

Prompt Garden should become a three-surface subsystem.

### Surface 1. Notebook

Keep `prompt_garden/control/prompt_garden_experiments_control.ipynb` as the operator-facing registration and orchestration workspace.

Its future role:

- create root prompts
- create child prompts and preserve lineage
- create few-shot roots and descendants
- create and inspect combos
- generate combo batches
- mark combos to include or skip
- create experiment definitions
- prepare execution commands for scripted runs

Its future non-role:

- it should not be the main place for reviewing 40 plus full model answers
- it should not be the main place for bulky run logging
- it should not be the primary UI for side-by-side answer comparison

### Surface 2. Script Runner

Add a script-driven execution layer for experiments.

Its future role:

- run one experiment or a filtered subset of it
- support resume and rerun behavior
- write structured raw outputs
- write normalized review artifacts
- update experiment summaries

This layer should become the repeatable way to execute Prompt Garden experiments.

### Surface 3. Streamlit Review App

Add a Streamlit review surface for experiment inspection and comparison.

Its future role:

- experiment overview
- answer tables with filters and sorting
- side-by-side comparison
- baseline-vs-challenger review
- text diffs and paragraph-level comparison
- similarity and outlier views powered by embeddings
- reviewer notes and selection of preferred variants

This layer should become the default place for reviewing prompt experiments at human scale.

## Proposed Future Layout

The current repository structure is already close enough that we can extend it without another reset.

Recommended additions:

```text
chemistry_bot/
  PROMPT_GARDEN_IMPROVEMENT_PLAN.md
  apps/
    prompt_garden_review.py
  docs/
    prompt-garden-review.md
  prompt_garden/
    control/
      prompt_garden_experiments_control.ipynb
    cases/
    experiments/
    prompts/
    registry/
    runs/
      raw/
      normalized/
    reports/
    cache/
      embeddings/
  scripts/
    run_prompt_experiment.py
    export_prompt_review_bundle.py
  src/
    chemistry_bot/
      promptops/
        garden.py
        eval.py
        runner.py
        review_store.py
        review_compare.py
        review_metrics.py
        review_embeddings.py
```

Notes:

- `apps/` is the preferred place for the Streamlit entry point because it is an interactive application surface rather than a library module
- `src/chemistry_bot/promptops/` remains the shared library layer behind the notebook, scripts, and Streamlit app
- generated run artifacts should move away from the notebook and into explicit folders with stable contracts

## Artifact Model

Prompt Garden should distinguish four different artifact types.

### 1. Source Artifacts

These are the objects we intentionally author and keep as source-of-truth:

- prompt texts
- node metadata
- combo metadata
- experiment definitions
- named case sets

### 2. Raw Run Artifacts

These capture what actually happened during model execution:

- experiment ID
- combo ID
- case ID
- model name
- request parameters
- fully materialized prompts
- raw model output
- timing metadata
- errors or parse failures

### 3. Normalized Review Artifacts

These transform raw outputs into a comparison-friendly structure:

- parsed answer fields
- normalized text blocks
- answer length statistics
- source usage counts
- evaluation helper outputs
- tags such as refusal, parse_error, or missing_field

### 4. Derived Review Artifacts

These are built from normalized outputs for faster human review:

- experiment summary tables
- ranking tables
- similarity matrices
- cached embeddings
- reviewer notes
- selected winners or preferred variants

## Comparison Model

The review layer should not depend on a single scoring idea. It should support several complementary review modes.

### Structured Field Comparison

Compare answers by explicit fields when the bot returns structured content.

Examples:

- `request_type`
- `certainty`
- `short_answer`
- `explanation`
- `examples`
- `experiment.kind`
- `experiment.reason`
- `source_ids`

### Text Diff Comparison

Support readable text review for humans.

Examples:

- full-answer diff
- field-specific diff
- sentence-level diff
- paragraph-level diff
- highlight insertions, removals, and reordered blocks

### Baseline-Vs-Challenger Comparison

Prompt review is easier when one combo is treated as the reference point.

The review flow should allow:

- selecting one combo as a baseline
- comparing all other combos against that baseline
- surfacing only meaningful deltas
- filtering to cases where answers diverge strongly

### Embedding-Assisted Review

Embeddings should be used as a reviewer aid, not as the definition of answer quality.

Useful roles:

- cluster near-duplicate answers
- surface outlier responses
- estimate semantic similarity when wording differs
- reduce the number of answers that must be inspected line by line

Embeddings should not be treated as a correctness score by themselves.

## Logging And Review Principles

The logging system should be optimized for reproducibility first and readability second.

Working rules:

- keep raw model outputs intact
- never overwrite the only copy of a run
- normalize into separate review artifacts instead of mutating raw outputs
- make all review tables rebuildable from raw or normalized artifacts
- keep large generated review outputs out of Git unless they become intentional curated examples

## Phased Roadmap

### Phase 1. Define Prompt Garden Contracts

Status:
Completed on June 25, 2026.

Objective:
Freeze the data contracts before building new tooling on top of them.

Main tasks:

- [x] define the source-of-truth objects for prompts, combos, experiments, case sets, runs, and reports
- [x] define where raw runs, normalized runs, and derived reports will live
- [x] define minimal required fields for each artifact type
- [x] define stable naming rules for run files, report files, and cache files
- [x] document which Prompt Garden artifacts are tracked in Git and which remain local or generated

Exit condition:
The notebook, runner, and Streamlit app can all rely on the same file and metadata contracts.

Phase 1 result:

- the stable contract document now lives in `docs/prompt-garden-contracts.md`
- the repository now has tracked placeholder directories for `cases/`, `runs/`, `reports/`, and `cache/`
- `.gitignore` now distinguishes tracked contract docs from future generated run and cache artifacts
- Prompt Garden now has an explicit Phase 1 file-and-metadata baseline for the later runner and Streamlit phases

### Phase 2. Slim The Notebook To Authoring And Registration

Status:
Completed on June 25, 2026.

Objective:
Keep the notebook powerful, but focused on authoring and experiment setup instead of heavy review.

Main tasks:

- [x] keep cells for creating prompt roots, child prompts, and few-shot variants
- [x] keep cells for combo creation, combo batch generation, and skip control
- [x] keep cells for experiment creation and experiment membership review
- [x] remove or demote bulky answer-review cells
- [x] add cells that generate the exact script commands needed to run prepared experiments
- [x] ensure notebook imports stay aligned with the package layout under `src/chemistry_bot/`

Exit condition:
The notebook is the control console for prompt lineage and experiment setup, not the main review surface.

Phase 2 result:

- the control notebook has been reduced to an authoring-and-registration workflow
- repeated bulky review cells and hard-coded experiment inspection clutter have been removed
- few-shot registration cells are now idempotent instead of creating accidental duplicates on rerun
- combo selection now has explicit include/skip control before experiment attachment
- the notebook now generates planned runner and Streamlit handoff commands instead of pretending to be the final review UI
- the old inline execution loop remains only as a temporary fallback bridge until the dedicated runner lands

### Phase 3. Introduce The Script Runner

Status:
Completed on June 25, 2026.

Objective:
Make experiment execution reproducible and scriptable.

Main tasks:

- [x] add `scripts/run_prompt_experiment.py` as the public execution command
- [x] add shared execution logic under `src/chemistry_bot/promptops/runner.py`
- [x] support running a whole experiment, one combo, or a selected combo subset
- [x] support filtering by case set or case IDs
- [x] support rerunning only failed items or only missing items
- [x] support resume behavior so interrupted runs can continue safely
- [x] write raw run artifacts and normalized review artifacts during execution

Exit condition:
A Prompt Garden experiment can be executed without relying on notebook cell order.

Phase 3 result:

- the public Prompt Garden runner now exists as `scripts/run_prompt_experiment.py`
- shared execution and artifact-writing logic now lives in `src/chemistry_bot/promptops/runner.py`
- the runner now supports experiment-wide, single-combo, and selected-subset execution
- case-set loading, case filtering, resume-on-missing, rerun-failed, and rerun-all modes are now available from the CLI
- raw run artifacts are now written under `prompt_garden/runs/raw/<experiment_id>/`
- normalized review artifacts are now written under `prompt_garden/runs/normalized/<experiment_id>/`
- a tracked default case set now exists as `prompt_garden/cases/default_chemistry_school_cases_v1.json`

### Phase 4. Normalize Logging And Review Data

Status:
Completed on June 25, 2026.

Objective:
Make run outputs easy to load, compare, and audit.

Main tasks:

- [x] create the `prompt_garden/runs/raw/` contract
- [x] create the `prompt_garden/runs/normalized/` contract
- [x] create summary report outputs under `prompt_garden/reports/`
- [x] define stable answer normalization rules
- [x] preserve prompt lineage and combo metadata inside normalized artifacts
- [x] add a small loader layer that converts normalized outputs into review tables cleanly

Exit condition:
A review tool can load one experiment into a structured table without notebook-specific parsing tricks.

Phase 4 result:

- Prompt Garden now has a dedicated review-store layer in `src/chemistry_bot/promptops/review_store.py`
- raw and normalized run artifacts now carry explicit schema versions, stable artifact paths, and richer prompt lineage snapshots
- normalized artifacts now include field-level normalized text blocks, paragraph splits, comparison text, and comparison hashes
- scripted runs now generate derived summary reports under `prompt_garden/reports/<scope>/`
- review rows, combo summaries, and case summaries can now be loaded directly from normalized artifacts without notebook-specific parsing

### Phase 5. Build The Comparison Engine

Status:
Completed on June 25, 2026.

Objective:
Provide reusable comparison logic before building the UI.

Main tasks:

- [x] add field-level comparison helpers
- [x] add full-text and field-specific diff helpers
- [x] add paragraph-level and sentence-level segmentation helpers
- [x] add baseline-vs-challenger comparison helpers
- [x] add ranking and summary metrics for review tables
- [x] add embedding-based similarity helpers for grouping and outlier detection

Exit condition:
The future Streamlit app can call shared library functions instead of implementing comparison logic inside UI code.

Phase 5 result:

- reusable comparison helpers now live in `src/chemistry_bot/promptops/review_compare.py`
- combo ranking, case difficulty, and baseline summary helpers now live in `src/chemistry_bot/promptops/review_metrics.py`
- lightweight similarity, clustering, and outlier helpers now live in `src/chemistry_bot/promptops/review_embeddings.py`
- the Prompt Garden package now exposes a comparison-engine surface that Phase 6 can call directly from Streamlit

### Phase 6. Build The Streamlit Review App

Status:
Completed on June 25, 2026.

Objective:
Give Prompt Garden a dedicated review surface that scales beyond notebook output cells.

Main tasks:

- [x] add `apps/prompt_garden_review.py`
- [x] add experiment picker and case-set filters
- [x] add sortable overview tables for combos, cases, and metrics
- [x] add side-by-side answer comparison
- [x] add baseline selection and challenger comparison mode
- [x] add diff panels for field and paragraph review
- [x] add similarity and outlier views based on cached embeddings
- [x] add lightweight reviewer notes and preferred-answer marking
- [x] add export options for compact review summaries

Exit condition:
Reviewing 40 or more answers no longer depends on scanning notebook output manually.

Phase 6 result:

- the first dedicated review UI now exists as `apps/prompt_garden_review.py`
- the app now supports experiment-scope selection, execution-signature filtering, case-set filtering, combo and case filters, and score or status filtering
- side-by-side comparison now includes structured field deltas, unified diffs, HTML diffs, and paragraph or sentence alignment views
- baseline-vs-challenger mode now has both combo-level summary rows and case-level detailed deltas
- similarity and outlier review now runs through the Phase 5 comparison engine and can persist cached similarity bundles under `prompt_garden/cache/embeddings/`
- reviewer notes and preferred-answer marking now persist to local report files under `prompt_garden/reports/<scope>/`
- compact review exports are now available directly from the app

### Phase 7. Document The Operator Workflow

Status:
Completed on June 25, 2026.

Objective:
Make the Prompt Garden workflow understandable to a new reviewer or collaborator.

Main tasks:

- [x] document the notebook -> script -> Streamlit workflow in `docs/prompt-garden-review.md`
- [x] update `prompt_garden/README.md` to explain each surface clearly
- [x] document the recommended daily loop for prompt experiments
- [x] document how curated prompt assets differ from generated run artifacts
- [x] document how to rebuild review tables and embedding caches

Exit condition:
A collaborator can understand where to author prompts, where to run experiments, and where to review results.

Phase 7 result:

- the supported operator workflow is now documented in `docs/prompt-garden-review.md`
- `prompt_garden/README.md` now explains the notebook, runner, and Streamlit surfaces explicitly
- the recommended daily Prompt Garden loop is now documented in one place
- curated assets and generated review artifacts are now described separately
- rebuild guidance now exists for raw runs, normalized artifacts, summary reports, review tables, and similarity caches

### Phase 8. Add Light Smoke Tests For Prompt Garden Tooling

Status:
Completed on June 25, 2026.

Objective:
Add confidence checks without turning Prompt Garden into a heavy test burden.

Main tasks:

- [x] add a loader smoke test for normalized review artifacts
- [x] add a diff-helper smoke test
- [x] add a runner dry-run or fixture-based smoke test
- [x] add a Streamlit import smoke test so the review app does not silently break
- [x] add one tiny tracked fixture experiment for tooling verification

Exit condition:
The Prompt Garden authoring, execution, and review surfaces have a minimal confidence layer.

Phase 8 result:

- a tracked fixture workspace now exists under `tests/fixtures/prompt_garden_smoke/`
- Prompt Garden smoke coverage now lives in `tests/test_prompt_garden_smoke.py`
- the smoke suite now covers normalized artifact loading, diff helpers, runner dry-run planning, and Streamlit review-module loading
- a dedicated one-command entry point now exists as `scripts/run_prompt_garden_smoke_tests.py`
- a lightweight live validation experiment was executed as `exp_000008`, producing four review signatures across `phi4-mini` and `gemma4:12b` with and without few-shot prompts
- the Streamlit review app was also started successfully against live Prompt Garden artifacts, confirming that the review surface can boot on the current workspace

## Feature Priority Inside The Streamlit App

The first version of the app should prioritize review value over visual complexity.

High priority:

- experiment selection
- filters
- sortable metrics table
- side-by-side comparison
- baseline-vs-challenger view
- readable text diffs

Medium priority:

- paragraph and sentence segmentation
- source usage visualization
- reviewer notes
- export summary

Later priority:

- embedding plots
- cluster navigation
- heatmaps
- richer annotation workflows

## Non-Goals For This Plan

This plan does not yet include:

- prompt quality optimization itself
- automated prompt selection as a replacement for human review
- a full benchmark suite for Prompt Garden
- migration away from the notebook entirely
- unifying the legacy bot and the RAG-first bot

## Success Criteria

This Prompt Garden improvement pass will be successful when:

- prompt authoring remains easy
- experiment execution becomes reproducible from scripts
- review no longer depends on large notebook output blocks
- answer comparison becomes faster and less error-prone
- baseline combos can be compared cleanly against challengers
- embeddings help reduce reviewer fatigue without pretending to judge correctness alone
- the repository presents Prompt Garden as a serious experimentation subsystem, not a pile of local artifacts
