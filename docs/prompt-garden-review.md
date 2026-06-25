# Prompt Garden Review Workflow

## Purpose

This document describes the supported operator workflow for Prompt Garden.

The workflow is intentionally split across three surfaces:

1. the notebook for authoring and experiment setup
2. the runner script for reproducible execution
3. the Streamlit app for human-scale review

Prompt Garden is now meant to be operated through that sequence rather than through one large notebook session.

## Supported Surfaces

### 1. Notebook

Primary entry point:

- `prompt_garden/control/prompt_garden_experiments_control.ipynb`

Use the notebook for:

- creating root prompts
- creating child prompts and preserving lineage
- maintaining few-shot prompt variants
- generating and selecting combos
- defining experiment membership
- preparing exact runner commands

Do not treat the notebook as the main answer-review surface.
Large comparison work now belongs in the Streamlit app.

### 2. Script Runner

Primary entry point:

- `scripts/run_prompt_experiment.py`

Use the runner for:

- executing one experiment reproducibly
- resuming interrupted runs
- rerunning only failed or missing cases
- running only selected combos or cases
- writing raw artifacts, normalized artifacts, and summary reports

This is the supported execution path for Prompt Garden experiments.

### 3. Streamlit Review App

Primary entry point:

- `apps/prompt_garden_review.py`

Use the app for:

- overview tables
- side-by-side answer comparison
- baseline-vs-challenger review
- similarity and outlier inspection
- reviewer notes and preferred-answer marking
- compact review exports

This is the supported answer-review surface once model outputs exist.

## Recommended Daily Loop

The standard working loop is:

1. Open the notebook and author or update prompts.
2. Generate or curate combos in the notebook.
3. Attach the selected combos to an experiment.
4. Copy the generated runner command or build one manually.
5. Execute the experiment from the script runner.
6. Open the Streamlit app and review the outputs.
7. Record reviewer notes and preferred answers.
8. Decide whether to keep the current prompt variants, branch them further, or rerun only a smaller subset.

That loop keeps authoring, execution, and review separate enough that each stage stays reproducible.

## Canonical Commands

### Dry-run an experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_experiment.py `
  --experiment-id exp_000007 `
  --dry-run
```

### Run an experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_experiment.py `
  --experiment-id exp_000007 `
  --model phi4-mini
```

### Rerun only one combo and one case

```powershell
.\.venv\Scripts\python.exe scripts\run_prompt_experiment.py `
  --experiment-id exp_000007 `
  --only-combo combo_000014 `
  --only-case-id covalent_bond_theory `
  --run-mode all
```

### Launch the review app

```powershell
.\.venv\Scripts\streamlit.exe run apps\prompt_garden_review.py
```

## Source Of Truth Vs Generated Artifacts

Prompt Garden now distinguishes between authored assets and generated review data.

### Authoritative Source Artifacts

Treat these as source-of-truth:

- `prompt_garden/prompts/`
- `prompt_garden/registry/`
- `prompt_garden/experiments/`
- `prompt_garden/cases/`

These define prompt lineage, combo structure, experiment intent, and named evaluation cases.

### Curated Public Assets

Treat this as the tracked public baseline:

- `prompt_garden/curated/`

This folder contains intentionally promoted prompt and combo examples that help tell the portfolio story.
It is not the full local workspace.

### Generated Execution And Review Artifacts

Treat these as rebuildable outputs:

- `prompt_garden/runs/raw/`
- `prompt_garden/runs/normalized/`
- `prompt_garden/reports/`
- `prompt_garden/cache/`

These are not the primary system of record for prompt authoring.
They are execution history and reviewer aids.

## What To Track In Git

Track:

- workflow documents
- contract documents
- the control notebook
- curated prompt examples
- intentionally promoted case sets

Do not track by default:

- full local prompt workspace history
- raw run logs
- normalized answer artifacts
- derived report bundles
- embedding and similarity caches
- reviewer noise from transient experiments

## How Review Data Is Rebuilt

### Raw And Normalized Artifacts

Raw and normalized artifacts are created by the runner.

If you need to rebuild them for one scope:

1. choose the experiment and subset in the notebook or manually
2. rerun `scripts/run_prompt_experiment.py`
3. use `--run-mode all` when you want a fresh rebuild for the selected subset
4. use `--run-mode missing` to resume interrupted work
5. use `--run-mode failed` to focus on previously failed items

### Summary Reports

Summary reports under `prompt_garden/reports/<scope>/` are generated automatically by the runner after execution.

If you rerun the selected scope, the runner will emit a fresh summary report file without deleting the earlier one.

### Review Tables

The Streamlit app rebuilds its overview tables directly from normalized artifacts by using:

- `src/chemistry_bot/promptops/review_store.py`
- `src/chemistry_bot/promptops/review_compare.py`
- `src/chemistry_bot/promptops/review_metrics.py`

You do not need to maintain manual notebook parsing code for those tables anymore.

### Similarity Caches

Similarity bundles can be cached under:

- `prompt_garden/cache/embeddings/`

The review app can compute a similarity bundle from the current filtered set and save it to cache.

If the cache becomes stale:

1. delete the corresponding cache file in `prompt_garden/cache/embeddings/`
2. reopen the app or recompute similarity in the app
3. save a fresh cache bundle if desired

Because the current similarity backend is deterministic, deleting and rebuilding the cache is safe.

## Notes And Preferred Answers

Reviewer notes are now stored per scope under:

- `prompt_garden/reports/<scope>/review_notes.json`

Use these notes for:

- recording why one combo is preferred
- flagging failure patterns
- leaving short guidance for the next review pass

These notes are local workflow artifacts, not prompt source-of-truth objects.

## Practical Role Split

When in doubt, use this rule:

- if you are changing prompt content or combo membership, use the notebook
- if you are invoking the model, use the runner
- if you are comparing answers, use the Streamlit app

That split is the current supported operator model for Prompt Garden.
