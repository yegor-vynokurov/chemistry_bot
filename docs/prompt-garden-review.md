# Prompt Garden Operator Workflow

## Purpose

This document describes the supported operator workflow for Prompt Garden.

The workflow is intentionally split across three surfaces:

1. the Streamlit panel for control and analysis
2. the runner script for reproducible execution
3. the notebook for deeper prompt authoring when needed

Prompt Garden is now meant to be operated through that sequence rather than through one large notebook session.

## Supported Surfaces

### 1. Streamlit Control And Analysis Panel

Primary entry point:

- `apps/prompt_garden_review.py`

Use the app for:

- workspace overview
- prompt, combo, and experiment inspection
- prompt text review, usage review, archive, and delete from `Prompt Workspace`
- root prompt creation
- prompt branching
- combo creation with prompt-set preview
- experiment editing and combo attachment management
- runner command generation with dry-run preview
- answer browsing
- prompt similarity inspection
- reviewer notes, experiment finalization, and cleanup

This is the default operator surface for day-to-day Prompt Garden work.

### Prompt Workspace And Cleanup Roles

Inside `Control`, use:

- `Prompt Workspace` for prompt inspection, branching, archive, and permanent delete
- `Combo Explorer` for combo inspection and creation
- `Experiments` for experiment editing and attachment management
- `Cleanup` only for combo and experiment cleanup flows

That keeps prompt lifecycle work in one place instead of splitting prompt review and prompt delete across multiple tabs.

### Prompt-First Inspection Loop

When reviewing one prompt in `Prompt Workspace`, the intended order is:

1. read `Prompt Text` first
2. inspect `Usage & Results` to understand combo coverage, model coverage, and top-scoring outcomes
3. use `Archive Prompt` as the normal retirement action when the prompt should stay in history
4. use `Delete Prompt` only from the `Danger Zone` after the blockers and recommendations say it is safe

### Current Streamlit Authoring Scope

The current Streamlit authoring surface is intentionally limited to safe create-style actions:

- create a new root prompt
- create a new branch from an existing prompt
- create a new combo from selected prompt roles

The current non-goals are:

- direct editing of an existing prompt in place
- changing the parent of an existing prompt
- editing the membership of an existing combo

That means the preferred model is:

- branch instead of overwrite
- create instead of mutate
- archive old assets instead of reshaping history

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

### 3. Notebook Authoring Surface

Primary entry point:

- `prompt_garden/control/prompt_garden_experiments_control.ipynb`

Use the notebook for:

- direct editing of prompt text
- parent or lineage corrections
- deeper few-shot maintenance
- bulk combo generation or structural curation
- deep authoring work that is still more convenient in notebook form

Do not treat the notebook as the main experiment-control or answer-review surface.

## Recommended Daily Loop

The standard working loop is:

1. Open the Streamlit app and inspect the current workspace or experiment.
2. Create a root prompt, branch a prompt, or create a combo in Streamlit when the change is routine.
3. Switch to the notebook only when prompt text, few-shot assets, or combo structure need deeper authoring work.
4. Return to the Streamlit app and attach combos or update experiment metadata.
5. Generate the runner command from the Streamlit control panel.
6. Execute the experiment from the script runner.
7. Review the outputs in the Streamlit app.
8. Record reviewer notes, final summary text, and final scores.
9. Decide whether to archive, branch further, or rerun only a smaller subset.

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

### Launch the control panel

```powershell
.\.venv\Scripts\streamlit.exe run apps\prompt_garden_review.py
```

After notebook-side changes, use the `Reload Cached Data` button in the app before checking the updated explorers or experiment views.

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
2. optionally inspect the target composition in the Streamlit control panel
3. rerun `scripts/run_prompt_experiment.py`
4. use `--run-mode all` when you want a fresh rebuild for the selected subset
5. use `--run-mode missing` to resume interrupted work
6. use `--run-mode failed` to focus on previously failed items

### Summary Reports

Summary reports under `prompt_garden/reports/<scope>/` are generated automatically by the runner after execution.

If you rerun the selected scope, the runner will emit a fresh summary report file without deleting the earlier one.

### Control And Review Tables

The Streamlit app rebuilds its control and review tables directly from normalized artifacts by using:

- `src/chemistry_bot/promptops/review_store.py`
- `src/chemistry_bot/promptops/review_compare.py`
- `src/chemistry_bot/promptops/review_metrics.py`

You do not need to maintain manual notebook parsing code for those tables anymore.

### Similarity Caches

Similarity bundles can be cached under:

- `prompt_garden/cache/embeddings/`

The Streamlit panel can compute a similarity bundle from the current filtered set and save it to cache.

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

- if you are inspecting, managing, doing routine prompt or combo creation, cleaning up, or reviewing an existing experiment, use the Streamlit app
- if you are invoking the model, use the runner
- if you are directly rewriting prompt text or doing deeper combo curation, use the notebook

That split is the current supported operator model for Prompt Garden.
