# Prompt Garden Workspace

Prompt Garden is a first-class subsystem of the project.
It remains a top-level repository directory by design.

This directory is the home of:

- prompt source assets
- prompt lineage and combo relationships
- experiment records
- experiment case sets
- run artifacts
- review reports and caches
- the Streamlit control panel and the notebook power-user surface

## Supported Workflow

Prompt Garden now has a supported three-surface workflow:

1. inspect, author routine prompt assets, manage, and review experiments in the Streamlit panel
2. execute experiments from the runner script
3. use the notebook only when deeper prompt authoring or structural prompt work is needed

The canonical operator guide for that workflow lives in:

- `docs/prompt-garden-review.md`
- `prompt_garden_step_by_step.md`

## Current Streamlit Authoring Scope

The Streamlit panel now supports these routine authoring flows directly:

- `Create Root Prompt`
- `Branch Prompt`
- `Create Combo`

These flows live in the shared bottom authoring workspace inside:

- `Control -> Prompt Workspace`
- `Control -> Combo Explorer`

Prompt inspection, archive, and permanent delete now also live in:

- `Control -> Prompt Workspace`

The `Cleanup` section is now reserved for combo and experiment cleanup only.

The intended prompt review order inside `Prompt Workspace` is:

- read `Prompt Text`
- inspect `Usage & Results`
- branch or archive when the prompt is still historically useful
- use the delete `Danger Zone` only after the blockers are cleared

The current Streamlit scope is intentionally conservative:

- create new prompt assets instead of editing prompts in place
- branch existing prompts instead of changing lineage manually
- create new combos instead of mutating combo membership

The notebook is still the right place for deeper or less common structural edits.

## Structure

- `control/`
  Notebook-based authoring and registration surface for prompt versioning, combo generation, and deeper prompt edits. It is no longer the default day-to-day operator surface.

- `prompts/`
  Prompt texts and few-shot assets used by the bot and by Prompt Garden experiments.

- `registry/`
  Graph-style metadata such as nodes, edges, combos, and compact experiment records.

- `experiments/`
  Full experiment objects and detailed experiment history.

- `cases/`
  Contract-ready location for case-set definitions used by Prompt Garden evaluation workflows.

- `runs/`
  Generated raw and normalized execution artifacts for future scripted experiment runs.

- `reports/`
  Derived review outputs such as summaries and comparison bundles.

- `cache/`
  Rebuildable analysis caches such as embeddings.

## Tracking Policy

At the current cleanup stage:

- `README.md` is tracked
- `control/prompt_garden_experiments_control.ipynb` is tracked
- `curated/` is tracked as the public Prompt Garden baseline
- placeholder contract docs inside `cases/`, `runs/`, `reports/`, and `cache/` are tracked
- the larger runtime workspace remains selectively ignored for now

Later cleanup phases will bring curated prompt assets and selected combo lineage into Git more deliberately.

## Curated Baseline

The tracked public baseline currently lives in:

- `curated/`

This curated subset includes:

- `combo_000014`
- `sys_000006`
- `usr_000006`
- `fsh_000002`

Use this subset as the reviewable and evolvable Prompt Garden baseline for the current school-focused chemistry bot.

## Improvement Roadmap

The detailed modernization roadmap for Prompt Garden now lives in:

- `PROMPT_GARDEN_IMPROVEMENT_PLAN.md` in local_sandbox

That plan defines the next intended architecture:

- the Streamlit panel becomes the main operator surface
- experiment execution moves to scripts
- the notebook remains the power-user authoring surface

Current notebook status:

- direct editing of existing prompt text stays in the notebook
- parent reassignment or lineage surgery stays in the notebook
- existing combo membership edits stay in the notebook
- few-shot-heavy and deeper structural authoring stays in the notebook
- scripted execution now lives in `scripts/run_prompt_experiment.py`
- scripted execution now writes raw runs, normalized review artifacts, and summary reports with stable schemas
- routine prompt creation, branching, combo creation, experiment setup, and review now belong in the Streamlit panel
- the inline execution block is now only a legacy bridge, not the recommended path

Tracked Prompt Garden case sets:

- `cases/default_chemistry_school_cases_v1.json`

The stable file and metadata contracts for that architecture live in:

- `docs/prompt-garden-contracts.md`

The Phase 4 review-loading layer now lives in:

- `src/chemistry_bot/promptops/review_store.py`

The Phase 5 comparison engine now lives in:

- `src/chemistry_bot/promptops/review_compare.py`
- `src/chemistry_bot/promptops/review_metrics.py`
- `src/chemistry_bot/promptops/review_embeddings.py`

The dedicated Phase 6 review surface now exists as:

- `apps/prompt_garden_review.py`

## Surface Roles

### Control And Analysis App

Use `apps/prompt_garden_review.py` for:

- workspace overview
- prompt, combo, and experiment inspection
- root prompt creation
- prompt branching
- combo creation with preview and duplicate guardrails
- experiment editing and combo attachment management
- runner command generation with dry-run preview
- answer browsing and experiment review
- prompt similarity inspection
- experiment notes, finalization, and cleanup

This is the default day-to-day entry point for Prompt Garden operations.

### Runner

Use `scripts/run_prompt_experiment.py` for:

- reproducible experiment execution
- resume and rerun behavior
- writing raw and normalized artifacts
- generating summary reports

### Notebook

Use `control/prompt_garden_experiments_control.ipynb` for:

- direct editing of existing prompts
- parent or lineage corrections
- few-shot registration and heavier few-shot curation
- bulk combo generation and deeper combo curation work
- power-user edits that are still faster in notebook form

Do not treat the notebook as the main experiment-control or answer-review surface.

## Recommended Daily Loop

The normal Prompt Garden loop is:

1. open `apps/prompt_garden_review.py` and inspect the current workspace
2. create a root prompt, branch a prompt, or create a combo directly in the Streamlit panel when the change is routine
3. switch to the notebook only if prompt text or combo structure needs deeper authoring work
4. return to the Streamlit panel and attach combos or update experiment metadata
5. generate the runner command from the panel
6. run the experiment with `scripts/run_prompt_experiment.py`
7. review answers, notes, and scores in `apps/prompt_garden_review.py`
8. finalize, archive, or branch again depending on what the experiment showed

## Curated Vs Generated Artifacts

Treat these as curated or authored assets:

- `curated/`
- `cases/`
- the tracked control notebook
- documentation and contracts

Treat these as generated or rebuildable artifacts:

- `runs/raw/`
- `runs/normalized/`
- `reports/`
- `cache/`

The generated zones support review and reproducibility, but they are not the primary source-of-truth for prompt authoring.

## Rebuild Guidance

Rebuild raw and normalized experiment artifacts by rerunning:

- `scripts/run_prompt_experiment.py`

Rebuild review tables by reopening the Streamlit app after new normalized artifacts exist.

If you made changes through the notebook and they do not appear immediately in the panel:

1. reopen the app if needed
2. press `Reload Cached Data`
3. return to the relevant explorer or experiment view

Rebuild similarity caches by deleting the relevant file under:

- `cache/embeddings/`

and then recomputing similarity in the Streamlit panel.
