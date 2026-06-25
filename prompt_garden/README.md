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
- the operator-facing notebook control surface

## Supported Workflow

Prompt Garden now has a supported three-surface workflow:

1. author and register prompts in the notebook
2. execute experiments from the runner script
3. review outputs in the Streamlit app

The canonical operator guide for that workflow lives in:

- `docs/prompt-garden-review.md`

## Structure

- `control/`
  Notebook-based authoring and registration surface for prompt versioning, combo management, and experiment setup. It is no longer intended to be the main large-scale answer-review UI.

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

- `PROMPT_GARDEN_IMPROVEMENT_PLAN.md`

That plan defines the next intended architecture:

- the notebook remains the authoring and registration surface
- experiment execution moves to scripts
- large-scale answer comparison moves to a Streamlit review app

Current notebook status:

- prompt authoring and branching stay in the notebook
- combo generation, selection, and experiment registration stay in the notebook
- scripted execution now lives in `scripts/run_prompt_experiment.py`
- scripted execution now writes raw runs, normalized review artifacts, and summary reports with stable schemas
- bulky answer comparison has been demoted
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

### Notebook

Use `control/prompt_garden_experiments_control.ipynb` for:

- prompt creation and branching
- few-shot registration
- combo generation and selection
- experiment setup
- preparing runner commands

### Runner

Use `scripts/run_prompt_experiment.py` for:

- reproducible experiment execution
- resume and rerun behavior
- writing raw and normalized artifacts
- generating summary reports

### Review App

Use `apps/prompt_garden_review.py` for:

- overview tables
- answer comparison
- baseline-vs-challenger review
- similarity and outlier inspection
- reviewer notes and export bundles

## Recommended Daily Loop

The normal Prompt Garden loop is:

1. update prompts or few-shot assets in the notebook
2. generate or choose combos in the notebook
3. attach combos to an experiment
4. run the experiment with `scripts/run_prompt_experiment.py`
5. review answers in `apps/prompt_garden_review.py`
6. record preferred answers and reviewer notes
7. branch prompts again or rerun a smaller subset

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

Rebuild similarity caches by deleting the relevant file under:

- `cache/embeddings/`

and then recomputing similarity in the review app.
