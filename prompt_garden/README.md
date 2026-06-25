# Prompt Garden Workspace

Prompt Garden is a first-class subsystem of the project.
It remains a top-level repository directory by design.

This directory is the home of:

- prompt source assets
- prompt lineage and combo relationships
- experiment records
- the operator-facing notebook control surface

## Structure

- `control/`
  Notebook-based control surface for prompt versioning, combo management, and experiment workflows.

- `prompts/`
  Prompt texts and few-shot assets used by the bot and by Prompt Garden experiments.

- `registry/`
  Graph-style metadata such as nodes, edges, combos, and compact experiment records.

- `experiments/`
  Full experiment objects and detailed experiment history.

## Tracking Policy

At the current cleanup stage:

- `README.md` is tracked
- `control/prompt_garden_experiments_control.ipynb` is tracked
- `curated/` is tracked as the public Prompt Garden baseline
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
