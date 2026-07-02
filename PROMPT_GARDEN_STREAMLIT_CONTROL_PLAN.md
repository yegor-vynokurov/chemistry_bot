# Prompt Garden Streamlit Control Plan

Last updated: 2026-06-26
Owner: Codex
Status: Completed

## Objective

Move the operator-facing Prompt Garden workflow from the notebook into a Streamlit control panel without breaking the current `notebook -> runner -> review` architecture during the transition.

The new Streamlit surface should let the user:

- inspect prompts, combos, and experiments
- understand how prompts, combos, and experiments are connected
- create and edit experiments
- attach and detach combos from experiments
- generate a ready-to-run command for `scripts/run_prompt_experiment.py`
- inspect experiment outputs answer by answer
- leave post-run notes and finalize experiments
- clean up old prompts, combos, and experiments safely
- inspect prompt similarity using the same embedding-style tooling now used for answer similarity

## Initial Scope

Included in this migration:

- prompt, combo, and experiment browsing
- experiment creation and metadata editing
- experiment composition overview
- command generation instead of in-app execution
- answer browsing for completed experiments
- prompt similarity inspection
- safe cleanup actions with dependency preview

Explicitly out of scope for the first implementation pass:

- prompt-to-prompt text diff UI
- answer-to-answer diff UI with highlighted text
- direct long-running experiment execution inside Streamlit
- graph rendering that depends on heavy extra libraries

## Success Criteria

The migration is complete for this phase when all of the following are true:

- the user can open one Streamlit app and reach both `Control` and `Analysis`
- the user no longer needs the notebook for routine experiment setup
- the user can generate a correct runner command for an experiment from the UI
- the user can inspect an experiment and click through its combos and answers
- prompt similarity works on prompt text items, not only answer records
- cleanup actions are guarded by dependency previews and confirmations
- smoke tests cover the new data-loading and control actions

## Working Rules

1. Keep the current runner script as the source of truth for execution.
2. Keep the current review data contracts stable unless a change is clearly justified.
3. Prefer safe archive and detach operations before hard delete.
4. Add domain operations to `PromptGarden` before wiring UI buttons to them.
5. Ship the panel in small vertical slices that remain usable after each step.
6. Mark progress directly in this document as steps are completed.

## Main Execution Plan

### [x] Step 1. Restructure the Streamlit app into maintainable modules

Purpose: prevent `apps/prompt_garden_review.py` from turning into one giant file when control features are added.

Planned work:

- keep `apps/prompt_garden_review.py` as the entry point
- extract reusable loaders, selectors, and render helpers into small modules under `src/chemistry_bot/promptops/`
- separate control-surface rendering from analysis-surface rendering
- preserve existing review features while refactoring

Primary files:

- `apps/prompt_garden_review.py`
- new `src/chemistry_bot/promptops/*` app-support modules
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the app still launches
- existing review smoke coverage still passes
- new control-panel code can be added without growing one monolithic file

### [x] Step 2. Add missing domain actions to `PromptGarden`

Purpose: the UI needs safe backend operations, not direct file edits from Streamlit code.

Planned work:

- add experiment metadata update helpers
- add combo detach helpers
- add dependency inspection helpers for prompts, combos, and experiments
- add archive helpers for prompts, combos, and experiments
- add guarded delete helpers with preflight checks

Primary files:

- `src/chemistry_bot/promptops/garden.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the app can call stable backend methods for management actions
- every destructive action has a previewable dependency check
- fixture-level tests cover the new operations

### [x] Step 3. Build a relationship and dependency index layer

Purpose: make it easy to answer "where is this prompt used?" and "what is inside this experiment?"

Planned work:

- add helper functions that map prompt -> combos -> experiments
- add helper functions that map experiment -> combos -> prompt roles
- add lightweight summary rows for prompt usage, combo usage, and experiment composition
- keep the data fast enough for Streamlit caching

Primary files:

- new `src/chemistry_bot/promptops/*store*.py` helper module
- `src/chemistry_bot/promptops/garden.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- a prompt can be inspected together with its dependents
- a combo can be inspected together with its prompt members and experiment membership
- an experiment can be rendered as a readable composition summary

### [x] Step 4. Generalize the embedding similarity backend

Purpose: reuse one similarity engine for both answers and prompts.

Planned work:

- refactor the current record-oriented embedding utilities into a generic text-item workflow
- keep answer similarity working without regression
- add prompt-item embedding support based on prompt text and prompt metadata
- add cache naming that distinguishes answer bundles from prompt bundles

Primary files:

- `src/chemistry_bot/promptops/review_embeddings.py`
- possibly new `src/chemistry_bot/promptops/*similarity*.py` helper module
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- prompt similarity can be computed from prompt items
- answer similarity still works through the existing app path
- cached prompt bundles and cached answer bundles do not conflict

### [x] Step 5. Create the new top-level app shell

Purpose: establish the operator experience before adding detailed views.

Planned work:

- add top-level `Control` and `Analysis` tabs
- keep sidebar workspace selection and cache reload behavior
- add global status header with selected workspace, counts, and shortcuts
- preserve current review scope loading while preparing shared navigation

Primary files:

- `apps/prompt_garden_review.py`
- extracted app-support modules from Step 1

Definition of done:

- the app clearly exposes `Control` and `Analysis`
- the current review flow still works
- the user has one visible entry point for future management tasks

### [x] Step 6. Build the Prompt Explorer

Purpose: let the user remember what each prompt is and where it belongs.

Planned work:

- add prompt search by `id`, title, type, branch, tree, and tags
- show prompt text, stats, metadata, and lineage
- show prompt usage summaries and dependent combos
- support few-shot prompt inspection in a readable way

Primary files:

- app-support modules created in Step 1
- `src/chemistry_bot/promptops/garden.py`
- new prompt explorer helper modules if needed

Definition of done:

- the user can find a prompt quickly
- the user can read the full prompt text
- the user can see what combos and experiments depend on it

### [x] Step 7. Build the Combo Explorer

Purpose: make combo inspection first-class instead of buried in notebook tables.

Planned work:

- list combos with filters by status, test status, tags, and kind
- show combo cards with prompt-role membership
- allow click-through from combo to prompt and from prompt back to combo
- display combo stats, notes, score, and experiment membership

Primary files:

- app-support modules created in Step 1
- `src/chemistry_bot/promptops/garden.py`

Definition of done:

- the user can understand a combo without opening the notebook
- every combo clearly shows which prompt ids it contains
- experiment membership is visible from the combo view

### [x] Step 8. Build the Experiment Builder and editor

Purpose: replace notebook cells that create experiments and attach combos.

Planned work:

- create experiments from a form
- load experiments by `id` or by name
- edit experiment metadata after creation
- attach and detach combos from an experiment
- show composition blocks for each attached combo

Primary files:

- app-support modules created in Step 1
- `src/chemistry_bot/promptops/garden.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the user can create a new experiment from the UI
- the user can revise goal, hypothesis, notes, tags, and status later
- the experiment view clearly lists attached combos and their prompt roles

### [x] Step 9. Add the Command Builder and dry-run preview

Purpose: make experiment execution practical without embedding long-running runs into Streamlit yet.

Planned work:

- expose model, bot variant, few-shot, case set, run mode, combo filters, and case filters
- generate a command string for `scripts/run_prompt_experiment.py`
- reuse the runner planning layer to show target counts and scope preview
- offer separate commands for full experiment and narrowed subsets

Primary files:

- `src/chemistry_bot/promptops/runner.py`
- app-support modules created in Step 1
- `tests/test_prompt_garden_smoke.py`
- `docs/prompt-garden-review.md`

Definition of done:

- the user can click `Generate command`
- the user gets a correct command to paste into the terminal
- the user also sees what that command is expected to run

### [x] Step 10. Build the Experiment Analysis workspace

Purpose: move post-run human review out of the notebook and into the panel.

Planned work:

- select experiments by name or `id`
- show experiment summary, status, score ranges, and attached combos
- browse normalized artifacts answer by answer
- open one answer and show question, short answer, explanation, score, combo, model, and few-shot
- keep answer browsing simple and manual for this phase

Primary files:

- `apps/prompt_garden_review.py`
- `src/chemistry_bot/promptops/review_store.py`
- app-support modules created in Step 1

Definition of done:

- the user can inspect completed experiment outputs without the notebook
- answers can be opened and browsed manually
- the experiment panel acts as the main review surface for this simplified phase

### [x] Step 11. Move post-analysis notes and experiment finalization into the panel

Purpose: make experiment interpretation a deliberate follow-up step, not a cell right after creation.

Planned work:

- expose editable experiment notes in the analysis view
- support final summary text and final subject score
- expose explicit finalize action
- keep existing review notes behavior where it is still useful

Primary files:

- `src/chemistry_bot/promptops/garden.py`
- app-support modules created in Step 1
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- experiment notes can be updated after results exist
- finalization is available from the analysis workspace
- the notebook no longer owns the main final-summary workflow

### [x] Step 12. Add safe cleanup flows for prompts, combos, and experiments

Purpose: allow users to reduce clutter without breaking the workspace accidentally.

Planned work:

- add archive actions for prompts, combos, and experiments
- add delete-preview panels that show dependencies and blockers
- allow hard delete only when constraints are satisfied or the cascade is explicit
- make cleanup actions visible but clearly marked as risky

Primary files:

- `src/chemistry_bot/promptops/garden.py`
- app-support modules created in Step 1
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- users can safely clean up old artifacts
- destructive actions are not silent
- dependency previews are part of the workflow

### [x] Step 13. Add Prompt Similarity to the analysis surface

Purpose: help the user identify duplicates, near-duplicates, and distinct prompt branches.

Planned work:

- allow prompt-item selection by type, tree, branch, and tags
- compute pairwise similarity, clusters, and outliers for prompt text
- show nearest prompt neighbors for a selected prompt
- support cache save and cache reload for prompt similarity bundles

Primary files:

- `src/chemistry_bot/promptops/review_embeddings.py`
- app-support modules created in Step 1
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the user can browse prompt similarity from the panel
- prompt clusters and near-duplicates are visible
- prompt similarity reuses the shared embedding-style backend

### [x] Step 14. Final integration, smoke coverage, and docs cleanup

Purpose: promote the panel from experiment to supported workflow.

Planned work:

- extend smoke tests for control and analysis loaders
- extend smoke tests for command generation and safe cleanup previews
- update root and Prompt Garden docs to point to the Streamlit control panel
- reduce the notebook to fallback or power-user status in documentation

Primary files:

- `tests/test_prompt_garden_smoke.py`
- `scripts/run_prompt_garden_smoke_tests.py`
- `README.md`
- `prompt_garden/README.md`
- `docs/prompt-garden-review.md`
- `scripts/README.md`

Definition of done:

- documentation reflects the new recommended operator flow
- smoke coverage protects the main control-panel behaviors
- the migration has a clear documented landing state

## Delivery Strategy

Recommended implementation order:

1. Step 1
2. Step 2
3. Step 3
4. Step 5
5. Step 6
6. Step 7
7. Step 8
8. Step 9
9. Step 10
10. Step 11
11. Step 12
12. Step 4
13. Step 13
14. Step 14

Rationale:

- Steps 1 to 3 create the backbone the UI needs.
- Steps 5 to 12 deliver the main control-panel experience.
- Steps 4 and 13 add prompt similarity once the main control flow is already useful.
- Step 14 locks the workflow down with tests and docs.

## Progress Log

- [x] Plan document created in repository root
- [x] Step 1 completed
- [x] Step 2 completed
- [x] Step 3 completed
- [x] Step 4 completed
- [x] Step 5 completed
- [x] Step 6 completed
- [x] Step 7 completed
- [x] Step 8 completed
- [x] Step 9 completed
- [x] Step 10 completed
- [x] Step 11 completed
- [x] Step 12 completed
- [x] Step 13 completed
- [x] Step 14 completed

## Current Next Action

Migration phase complete for this Streamlit control-panel pass. Use the panel as the documented main operator workflow, with the notebook reserved for authoring and deeper power-user edits.
