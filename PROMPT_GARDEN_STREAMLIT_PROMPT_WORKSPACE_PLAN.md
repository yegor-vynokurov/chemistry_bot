# Prompt Garden Streamlit Prompt Workspace Plan

Last updated: 2026-06-29
Owner: Codex
Status: In Progress

## Objective

Unify prompt inspection, dependency understanding, and prompt actions into one Streamlit workspace so the user can:

1. choose one prompt
2. read its text
3. understand where it is used and how well it performed
4. decide whether to branch, archive, or delete it

without jumping between separate prompt-explorer and cleanup flows.

## Problem This Plan Solves

The current prompt workflow is split in a way that creates extra operator effort:

- the user may inspect a prompt in one place and then search for the same prompt again in cleanup
- the current dependency view is backend-oriented JSON, not a human-readable decision aid
- the current dependency view explains structural usage, but not whether the prompt was historically valuable in experiments
- routine prompt actions are not grouped around the main inspection surface

That means the interface does not yet support the real operator question:

- "I found a prompt with issues. Before I archive or delete it, was it actually useful anywhere?"

## Initial Scope

Included in this prompt-workspace pass:

- one prompt selector and filter flow
- one text-first prompt detail surface
- one human-readable `Usage & Results` block
- combo-level and model-level performance summaries for the selected prompt
- inline `Branch Prompt` and `Archive Prompt` actions on the same screen
- `Delete Prompt` moved into the same prompt workspace under a dedicated danger zone
- removal of duplicated prompt lookup UI from prompt cleanup

Explicitly out of scope for this pass:

- in-place editing of existing prompts
- reparenting prompt lineage
- combo membership editing
- prompt-to-prompt diff UI
- broad redesign of combo and experiment sections

## UX Direction

The prompt workflow should become prompt-centric rather than action-centric.

The user experience should be:

1. filter prompts once
2. select one prompt once
3. read the prompt text first
4. read `Usage & Results`
5. choose the next action

The preferred layout is:

1. `Filters`
2. `Prompt`
3. `Usage & Results`
4. `Actions`
5. `Danger Zone`

The new `Usage & Results` block should answer these operator questions directly:

- where is this prompt used
- in which role is it used
- which combos using this prompt performed best
- which models performed best with combos using this prompt
- is the prompt safe to delete
- if it is not safe to delete, why not

## Success Criteria

This migration is complete for this phase when all of the following are true:

- the user no longer has to reselect the same prompt in a separate cleanup panel
- one prompt screen shows full prompt text and readable usage information together
- the raw `Dependency Preview` JSON is replaced by human-readable summaries
- the user can see top combos and top models for the selected prompt when run data exists
- `Branch Prompt` and `Archive Prompt` are available beside the main prompt detail flow
- `Delete Prompt` is available from the same screen but clearly separated as risky
- prompts with no combos or no run history still render a useful empty-state explanation
- smoke tests protect the new bundle loading and prompt action paths

## Working Rules

1. Keep prompt text inspection primary and action controls secondary.
2. Reuse normalized review artifacts and review-row helpers instead of inventing a second scoring source.
3. Translate technical blocker codes into operator language before rendering them in Streamlit.
4. Keep `Delete Prompt` guarded and visually separate from routine actions.
5. Remove duplicated prompt browsing UI instead of adding a third variant.
6. Mark progress directly in this document as work lands.

## Main Execution Plan

### [x] Step 1. Define the unified prompt workspace shell and navigation

Purpose: replace the split `Prompt Explorer` plus `Prompt Cleanup` mental model with one prompt-centric operator surface.

Planned work:

- decide whether to rename `Prompt Explorer` to `Prompt Workspace` or keep the current section label while changing the content
- keep one prompt selection state shared by inspection and actions
- remove the need to search for the same prompt in two separate sections
- leave combo and experiment cleanup in their own cleanup flows for now

Primary files:

- `src/chemistry_bot/promptops/review_app_control.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- one prompt selection flow is the source of truth for prompt inspection and prompt actions
- the prompt view no longer depends on a second prompt-cleanup selector

### [x] Step 2. Add a dedicated prompt-workspace backend bundle

Purpose: give the UI one stable source for prompt text, usage, performance, and deletion safety.

Planned work:

- add a bundle loader such as `load_prompt_workspace_bundle(...)`
- aggregate prompt metadata, prompt text, lineage data, combo usage, experiment usage, and review-derived performance rows
- compute summary counts for:
  - child prompts
  - dependent combos
  - dependent experiments
  - total recorded runs
- compute derived highlights for:
  - top combos by average or best score
  - top models by average score and pass rate
  - latest observed run timestamp
- keep prompts with no run history valid and explicit in the bundle

Primary files:

- `src/chemistry_bot/promptops/garden_index.py`
- `src/chemistry_bot/promptops/review_store.py`
- `src/chemistry_bot/promptops/review_app_data.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the Streamlit layer can load one prompt bundle and does not need to stitch together multiple ad hoc queries
- the bundle exposes both structural usage and performance summaries

### [x] Step 3. Translate dependency data into operator-readable safety summaries

Purpose: replace backend-style blocker JSON with clear human guidance.

Planned work:

- map technical blocker codes such as `not_archived`, `has_child_prompts`, and `used_by_combos` to readable labels
- expose a short delete-safety summary:
  - `safe_to_delete`
  - plain-language reason
  - blocking child prompts, combos, and experiments
- separate "used here" information from "delete blocked because..." information
- keep raw JSON only behind an optional debug expander if it remains useful internally

Primary files:

- `src/chemistry_bot/promptops/garden.py`
- `src/chemistry_bot/promptops/review_app_control.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- a user can understand prompt delete safety without reading raw JSON
- the UI explains both current usage and deletion blockers in plain language

### [x] Step 4. Build the text-first prompt detail layout

Purpose: make prompt reading the first task, not a secondary tab.

Planned work:

- render prompt title and compact metadata line at the top
- render the full prompt text inline without requiring a tab switch
- show file-read problems or canonical-path fallbacks as readable warnings
- keep low-level metadata secondary and collapsible

Primary files:

- `src/chemistry_bot/promptops/review_app_control.py`

Definition of done:

- the user can open one prompt and immediately read its text
- the prompt text no longer competes visually with raw metadata blocks

### [x] Step 5. Build the new `Usage & Results` block

Purpose: help the user decide whether a flawed prompt is disposable, historically valuable, or worth branching.

Planned work:

- add a summary strip that states:
  - how many child prompts depend on this prompt
  - how many combos use it
  - how many experiments include those combos
  - whether there is recorded run history
- add a `Combo Usage` table with rows such as:
  - combo id
  - prompt role
  - experiment count
  - run count
  - average score
  - best score
  - pass rate
  - best observed model
  - latest run timestamp
- add a `Model Summary` table with rows such as:
  - model
  - run count
  - average score
  - best score
  - pass rate
  - strongest combo using this prompt
- add highlight callouts like:
  - `Top combos`
  - `Top models`
  - `No recorded runs yet`

Primary files:

- `src/chemistry_bot/promptops/review_store.py`
- `src/chemistry_bot/promptops/review_app_data.py`
- `src/chemistry_bot/promptops/review_app_control.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the user can tell whether the prompt performed well anywhere
- the UI shows prompt usage results in readable summaries and tables rather than raw dependency JSON

### [x] Step 6. Move routine prompt actions next to prompt inspection

Purpose: let the user act immediately after understanding the prompt.

Planned work:

- keep the current branch flow available from the prompt workspace
- add `Archive Prompt` beside `Branch Prompt`
- keep successful actions refreshing the same prompt-centric workspace
- keep archive as a routine action and not as a hidden cleanup-only action

Primary files:

- `src/chemistry_bot/promptops/review_app_control.py`
- `tests/test_prompt_garden_smoke.py`

Definition of done:

- the user can read a prompt and then branch or archive it without leaving the prompt view
- routine prompt actions are grouped around the selected prompt

### [x] Step 7. Move prompt deletion into a dedicated danger zone on the same screen

Purpose: keep delete possible without making it the default action or forcing a second prompt search flow.

Planned work:

- place `Delete Prompt` under a `Danger Zone` block on the prompt workspace
- show delete blockers in plain language above the delete control
- keep the archive-first recommendation visible
- disable permanent delete when prompt safety conditions are not satisfied

Primary files:

- `src/chemistry_bot/promptops/review_app_control.py`
- `src/chemistry_bot/promptops/garden.py`

Definition of done:

- prompt deletion can be reviewed from the same screen as prompt text and usage
- destructive controls remain visibly separate and guarded

### [x] Step 8. Retire duplicated prompt cleanup browsing UI

Purpose: reduce interface duplication and keep one clear prompt workflow.

Planned work:

- remove the separate prompt-selector-and-preview flow from `Prompt Cleanup`
- keep combo and experiment cleanup sections intact
- update any empty-state or navigation copy so the user understands that prompt archive and delete now live in the main prompt workspace

Primary files:

- `src/chemistry_bot/promptops/review_app_control.py`
- `docs/prompt-garden-review.md`
- `prompt_garden/README.md`
- `prompt_garden_step_by_step.md`

Definition of done:

- there is only one prompt browsing surface for prompt text, usage, archive, and delete
- the cleanup area no longer duplicates prompt selection UI

### [x] Step 9. Add smoke coverage and docs updates

Purpose: lock the new prompt-centric workflow in as the supported path.

Planned work:

- add smoke tests for:
  - prompt workspace bundle loading
  - prompts with no runs
  - prompts used by scored combos
  - readable delete blockers
  - archive action availability from the prompt workspace
- update prompt workflow documentation to describe:
  - one prompt workspace
  - text-first inspection
  - usage and result interpretation
  - archive versus delete guidance

Primary files:

- `tests/test_prompt_garden_smoke.py`
- `scripts/run_prompt_garden_smoke_tests.py`
- `docs/prompt-garden-review.md`
- `prompt_garden/README.md`
- `README.md`
- `prompt_garden_step_by_step.md`

Definition of done:

- the prompt-centric flow is documented
- regression coverage protects the most important inspection and action paths

## Implementation Order

Recommended build order:

1. Step 1
2. Step 2
3. Step 3
4. Step 4
5. Step 5
6. Step 6
7. Step 7
8. Step 8
9. Step 9

Rationale:

- Steps 1 to 3 establish the architecture and backend semantics.
- Steps 4 and 5 make the prompt screen genuinely useful before actions are moved.
- Steps 6 and 7 place routine and destructive actions around the new decision surface.
- Steps 8 and 9 remove duplication and finalize the migration with tests and docs.

## Acceptance Criteria

This phase is complete when all of the following are true:

- the user filters and selects a prompt in one place
- the prompt text is visible on the main prompt screen without a tab switch
- `Usage & Results` explains prompt usage and performance in human-readable form
- the user can tell which combos and models worked best with that prompt
- `Branch Prompt` and `Archive Prompt` live on the same prompt screen
- `Delete Prompt` is available from the same prompt screen in a danger zone
- raw dependency JSON is no longer the default way to understand prompt safety
- prompt cleanup no longer requires duplicated prompt lookup UI
- smoke coverage and docs reflect the new operator workflow

## Progress

- [x] Plan document created
- [x] Step 1 completed
- [x] Step 2 completed
- [x] Step 3 completed
- [x] Step 4 completed
- [x] Step 5 completed
- [x] Step 6 completed
- [x] Step 7 completed
- [x] Step 8 completed
- [x] Step 9 completed

## Current Next Action

All planned prompt-workspace migration steps are complete. The prompt-centric inspection, archive, and delete flow is now the supported operator path.

Steps 1 to 8 are in place. Next move: add the final smoke coverage and workflow doc updates that lock the prompt-centric flow in as the supported operator path.
