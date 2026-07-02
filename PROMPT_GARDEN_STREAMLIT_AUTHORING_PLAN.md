# Prompt Garden Streamlit Authoring Plan

Last updated: 2026-06-26
Owner: Codex
Status: In Progress

## Objective

Add the first minimal authoring flows directly into the Streamlit control panel so the user can do routine prompt and combo creation without going back to the notebook every time.

This phase includes exactly three operations:

1. `Create Root Prompt`
2. `Branch Prompt`
3. `Create Combo`

## Explicit Non-Goals

These operations are intentionally out of scope for this phase:

- direct editing of an existing prompt
- changing the parent of an existing prompt
- editing the membership of an existing combo

The supported model is:

- branch instead of editing prompts in place
- create a new combo instead of mutating an old combo
- archive unused prompts or combos instead of reshaping history

## UI Direction

The authoring area should live at the bottom of the `Control` surface, in the place where the current `What Comes Next` block sits.

That means:

- action buttons stay close to where the user is working
- the actual editor opens in one shared bottom authoring block
- the authoring block appears above `What Comes Next`
- the authoring block is mode-based, not tab-based

The shared bottom authoring block should support three modes:

1. `create_root_prompt`
2. `branch_prompt`
3. `create_combo`

Each mode must show exactly two action buttons:

- a primary save button
  For example: `Register Prompt` or `Register Combo`
- a secondary `Cancel` button

`Cancel` should always close the current authoring mode and clear the temporary form state.

## UX Rules

### Prompt Explorer

- `Branch Prompt` button lives under the selected prompt detail
- `Create Root Prompt` button lives near the bottom of the Prompt Explorer section, just before the shared bottom authoring block

### Combo Explorer

- `Create Combo` button lives near the bottom of the Combo Explorer section, just before the shared bottom authoring block

### Shared Authoring Block

- only one authoring block is open at a time
- opening a new mode replaces the previous draft after confirmation-safe state reset
- after successful save, the block closes and the app refreshes cached data
- after successful save, the newly created prompt or combo should become easy to inspect immediately

## Main Execution Plan

### [x] Step 1. Add shared authoring mode state to the Streamlit app

Purpose: create one stable place where Prompt Explorer and Combo Explorer can open the same bottom editor.

Planned work:

- add session-state keys for:
  - current authoring mode
  - current source tab
  - selected parent prompt id for branching
  - draft values
- add helper functions like:
  - open authoring mode
  - clear authoring mode
  - load defaults for each mode
- render the new shared authoring block above `What Comes Next`

Definition of done:

- the control panel can enter and leave an authoring mode without reloading the whole page unexpectedly
- only one draft mode is active at a time
- `Cancel` returns the page to normal browsing state

### [x] Step 2. Build the `Create Root Prompt` flow

Purpose: allow fast creation of a new root prompt without using the notebook.

Planned work:

- add a `Create Root Prompt` button in Prompt Explorer
- when clicked, open the shared authoring block in `create_root_prompt` mode
- add fields for:
  - `prompt type`
  - `tree_id`
  - `title`
  - `branch`
  - `tags`
  - `text`
- set reasonable defaults:
  - `branch = main`
  - empty tags allowed
- validate required fields before save
- save through `PromptGarden.create_root(...)`

Definition of done:

- the user can create a root prompt from Streamlit
- the new prompt appears in Prompt Explorer after refresh
- the new prompt is created with the correct type, tree, path, and text

### [x] Step 3. Build the `Branch Prompt` flow

Purpose: make prompt iteration the default safe workflow inside Streamlit.

Planned work:

- add a `Branch Prompt` button under selected prompt detail
- when clicked, open the shared authoring block in `branch_prompt` mode
- inherit automatically from the selected prompt:
  - `parent_id`
  - `prompt type`
  - `tree_id`
  - current text
- expose editable fields for:
  - `title`
  - `branch`
  - `tags`
  - `text`
- show parent prompt summary in read-only form
- save through `PromptGarden.create_child(...)`

Definition of done:

- the user can branch any existing prompt without manual copying
- the child prompt inherits the correct lineage automatically
- the new branch appears in Prompt Explorer and in lineage data

### [x] Step 4. Add prompt authoring guardrails

Purpose: make the first authoring flows safe enough for daily use.

Planned work:

- block save when required text fields are empty
- warn when branching from an archived prompt
- warn when branching from a prompt with file-read issues
- normalize tags input consistently
- keep `parent_id`, `type`, and `tree_id` read-only in branch mode
- show a success message with the new prompt id after save

Definition of done:

- the root and branch flows are hard to misuse accidentally
- the user understands what object was created and from where

### [x] Step 5. Build the `Create Combo` flow

Purpose: let the user register prompt combinations directly from Streamlit.

Planned work:

- add a `Create Combo` button in Combo Explorer
- when clicked, open the shared authoring block in `create_combo` mode
- add selector fields for:
  - `system prompt`
  - `user prompt`
  - optional `fewshot prompt`
- add metadata fields for:
  - `title`
  - `notes`
  - `tags`
- use readable option labels:
  - prompt id
  - prompt title
  - branch
  - short prompt preview
- save through `PromptGarden.create_combo(...)`

Definition of done:

- the user can create a combo without the notebook
- the saved combo appears in Combo Explorer immediately after refresh
- the combo clearly shows the selected prompt roles

### [x] Step 6. Add combo creation guardrails

Purpose: prevent noisy or duplicate combo registration.

Planned work:

- require `system` and `user` selections
- keep `fewshot` optional
- hide archived prompts from combo selectors by default
- add duplicate detection so the same prompt-role combination is not silently registered multiple times
- show a preview of the chosen prompt set before save
- show the resulting combo id after successful registration

Definition of done:

- the combo form is fast to use
- obvious duplicate combos are blocked or clearly warned about
- the user can see what will be registered before pressing save

### [x] Step 7. Integrate authoring refresh behavior

Purpose: make the new flows feel native inside the current Streamlit app.

Planned work:

- clear `st.cache_data` after successful create actions
- keep the user in `Control`
- reselect the newly created prompt after root or branch creation
- reselect the newly created combo after combo creation
- close the authoring block after successful save

Definition of done:

- creation feels continuous, not disconnected
- the user does not need manual reload steps after every save

### [x] Step 8. Add smoke coverage for the new authoring flows

Purpose: protect the first Streamlit-native authoring workflows against regression.

Planned work:

- extend smoke tests for:
  - root prompt creation reflected in prompt index bundles
  - child prompt branching reflected in lineage bundles
  - combo creation reflected in combo index bundles
  - duplicate combo prevention behavior
- keep fixture-level tests lightweight and file-based

Definition of done:

- the most important authoring paths have automated regression coverage
- the new flows do not rely only on manual clicking

### [x] Step 9. Update docs after implementation

Purpose: align the operator guide with the new capabilities.

Planned work:

- update `prompt_garden_step_by_step.md`
- update `prompt_garden/README.md`
- update `docs/prompt-garden-review.md`
- explain that:
  - basic prompt creation can now happen in Streamlit
  - basic branching can now happen in Streamlit
  - basic combo creation can now happen in Streamlit
  - the notebook remains for deeper power-user work

Definition of done:

- the documented workflow matches the UI
- users understand when Streamlit is enough and when the notebook is still useful

## Implementation Order

Recommended build order:

1. shared authoring mode state
2. `Create Root Prompt`
3. `Branch Prompt`
4. prompt guardrails
5. `Create Combo`
6. combo guardrails
7. refresh behavior
8. smoke tests
9. docs cleanup

## Progress

- [x] Step 1 completed
- [x] Step 2 completed
- [x] Step 3 completed
- [x] Step 4 completed
- [x] Step 5 completed
- [x] Step 6 completed
- [x] Step 7 completed
- [x] Step 8 completed
- [x] Step 9 completed

## Acceptance Criteria

This phase is complete when all of the following are true:

- a user can create a root prompt in Streamlit
- a user can branch an existing prompt in Streamlit
- a user can create a combo in Streamlit
- the shared bottom authoring block replaces the old empty space above `What Comes Next`
- every authoring mode has a clear primary save button and a clear `Cancel` button
- the user no longer needs the notebook for routine root prompt creation, prompt branching, or combo creation
- smoke tests protect the new creation paths

## Notes For Implementation

- do not add in-place editing in this phase
- do not add reparenting in this phase
- do not add combo membership editing in this phase
- prefer “create new and archive old” over mutating history
- keep the UI small and obvious rather than trying to expose every low-level field at once
