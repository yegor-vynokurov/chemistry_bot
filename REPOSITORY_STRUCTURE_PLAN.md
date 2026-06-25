# Repository Structure And Documentation Plan

## Purpose

This plan describes how to bring the repository to a resume-ready state without changing the application logic. The focus is on repository structure, file organization, naming, documentation, docstrings, comments, and artifact hygiene.

This document is also the working step-by-step roadmap for moving the repository from its current experimental state to a good public architecture with minimal future large-scale rewrites.

## Scope Of This Planning Pass

The goal of this planning pass is not to make the project final. The goal is to move it into a good intermediate state:

- the architecture is understandable
- the active experimental direction is clear
- the repository is reproducible
- the main functionality can be smoke-tested
- future cleanup can happen incrementally instead of through one large rewrite

## How This Document Should Be Used

This file should be treated as an execution roadmap, not only as an architecture note.

Working rule:

- use the phased roadmap to decide what to change next
- use the separate quality-criteria file to judge whether a change is good enough
- prefer small coherent cleanup steps over sweeping refactors
- keep the repository usable after each phase
- mark completed phases and tasks inside this roadmap as work progresses

## Current Verified Baseline

The roadmap should start from a verified baseline rather than assumptions.

Verified locally on June 25, 2026:

- Ollama is available locally
- `phi4-mini` is available as a local chat model
- `embeddinggemma` is available as a local embedding model
- the Chroma manifest exists at `data/indexes/introductory_chemistry_chroma/introchem_theory_v1.manifest.json`
- `src/chemistry_bot/retrieval/introchem_rag.py` successfully connected to the current vector index and `retrieve("What is a chemical equation?")` returned hits
- `src/chemistry_bot/cli/rag_bot.py` successfully answered a smoke-test question through `CliBot.invoke_once(...)` using `phi4-mini`
- `scripts/build_introchem_rag_bundle.py --dry-run` successfully expanded into the expected parse -> chunk -> index command sequence
- that smoke test returned a structured theory answer and included RAG source references

This means the plan can assume a real working experimental baseline for:

- local Ollama connectivity
- vector retrieval connectivity
- end-to-end question answering through the RAG-enabled bot

## Working Principles For Execution

The cleanup should follow these principles:

1. Keep the repository honest about experiments.
   The new RAG-enabled CLI path and the older CLI path should stay clearly separated until convergence is intentional.

2. Prefer structural clarity before deep internal refactoring.
   We should first improve names, placement, documentation, and reproducibility contracts.

3. Preserve working baselines.
   Before changing structure aggressively, maintain at least one verified runnable bot path.

4. Make each phase independently useful.
   After each phase, the repository should be easier to understand than before.

5. Avoid unnecessary churn in generated data.
   Reproducibility matters more than committing every derived artifact.

## Target State For This Iteration

For the current iteration, "good enough" means:

- the repository clearly presents the RAG-enabled experimental bot as the main direction
- the old and new bot variants are not confused with each other
- Prompt Garden is visible as a first-class subsystem
- the textbook-to-RAG pipeline is documented as one reproducible workflow
- at least a lightweight smoke-test path exists for core question-answer functionality
- the repository has strong documentation and structure even if some code internals remain experimental

It does not yet require the project to be fully finalized, fully packaged, or exhaustively tested.

## Phased Roadmap

### Phase 1. Freeze The Public Story

Status:
Completed on June 25, 2026.

Objective:
Make the repository tell one clear story before moving files around.

Main tasks:

- [x] finalize the public product description for an English-language chemistry bot for middle-school and high-school students
- [x] state that the current main experimental path is `src/chemistry_bot/cli/rag_bot.py`
- [x] state that `src/chemistry_bot/retrieval/introchem_rag.py` is the bot-facing retrieval integration layer
- [x] keep `src/chemistry_bot/cli/legacy_bot.py` clearly documented as the older parallel experimental variant
- [x] split architecture roadmap and stable quality criteria into separate documents
- [x] make `README.md` the canonical public entry point for the current repository story

Exit condition:
Someone opening the root of the repository can understand what the project is, who it is for, and which bot variant is currently the main one.

### Phase 2. Clean The Root Without Breaking The Workflow

Status:
Completed on June 25, 2026.

Objective:
Reduce top-level noise while keeping the project runnable.

Main tasks:

- [x] define which top-level files are active, temporary, or removable
- [x] keep `archive/` temporary and outside the final repository design
- [x] isolate runtime artifacts such as logs from source files
- [x] decide where the active notebook, Prompt Garden workspace, and documentation should live
- [x] keep old helper files out of the main project story

Exit condition:
The root directory contains mostly high-signal project files and clearly named top-level folders.

Current Phase 2 root classification:

- Active now:
  `README.md`, `REPOSITORY_STRUCTURE_PLAN.md`, `PROJECT_QUALITY_CRITERIA.md`,
  `src/chemistry_bot/cli/rag_bot.py`, `src/chemistry_bot/retrieval/introchem_rag.py`, `src/chemistry_bot/cli/legacy_bot.py`,
  `src/chemistry_bot/promptops/garden.py`, `src/chemistry_bot/promptops/eval.py`, `requirements.txt`, `requirements-rag.txt`,
  `src/`, `docs/`, `scripts/`, `tests/`, `config/`, `prompt_garden/README.md`,
  `prompt_garden/control/prompt_garden_experiments_control.ipynb`

- Temporary but still present:
  `archive/`

- Local-only or outside the public project story:
  `.venv/`, `.vscode/`, `.env`, `data/`, `logs/`, `local_sandbox/`

- Cleanup note:
  `.gitignore` has been tightened so runtime logs and local-only helper files are explicitly treated as non-public artifacts. Active bot logging has been redirected from the repository root to `logs/chat_session.log`. Supporting RAG documentation has been moved out of the root into `docs/`. The active Prompt Garden notebook now lives under `prompt_garden/control/`, and the workspace has its own tracked `prompt_garden/README.md`. Local helper scripts that do not belong to the project story now live under `local_sandbox/` instead of the repository root.

### Phase 3. Establish The Near-Final Folder Layout

Status:
Completed on June 25, 2026.

Progress note:
`docs/` and `scripts/` now exist as public-facing structure layers, the Prompt Garden notebook lives under `prompt_garden/control/`, and the repository now has a package-oriented scaffold under `src/chemistry_bot/`.

Objective:
Move the repository into a layout that is close to the long-term target, even if some code remains experimental.

Main tasks:

- [x] introduce or normalize `docs/`, `scripts/`, and package-oriented source layout
- [x] keep `prompt_garden/` as a top-level subsystem
- [x] place the Prompt Garden notebook under `prompt_garden/control/`
- [x] prepare a stable place for bot entry points and retrieval integration modules
- [x] keep data generation and data consumption paths easy to understand

Exit condition:
Most active files already live near their intended long-term location, so later cleanup needs fewer disruptive moves.

Phase 3 result:

- the repository now has a package-oriented scaffold under `src/chemistry_bot/`
- the active bot, retrieval, and prompt-operations implementation modules now live under `src/chemistry_bot/`
- public scripts, smoke tests, and the Prompt Garden notebook now import through the new package path
- `prompt_garden/` remains a top-level subsystem by explicit structure and documentation
- data and fixture placement are now documented in `docs/repository-layout.md`, `data/README.md`, and `config/README.md`
- shared Introductory Chemistry corpus paths now live in `src/chemistry_bot/retrieval/layout.py`

### Phase 4. Curate Prompt Garden For Public Use

Status:
Completed on June 25, 2026.

Objective:
Expose the valuable Prompt Garden assets without exposing unnecessary noise.

Main tasks:

- [x] adjust `.gitignore` to allow curated Prompt Garden assets into Git
- [x] track the current baseline prompts and combo lineage
- [x] retain graph relationships that explain prompt evolution
- [x] keep bulky or noisy runtime outputs selective
- [x] document Prompt Garden as both functional tooling and portfolio evidence

Exit condition:
The repository shows meaningful prompt engineering artifacts, not just code that references them.

Phase 4 result:

- the public tracked Prompt Garden baseline now lives under `prompt_garden/curated/`
- the curated baseline includes `combo_000014`, `sys_000006`, `usr_000006`, and `fsh_000002`
- the full runtime Prompt Garden workspace remains selective, while the reviewable exemplar set is now visible in Git

### Phase 5. Consolidate The Corpus Build Workflow

Status:
Completed on June 25, 2026.

Objective:
Make the RAG pipeline reproducible through one public-facing build command.

Main tasks:

- [x] verify how the textbook file should be distributed based on license status
- [x] document the exact corpus source and acquisition path
- [x] wrap parsing, chunking, and index building into one reproducible command path
- [x] keep `data/` ignorable if the build path is simple and reliable
- [x] retain lightweight fixtures needed for checks and demos

Exit condition:
A contributor can rebuild the corpus pipeline without manually stitching together multiple undocumented commands.

Phase 5 result:

- corpus provenance and license notes are now documented in `docs/data-sources.md`
- the public build wrapper now exists as `scripts/build_introchem_rag_bundle.py`
- the wrapper has been dry-run checked against the current parse -> chunk -> index flow
- `data/` remains ignored while the reproducible build path is made explicit
- lightweight fixtures such as `config/introchem_retrieval_queries.jsonl` remain part of the intended sanity-check workflow

### Phase 6. Add Functional Smoke Tests For Core QA

Status:
Completed on June 25, 2026.

Objective:
Prove that the repository is functional for its main use case, even before broader test coverage exists.

Main tasks:

- [x] create a retrieval-only smoke test for `src/chemistry_bot/retrieval/introchem_rag.py`
- [x] create an end-to-end smoke test for `src/chemistry_bot/cli/rag_bot.py`
- [x] use `phi4-mini` for the bot smoke path to keep test latency practical
- [x] keep Prompt Garden testing out of scope for this phase
- [x] define simple success signals such as structured answer returned, no crash, and optional source usage when RAG is enabled

Exit condition:
There is at least one fast sanity-check path that says, in effect, "the main question-answer workflow still works."

Phase 6 result:

- the repository now has tracked smoke tests under `tests/test_core_smoke.py`
- the public one-command runner now exists as `scripts/run_core_smoke_tests.py`
- the retrieval smoke path uses tracked fixtures from `config/introchem_retrieval_queries.jsonl`
- the bot smoke path uses `phi4-mini` and the current RAG-enabled CLI bot contract

### Phase 7. Portfolio Polish Without Premature Perfectionism

Status:
Completed on June 25, 2026.

Objective:
Make the repository feel intentional and credible without pretending the project is finished.

Main tasks:

- [x] polish README quality
- [x] improve folder-level documentation
- [x] align docstrings and comments with the active architecture
- [x] remove temporary legacy material from the public repo
- [x] leave open room for future code-level improvement without another structural reset

Exit condition:
The repository is strong enough to show publicly and stable enough that future changes are mostly incremental.

Phase 7 result:

- the root README now presents a cleaner public story with quickstart, repository map, and documentation links
- folder-level documentation has been added or improved for `docs/`, `src/`, `tests/`, `data/`, `config/`, `scripts/`, and `prompt_garden/`
- active modules now have clearer module docstrings aligned with their architectural role
- the unused legacy file `src/build_introchem_rag_chunks.py` has been removed from the public tracked structure
- the old root-level implementation modules have been migrated into `src/chemistry_bot/`
- the repository now presents the current working baseline without depending on duplicate root readmes or outdated prototype entry points

## Current Project Map

At the moment the repository contains three closely related but differently organized parts:

1. The dialogue bot layer.
   The dialogue bot layer currently has two parallel variants.
   `src/chemistry_bot/cli/rag_bot.py` is now the more important experimental CLI entry point because it includes RAG-aware behavior.
   `src/chemistry_bot/cli/legacy_bot.py` remains the previous non-RAG or pre-RAG-oriented CLI path and should not be silently merged into the new version while the experiment is still evolving.

   `src/chemistry_bot/retrieval/introchem_rag.py` is the supporting RAG orchestration module used by `src/chemistry_bot/cli/rag_bot.py` and should be documented as part of the bot-facing retrieval integration layer.

2. The RAG data pipeline.
   The `src/` scripts build the knowledge base in several stages:
   raw XHTML or educational materials -> normalized chapter data -> RAG chunks -> Chroma index -> retrieval queries and review output.

3. The prompt experimentation layer.
   `src/chemistry_bot/promptops/garden.py`, `src/chemistry_bot/promptops/eval.py`, `prompt_garden/`, and `prompt_garden/control/prompt_garden_experiments_control.ipynb` implement local prompt versioning, combo generation, graph relationships between prompt nodes and combos, evaluation, and experiment tracking.

There is also a fourth category: legacy or unrelated material.

- `archive/` currently contains historical versions and experiments kept only as temporary local reference.
- `main.py` is an older OpenRouter-based prototype and does not represent the current local-LLM RAG bot architecture.
- `test_torch.py` and `trics.py` are local helper files and are not part of the main product story.
- `chat_session.log` is a runtime artifact, not repository source.

## Main Repository Problems

- The root of the repository mixes active source files, runtime artifacts, notebooks, historical files, and documentation with no clear boundaries.
- The repository has two readme-like files with conflicting roles.
- The canonical project story is unclear: the current bot, the RAG pipeline, and Prompt Garden are all present, but their relationship is not explained from the top level.
- Essential lightweight project assets are mixed with generated or local-only assets.
- Some directories that describe the real project shape are ignored entirely, so the public repository may not reflect how the project actually works locally.
- Several filenames look versioned or temporary (`*_v3`, `READMY.md`, experimental helpers in root), which weakens the portfolio impression.
- Documentation language and encoding are inconsistent.
- There is no single documented rule for what belongs in Git, what is generated, what is archival, and what is only local.

## Target Repository Narrative

The repository should present one clear story:

`Chemistry Bot` is an English-language local-LLM educational chemistry assistant for middle-school and high-school students, with a custom RAG preparation pipeline and a prompt experimentation subsystem used to improve answer quality and safety.

That story should be visible immediately from the root README and from the top-level directory layout.

## Target Audience

The repository should state the intended audience explicitly and consistently:

- middle-school students
- high-school students

The project should not present itself as a tutor for university, graduate, or specialist chemistry training. That audience boundary affects prompt design, examples, evaluation criteria, README wording, and sample outputs.

## Project Language And Corpus Scope

The public repository should describe itself as an English-language project:

- bot prompts are primarily in English
- the main corpus is in English
- the primary textbook source is in English
- documentation for GitHub and portfolio use should be in English

Multilingual test queries may still exist as experiments, but the main product story should remain clearly English-first.

## Target Top-Level Structure

Preferred target layout:

```text
chemistry_bot/
  README.md
  LICENSE
  .gitignore
  .env.example
  requirements/
    base.txt
    rag.txt
    dev.txt
  docs/
    architecture.md
    repository-layout.md
    rag-pipeline.md
    prompt-garden.md
    data-sources.md
  prompt_garden/
    README.md
    control/
      prompt_garden_experiments_control.ipynb
    prompts/
      system/
      user/
      fewshot/
      safety/
      assistant/
      tool/
    registry/
      nodes.jsonl
      edges.jsonl
      combos.jsonl
      experiment_nodes.jsonl
      experiments.jsonl
      runs.jsonl
    experiments/
      exp_*.json
  src/
    chemistry_bot/
      __init__.py
      cli/
      promptops/
      retrieval/
      schemas/
      utils/
  scripts/
    run_cli_bot.py
    parse_introchem.py
    build_rag_chunks.py
    build_vector_index.py
    build_introchem_rag_bundle.py
  config/
    retrieval_queries.jsonl
  data/
    README.md
    sample/
```

This structure separates:

- source code
- runnable scripts
- human-facing documentation
- prompt assets and experiment history
- lightweight tracked configuration
- supported notebook-based control surfaces
- local or generated data

Temporary local historical material may exist during cleanup, but it should not remain part of the final repository structure.

In this target structure, `prompt_garden/` stays visible at the repository top level on purpose. It is not just helper runtime data. It is part of the product architecture because it stores prompt lineage, combo history, graph edges, experiment metadata, and the evolving prompt corpus used by the bot.

The Prompt Garden control notebook should also live inside the Prompt Garden subsystem rather than being presented as a random exploratory notebook. It is an operator interface for prompt versioning and experiment management.

## File Relocation Plan

The following mapping would make the repository easier to understand:

- `src/chemistry_bot/cli/rag_bot.py` -> current primary experimental application entry point inside `src/chemistry_bot/cli/`
- `src/chemistry_bot/retrieval/introchem_rag.py` -> bot-facing RAG integration module inside `src/chemistry_bot/retrieval/` or `src/chemistry_bot/cli/bridges/`
- `src/chemistry_bot/cli/legacy_bot.py` -> retained as the previous CLI variant, clearly marked as pre-RAG or non-RAG experimental lineage rather than merged prematurely into the new entry point
- `src/chemistry_bot/promptops/garden.py` -> active prompt-management module inside `src/chemistry_bot/promptops/`
- `src/chemistry_bot/promptops/eval.py` -> evaluation module inside `src/chemistry_bot/promptops/`
- `prompt_garden/` -> remains a first-class top-level project directory as the persisted prompt graph, combo registry, and experiment history workspace
- `src/parse_introchem_xhtml_v3.py` -> retrieval pipeline module inside `src/chemistry_bot/retrieval/` or `scripts/`
- `src/build_introchem_rag_chunks_v3.py` -> retrieval pipeline module inside `src/chemistry_bot/retrieval/` or `scripts/`
- `src/introchem_vector_search.py` -> retrieval/index/search CLI inside `src/chemistry_bot/retrieval/` or `scripts/`
- `prompt_garden_experiments_control.ipynb` -> `prompt_garden/control/` as an official Prompt Garden control surface
- `main.py` -> temporary legacy reference only, then remove from the main project or move to a private/local-only sandbox outside the final repository
- `test_torch.py` and `trics.py` -> outside the main repo story, then remove from the final repository or keep only in a private/local-only sandbox
- `chat_session.log` -> ignored runtime artifact under `logs/` or root ignore rules

## Active Vs Legacy Boundary

The repository should make the active code path obvious.

Active area:

- current local-LLM chemistry bot
- current experimental RAG-enabled bot path centered on `src/chemistry_bot/cli/rag_bot.py`
- the RAG bridge/orchestration layer centered on `src/chemistry_bot/retrieval/introchem_rag.py`
- current retrieval pipeline
- current prompt evaluation tooling

Legacy area:

- temporary historical snapshots kept locally during cleanup
- superseded scripts
- one-off experiments
- diagnostic utilities not needed to understand or run the project

These legacy materials should not be treated as a future repository subsystem.

Repository rule:

- `archive/` is temporary and for local reference only
- `archive/` should not appear in the final resume-oriented repository structure
- after useful reference value is exhausted, `archive/` should be removed
- old files should either be deleted or kept outside the public project workspace

## Parallel Bot Variants During Experimentation

The plan should explicitly preserve the distinction between the old CLI bot and the new RAG-enabled CLI bot.

Current intent:

- `src/chemistry_bot/cli/rag_bot.py` is the main experimental direction because it includes RAG
- `src/chemistry_bot/retrieval/introchem_rag.py` manages or coordinates RAG calls used by that new entry point
- `src/chemistry_bot/cli/legacy_bot.py` remains a separate earlier variant for comparison and reference

Repository rule for now:

- do not collapse these files into one unclear hybrid module
- do not rename the older file as if it were already replaced unless the experiment is complete
- document the relationship between the two variants clearly in README and architecture notes
- when the time comes to converge, do it intentionally as a separate repository-cleanup step

This keeps the public repository honest about the current state of the project and avoids hiding an active architectural experiment.

## README Plan

Only one root README should act as the canonical repository entry point.

Recommended role split:

- `README.md` becomes the main repository page for GitHub.
- Current Prompt Garden content moves to `docs/prompt-garden.md`.
- The current `READMY.md` content is either merged into the new root README or split across `docs/rag-pipeline.md` and quickstart sections.
- The typo-named `READMY.md` should not remain as a second root readme in the final portfolio version.

## Root README Content Plan

The root README should include only the information a recruiter, reviewer, or collaborator needs first:

- one-paragraph project summary
- explicit statement that the bot is for middle-school and high-school learners
- explicit statement that the main repository language is English
- what problem the bot solves
- current capabilities
- note that the current main experimental CLI includes RAG
- short architecture overview
- repository map
- quickstart for local run
- quickstart for RAG data preparation
- explanation of what data is included and what must be provided locally
- note about Prompt Garden as the experiment layer
- example CLI session or screenshot

The root README should not try to be the full technical manual.

## Subdirectory Documentation Plan

Short README files should exist where the folder itself needs explanation:

- `docs/README.md` if the docs folder grows
- `data/README.md` describing raw, normalized, rag, indexes, sample, and what is tracked
- `prompt_garden/README.md` describing the Prompt Garden workspace, control notebook, and registry model
- `config/README.md` only if configuration becomes non-trivial

## Documentation Language And Encoding

For a resume-oriented public repository, the primary documentation language should be English.

Recommended policy:

- all root and `docs/` documentation in English
- optional local notes in another language only if clearly separated
- all Markdown and source files saved in UTF-8
- examples may include multilingual chemistry prompts, but explanatory text should stay consistent

This will also eliminate the current impression of mixed-language and mixed-encoding documentation.

## Naming Policy

The repository should adopt a consistent naming policy:

- use descriptive English filenames
- avoid typo-like names such as `READMY.md`
- avoid version suffixes in active filenames when the old version is no longer active
- keep versioned or superseded scripts out of the final public repository unless they are still actively needed
- use one naming style for scripts and modules

If multiple generations must coexist temporarily, their status should be explained in documentation, not only encoded in filenames.

## Prompt Garden As A First-Class Subsystem

`prompt_garden/` should remain explicit in the final repository structure.

For this project, Prompt Garden is not just tooling around the bot. It is one of the core differentiators of the repository because it preserves:

- prompt source texts
- prompt-node history
- graph edges between prompt nodes, combos, and experiments
- combo definitions
- experiment definitions
- experiment result history
- reproducible prompt-evaluation context

That means the public repository structure should make Prompt Garden visible immediately, not hide it inside generic config or treat it as disposable runtime output.

## Prompt Garden Internal Structure Plan

The `prompt_garden/` directory should be documented as a structured workspace with separate roles:

- `control/` for the operator-facing notebook and any future control helpers
- `prompts/` for the actual versioned prompt content
- `registry/` for graph metadata such as nodes, edges, combos, and compact experiment records
- `experiments/` for full experiment objects and detailed result history
- `README.md` explaining the folder model, ID conventions, and which files are source-of-truth

The notebook should be described as a supported functional interface for:

- prompt versioning
- combo management
- experiment creation
- experiment review
- prompt evolution workflows

It should not be documented merely as an exploratory notebook.

## Prompt Garden Tracking Policy

Because Prompt Garden is core to the project, not every file in it should be treated the same.

Should usually be tracked in Git:

- `prompts/**`
- `registry/nodes.jsonl`
- `registry/edges.jsonl`
- `registry/combos.jsonl`
- `registry/experiment_nodes.jsonl`
- compact experiment metadata that explains how prompt evolution happened
- a small curated set of experiment objects if they help demonstrate the project in a portfolio context

Should be reviewed case by case:

- `registry/experiments.jsonl`
- detailed experiment result files in `experiments/`
- notebook-generated evaluation outputs

May become local-only, rotated, or pruned if they grow too large:

- `registry/runs.jsonl`
- raw execution logs
- bulky repeated experiment outputs that do not improve repository comprehension

The key rule is that Prompt Garden history, relationships, and prompt lineage should remain visible, while noisy high-volume raw run data can be managed more selectively.

## Curated Prompt Assets To Unignore And Track

The `.gitignore` policy should be refined so that the repository tracks a curated set of high-value Prompt Garden assets instead of hiding the entire subsystem.

The current seed set to bring out of ignore rules should include:

- the current best combo definition: `combo_000014`
- the prompt nodes referenced by that combo: `sys_000006` and `usr_000006`
- the current few-shot prompt used by the bot: `fsh_000002`

The goal of tracking these files is not only reproducibility. They are also portfolio artifacts that show:

- how the project encodes prompt structure
- how system and user prompts evolve
- how few-shot examples are curated
- which prompt assets are the current recommended baseline for future iterations

The tracked set should therefore include both the prompt text files and the registry records that preserve their relationships.

## `.gitignore` Strategy For Prompt Garden

Instead of a blanket ignore for `prompt_garden/`, the long-term ignore strategy should support selective inclusion through explicit allowlists.

The plan should allow:

- tracked prompt exemplars and their registry metadata
- tracked curated combos and few-shot baselines
- tracked structural files that explain graph relationships
- ignored high-volume runtime logs and non-essential run history

This gives the repository a clean public face without erasing the most important evidence of prompt engineering work.

## Data And Artifact Policy

The repository should explicitly define which data is:

- source input
- generated intermediate output
- generated index output
- reproducibility fixture
- local-only artifact

Recommended rules:

- the whole `data/` directory may remain ignored if the corpus build is simple and reproducible from a documented command
- `data/raw/` should not be committed blindly; it depends on the license status of the source material
- `data/normalized/`, `data/rag/`, and `data/indexes/` should usually be treated as generated artifacts
- only small representative samples should be tracked in Git when needed for demonstration
- `config/introchem_retrieval_queries.jsonl` is lightweight and useful for reproducibility, so config-like evaluation files should be tracked
- logs, notebook checkpoints, local caches, and vector indexes should be ignored

For this repository specifically, a reproducible pipeline is more important than committing all derived artifacts. If one command can regenerate normalized data, chunks, and embeddings from the source textbook, then keeping `data/` ignored is a reasonable default.

## Data Provenance And Licensing Documentation

Because the repository uses textbook and educational source materials, the project documentation should clearly state:

- what datasets or documents are used
- whether they are redistributed in the repository or expected locally
- the provenance of each corpus
- any license or usage restrictions

For a portfolio repository this is not optional. It directly affects credibility and professionalism.

For the current corpus, the plan should explicitly mention:

- primary corpus: `Introductory Chemistry, 1st Canadian Edition`
- action item: verify the redistribution license for the textbook file itself
- preferred outcome: include the source textbook file in the repository if the license allows redistribution
- fallback outcome: provide a stable download link and a documented acquisition step if redistribution is not allowed

Because the parser is already tuned for this exact book and the source file is relatively small, keeping the source text in the repository is desirable if licensing permits it.

## Dependency Documentation Plan

Dependency documentation should be reorganized so the project reads as intentional rather than ad hoc:

- keep one base dependency list for the bot
- keep RAG-specific dependencies separately
- keep optional development or notebook dependencies separately
- document which command uses which dependency file

Whether the final form stays as `requirements/*.txt` or later moves to another package-management format, the public structure should already make the dependency split clear.

## Entry Point Clarity

The repository should expose a small number of obvious entry points:

- one main command for the current RAG-enabled experimental bot
- one clearly labeled command for the older non-RAG or earlier bot variant, if it remains in the repo during the transition
- one main build command that turns the textbook into normalized content, RAG chunks, and embeddings
- one command path for retrieval search and evaluation
- one official notebook interface for Prompt Garden operations

Right now that information exists, but it is scattered between root files and readmes.

## RAG Integration Boundary

The final repository structure should make the responsibility split between the new bot entry point and the RAG integration module explicit.

Desired conceptual split:

- `src/chemistry_bot/cli/rag_bot.py` owns the user-facing bot flow
- `src/chemistry_bot/retrieval/introchem_rag.py` owns retrieval invocation, retrieval assembly, and the contract between bot logic and the Introductory Chemistry index

That separation should remain visible in filenames, module placement, and documentation.

The plan should avoid mixing:

- prompt orchestration concerns
- chat-loop concerns
- retrieval-invocation concerns
- corpus-specific RAG helper logic

This does not require immediate refactoring of code internals. It only establishes the repository-structure rule that these concerns should be documented and laid out separately.

## One-Command Corpus Build Plan

The RAG preparation flow should be wrapped in one documented command so the repository feels reproducible and easy to trust.

Target outcome:

- one command receives the textbook file path or uses the tracked source file
- the command runs parsing
- the command builds RAG chunks
- the command builds or rebuilds embeddings and the vector index
- the command prints a short success summary with output locations

The implementation can still call multiple internal scripts, but the public interface should feel like one pipeline, not several manual steps.

This is especially important because the parser is intentionally specialized for `Introductory Chemistry, 1st Canadian Edition`.

## Testing

Testing is currently one of the weaker parts of the project and should be treated as a dedicated architectural workstream, not a side note.

The immediate goal is not broad coverage. The immediate goal is confidence in the main value path: question answering for the school-focused chemistry bot.

### Testing Scope For The Next Iteration

Include:

- retrieval connectivity for `src/chemistry_bot/retrieval/introchem_rag.py`
- end-to-end question answering for `src/chemistry_bot/cli/rag_bot.py`
- basic reproducibility checks for the current RAG-enabled bot configuration
- simple pass/fail smoke tests that are fast enough to run often

Do not include yet:

- Prompt Garden test automation
- broad prompt-quality benchmarking
- large regression suites
- full notebook test automation

Prompt Garden testing should be planned later as its own dedicated block.

### Verified Local Test Baseline

Verified on June 25, 2026:

- `src/chemistry_bot/retrieval/introchem_rag.py` successfully retrieved textbook hits from the existing Chroma index
- `src/chemistry_bot/cli/rag_bot.py` successfully produced a structured answer through `CliBot.invoke_once(...)`
- the end-to-end smoke test worked with `phi4-mini`
- the answer used retrieved source IDs, confirming that the RAG-enabled path is functionally connected

### Minimum Test Targets

The repository should eventually contain at least these practical tests:

1. Retrieval smoke test.
   Confirm that the index opens, retrieval runs, and at least one hit is returned for a known simple chemistry question.

2. Bot smoke test.
   Confirm that the RAG-enabled bot can answer a simple school-level theory question without crashing and returns a valid structured object.

3. Optional source-usage assertion.
   For selected questions, verify that the answer includes at least one source ID when RAG is enabled and relevant.

4. Fixture-based retrieval check.
   Use tracked fixtures such as `config/introchem_retrieval_queries.jsonl` for lightweight repeated sanity checks.

### Fast Model Choice For Smoke Tests

The default heavy smoke-test model should not be `gemma4:12b`.

For practical routine checks, the plan should assume:

- use `phi4-mini` for QA smoke testing
- keep heavier models optional for manual comparison, not for the fastest routine validation path

This keeps the sanity-check loop short enough to be used regularly during cleanup.

### Testing UX Goal

The project should move toward a simple developer experience:

- run one smoke-test command
- wait a short time
- see a clear `OK`-style result
- continue working with confidence

This is especially important for an experimental educational bot project where prompt, retrieval, and structure will keep evolving.

## Smoke-Test And Confidence Workflow

The repository should include a lightweight test path whose purpose is confidence, not exhaustive validation.

The ideal workflow is:

- run one command or one IDE-friendly action
- execute retrieval smoke tests based on tracked fixtures such as `config/introchem_retrieval_queries.jsonl`
- see a clear final status such as `OK`
- feel safe changing prompts, retrieval settings, or documentation and re-running the check

The public documentation should frame this as a friendly experimentation workflow:

- the project is stable enough to verify quickly
- prompt and retrieval changes are expected
- the repository is designed for exploration and iteration, not only static presentation

## Corpus Test Fixtures

Small tracked fixtures that support the smoke-test workflow should remain in Git.

For now this includes files such as:

- `config/introchem_retrieval_queries.jsonl`

These files should be documented as reproducibility fixtures and local sanity checks, not as production data.

## Module Docstring Standard

Every active Python module should start with a short docstring that answers:

- what this module does
- whether it is part of the bot, retrieval pipeline, or prompt experimentation layer
- its main inputs and outputs
- whether it is an executable CLI module or an importable library module

This is especially important for:

- the bot CLI module
- the parser
- the chunk builder
- the vector search/index builder
- the prompt garden module
- the evaluation helper module

## Function And Class Docstring Standard

Public classes and non-trivial functions should document:

- purpose
- arguments
- return values
- side effects such as file writes, network calls, or model calls
- assumptions important for chemistry safety or RAG correctness

Docstrings should explain intent, not repeat the implementation line by line.

## Comment Policy

Comments should be used sparingly and only where they add real value.

Good comment targets in this project:

- chemistry-safety constraints
- parsing heuristics for messy source XHTML
- retrieval metadata design decisions
- prompt-materialization decisions
- reasons why a generated artifact is written in a specific format

Comments should not narrate obvious Python code.

## Notebook Policy

This repository should not treat the Prompt Garden notebook as a disposable or secondary artifact.

For this project, the Prompt Garden notebook is part of the functional surface area. It should be positioned and documented as:

- an official operator console for prompt management
- a supported interface for prompt versioning and experiment workflows
- part of the Prompt Garden subsystem, not a detached scratchpad

Recommended presentation:

- the control notebook lives under `prompt_garden/control/`
- `prompt_garden/README.md` explains when to use the notebook versus CLI scripts
- notebook outputs are curated intentionally
- the repository still explains the architecture without requiring a notebook deep dive

## Temporary Local Archive Policy

At the moment `archive/` is acceptable only as a short-term local holding area for old versions and reference material.

The plan should state this clearly:

- `archive/` is not part of the intended final repository design
- it exists only to help the transition while the project is being cleaned up
- nothing in `archive/` should be presented as active functionality
- once the needed reference value is gone, the folder should be removed entirely

## Root Cleanliness Standard

The root directory should contain only high-signal files:

- main README
- license
- dependency entry files
- environment example
- clearly named top-level folders

The root should not contain:

- runtime logs
- unrelated experiments
- duplicate readmes
- obsolete prototypes posing as active entry points

## Resume-Ready Quality Bar

After reorganization, a person opening the repository should understand within a few minutes:

- what the project is
- which file or command starts the main bot
- where the RAG pipeline lives
- how prompt experimentation fits into the project
- what data is needed locally
- which folders are source, generated, configuration, notebook, and prompt-history related

That clarity is the main goal of this plan.

## Out Of Scope

This plan does not cover:

- model quality improvements
- prompt redesign
- retrieval quality tuning
- code optimization
- feature roadmap
- test strategy expansion beyond structural placement and documentation expectations
