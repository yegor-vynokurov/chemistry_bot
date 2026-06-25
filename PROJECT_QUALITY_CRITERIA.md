# Project Quality Criteria

## Purpose

This document defines the stable quality criteria for the repository.

Use it as a reference when reviewing changes to:

- repository structure
- documentation
- file placement
- naming
- prompt artifacts
- reproducibility
- smoke-test coverage

This file is intentionally separate from the execution roadmap. The roadmap describes what to do next. This file describes what "good enough" looks like.

## 1. Product Clarity

The repository should clearly communicate all of the following:

- the project is an English-language chemistry assistant
- the intended audience is middle-school and high-school students
- the project is not positioned as a university or graduate-level chemistry tutor
- the current main experimental bot direction is the RAG-enabled path
- Prompt Garden is a core subsystem, not an incidental helper

Pass condition:
Someone opening the repository can understand the product, audience, and current main architecture within a few minutes.

## 2. Repository Story

The repository should present one coherent public story:

- a school-focused local-LLM chemistry bot
- a reproducible textbook-to-RAG pipeline
- a visible prompt-versioning and experiment-management subsystem

Pass condition:
The README, folder layout, and file names all reinforce the same story rather than competing stories.

## 3. Active Architecture Clarity

The active architecture should be distinguishable from older experiments.

Required clarity:

- `src/chemistry_bot/cli/rag_bot.py` is presented as the main experimental bot entry path
- `src/chemistry_bot/retrieval/introchem_rag.py` is presented as the RAG integration layer for the bot
- `src/chemistry_bot/cli/legacy_bot.py` remains clearly labeled as an older parallel variant while it still exists
- `scripts/run_chemistry_bot_rag.py` can serve as the stable public launcher while internal module placement continues to evolve
- temporary legacy materials are not presented as active architecture

Pass condition:
There is no ambiguity about which files represent the current direction of the project.

## 4. Folder Structure Quality

The repository should move toward a near-final layout even before the project is fully finished.

Expected structural qualities:

- top-level folders have clear roles
- active code is separated from documentation, generated data, and runtime artifacts
- Prompt Garden remains a first-class top-level subsystem
- the control notebook lives with Prompt Garden rather than as a detached scratch file
- stable user-facing scripts can exist separately from the still-evolving internal module layout
- temporary archive material is not part of the final intended structure

Pass condition:
Most active files already live in places that would still make sense later, reducing future large disruptive moves.

## 5. Root Directory Quality

The root directory should contain only high-signal files and folders.

Should usually remain in root:

- main README
- quality criteria
- planning and architecture documents
- top-level project subsystems
- environment and dependency entry files

Should not remain as root noise:

- runtime logs
- duplicate readmes
- unrelated experiments
- ambiguous temporary files
- old prototypes that look like active entry points

Pass condition:
The root feels intentional rather than crowded.

## 6. README Quality

The main README should be short, accurate, and useful to a public reader.

It should cover:

- what the project is
- who it is for
- current main architecture
- how to run the main bot
- which script is the recommended public entry point
- how to rebuild or obtain the corpus
- how Prompt Garden fits into the project
- how to run the basic sanity-check path

Pass condition:
A new reader can get oriented without opening many other files first.

## 7. Documentation Quality

Documentation should be consistent, English-first, and UTF-8 clean.

Expected qualities:

- one canonical root README
- supporting docs live in predictable places
- no contradictory setup instructions
- no typo-like documentation filenames in the final public repo
- folder-level docs exist where structure needs explanation

Pass condition:
Documentation supports the architecture instead of creating confusion.

## 8. Naming Quality

Names should communicate purpose and status.

Expected qualities:

- active names are descriptive and English
- experiment status is explicit where relevant
- obsolete version suffixes are avoided in the long term unless truly needed
- temporary legacy files are clearly marked or removed

Pass condition:
A filename usually tells the reader what the file is for and whether it is current.

## 9. Prompt Garden Quality

Prompt Garden should be visible as a meaningful project asset.

Expected qualities:

- prompt source texts are organized and understandable
- graph lineage between prompts, combos, and experiments is preserved where useful
- curated prompt exemplars are tracked in Git
- bulky low-signal run artifacts are managed selectively
- the control notebook is documented as a supported operator interface

Pass condition:
Prompt engineering work is visible, reproducible, and not buried behind ignore rules.

## 10. Curated Prompt Baseline

The repository should track a curated baseline of important prompt assets.

Current baseline target:

- `combo_000014`
- `sys_000006`
- `usr_000006`
- `fsh_000002`

Current tracked Prompt Garden control surface:

- `prompt_garden/control/prompt_garden_experiments_control.ipynb`

Current tracked curated baseline location:

- `prompt_garden/curated/`

Pass condition:
The repo includes enough real prompt artifacts to demonstrate the working baseline and future evolution path.

## 11. Data Provenance And Licensing

The repository should be explicit about its corpus and distribution rules.

Required clarity:

- the primary source is `Introductory Chemistry, 1st Canadian Edition`
- the license or redistribution terms are checked and documented
- if redistribution is allowed, the repository may include the source textbook file
- if redistribution is not allowed, the repository provides a stable acquisition path

Pass condition:
There is no silent ambiguity about source ownership or redistribution.

## 12. Reproducibility Quality

The project should be reproducible without needing undocumented manual steps.

Expected qualities:

- the corpus build process is documented
- the project exposes one public-facing build command for the textbook-to-RAG path
- generated data can be regenerated reliably
- lightweight fixtures stay tracked when they support sanity checks

Pass condition:
Another person can rebuild the main data pipeline with clear instructions.

## 13. Testing Quality

Testing is currently expected to be lightweight but real.

Required near-term qualities:

- at least one retrieval smoke test exists or is clearly planned
- at least one end-to-end QA smoke test exists or is clearly planned
- the QA smoke path uses a practical local model such as `phi4-mini`
- Prompt Garden testing is explicitly deferred rather than forgotten

Pass condition:
The project has a realistic way to confirm that core question-answer functionality still works.

## 14. Testing UX

The sanity-check experience should be simple.

Desired qualities:

- a short command or small number of commands
- fast enough to run during ordinary cleanup work
- clear pass/fail output
- fixtures that support repeated verification

Pass condition:
Running the core sanity check feels routine, not burdensome.

## 15. Code Documentation Quality

Active code should be understandable without reverse-engineering every file.

Expected qualities:

- active modules have meaningful module docstrings
- public classes and important functions have useful docstrings
- comments explain non-obvious reasoning, not obvious syntax
- architecture-sensitive files document their role in the system

Pass condition:
The codebase communicates intent, not only implementation.

## 16. Interim Good-State Standard

Because the project is still evolving, quality should be judged by whether it is in a strong intermediate state, not by whether it is final.

Good interim state means:

- the architecture is coherent
- the repository is honest about experimental areas
- the main bot flow is verifiable
- future changes can be incremental instead of disruptive

Pass condition:
The project is already credible and maintainable even before it is feature-complete.

## 17. Out Of Scope For This Quality Pass

These are not required yet for the current repository-quality milestone:

- final code optimization
- complete test coverage
- Prompt Garden automated test coverage
- full packaging or release engineering
- final architectural convergence of all experiments

Pass condition:
The team can defer these items intentionally without weakening the structural cleanup effort.
