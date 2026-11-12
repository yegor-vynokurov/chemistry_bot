# PR 0: Tutor Graph Architecture

## Status

Historical draft written before implementation. The tutor graph now exists under `src/chemistry_bot/tutor_graph/`; this document records its design rationale, not qualification of the current runtime. The Compose run path is described in [compose-prototype.md](compose-prototype.md).

## Purpose

The current `rag_bot.py` remains the stable reference-oriented chemistry bot. It answers chemistry questions using local RAG, Wikipedia enrichment, prompt logging, and structured output.

This PR defines a separate future bot: `chemistry_tutor_graph`. Its goal is not to replace the reference bot, but to become a learning-oriented dialogue tutor that remembers a student's learning history, asks diagnostic questions when useful, tracks skill mastery over time, and safely improves its teaching materials through reviewed proposals.

The key difference:

```text
rag_bot.py              → answers chemistry questions well
chemistry_tutor_graph   → teaches one student over time
```

## Guiding Principles

1. Keep the current RAG bot stable.
2. Build the tutor as a separate LangGraph-based bot.
3. Use tools for small explicit actions: save, retrieve, classify, assess, update, propose.
4. Store learning state in SQLite first.
5. Do not fine-tune model weights at this stage.
6. Do not let the model silently rewrite its own teaching materials.
7. Use human-reviewed proposals for durable changes to concept cards, learned knowledge, and prompt hints.
8. Keep memory small, curated, and distinct from raw chat history.
9. Use deterministic formulas for long-term skill updates whenever possible.
10. Treat chemistry safety rules as higher priority than Socratic dialogue.

## Why a Separate Bot

The existing RAG bot is already useful as a reference assistant. It should stay stable while the tutor experiment grows nearby.

The tutor bot needs a different loop:

```text
understand the request
load student state
load active concept cards
retrieve context only if needed
choose a tutor move
answer or ask a question
assess the student's answer
update skill estimates
save the learning event
optionally propose improvements
```

This is easier to represent as a graph than as one long chain.

## Architectural Inspirations

### Agno-inspired separation

Agno's architecture suggests separating persistent learning into several stores. We will implement the same idea in our own SQLite schema, not depend on Agno yet.

Mapping for this project:

```text
User Profile       → student profile: grade, language, preferences
User Memory        → curated durable observations about the student
Session Context    → current active lesson state
Entity Memory      → chemistry concepts and relations
Learned Knowledge  → reusable teaching insights across students
Decision Log       → why the bot chose a route or tutor move
```

### Hermes-inspired guardrails

Hermes suggests two useful ideas:

1. Memory should be limited and curated, not a dump of all chat history.
2. Durable self-improvement should go through approval.

For this project:

```text
Raw chat history       → stored in turns
Curated student memory → stored separately
Concept card updates   → staged as pending proposals
Learned tutor notes     → staged as pending proposals
Prompt hints            → staged as pending proposals
```

The model may propose durable changes, but it should not silently apply them.

## Core Stores

The first SQLite database can live at:

```text
data/tutor_graph/tutor.db
```

Proposed store groups:

```text
students
sessions
turns
session_state
student_profile
student_memory
student_skill_state
concept_cards
concept_relations
learned_knowledge
decision_log
retrieval_events
assessment_events
pending_proposals
question_classifications
```

Not all tables need to be created at once. They define the direction.

## Memory Types

### Raw history

Raw messages from the student and tutor.

Use for:

```text
recent context
conversation replay
debugging
training datasets later
```

Do not inject all raw history into prompts.

### Session state

Small mutable state for the current learning session.

Example:

```json
{
  "active_concept_id": "covalent_bond",
  "last_bot_question": "What do atoms share in a covalent bond?",
  "awaiting_student_answer": true,
  "last_tutor_move": "diagnostic_question"
}
```

### Student memory

Curated durable observations about this student.

Examples:

```text
The student often describes chemical bonds as atoms being close together.
The student needs extra support distinguishing electron sharing from transfer.
The student responds well to short examples before terminology.
```

### Student skill state

Numerical estimates of mastery by concept and skill.

Example skills:

```text
theory
examples
equations
reaction_balancing
safety
```

Example state:

```text
student_id: user_123
concept_id: covalent_bond
skill_name: theory
mastery_probability: 0.61
evidence_count: 5
```

### Concept cards

A local structured teaching wiki for the bot. Concept cards are not written all at once. They appear and improve as the tutor is used.

Example card:

```json
{
  "id": "covalent_bond",
  "title": "Covalent bond",
  "canonical_definition": "A covalent bond is a chemical bond in which atoms share one or more pairs of electrons.",
  "minimal_understanding": [
    "mentions atoms",
    "mentions sharing electrons",
    "does not confuse sharing with full transfer"
  ],
  "common_misconceptions": [
    "bond means atoms are simply close",
    "electrons are fully transferred",
    "covalent always means weak"
  ],
  "examples": [
    "H2: two hydrogen atoms share one pair of electrons",
    "Cl2: two chlorine atoms share one pair of electrons",
    "H2O: oxygen shares electron pairs with hydrogen atoms"
  ],
  "diagnostic_questions": [
    "What is being shared in a covalent bond?",
    "How is sharing different from electron transfer?"
  ]
}
```

### Learned knowledge

Reusable teaching insights across students.

Examples:

```text
For covalent_bond, students often need contrast with ionic_bond early.
If a student defines a bond only as closeness, ask about electrons before giving examples.
For ionic vs covalent, reaction examples should stay conceptual and avoid lab procedures.
```

### Decision log

A structured record of why the bot made a decision.

Example:

```json
{
  "node_name": "select_tutor_move",
  "decision_type": "tutor_move",
  "decision": "diagnostic_question",
  "confidence": 0.82,
  "reason": "The question is broad and the student's mastery for covalent_bond is unknown."
}
```

This supports debugging and future evaluation.

## Learning Modes

Borrowing the idea of different learning modes, the tutor should use three levels.

### Always

Actions that can happen automatically:

```text
save turns
save retrieval events
save decision log entries
save assessment events
update numerical skill state after assessment
```

### Agentic

Actions the bot may choose when appropriate:

```text
retrieve RAG context
retrieve Wikipedia context
create draft concept card
ask a diagnostic question
ask for an example
switch from Socratic mode to direct explanation
```

### Propose

Durable self-improvements that require review:

```text
concept card patches
student memory updates
learned knowledge updates
prompt hints
classifier label corrections
```

## Request Classification

A classifier node should route each user input before generation.

Initial schema:

```python
class QuestionClassification(BaseModel):
    main_type: Literal[
        "reference_closed",
        "theory_learning_open",
        "student_answer_to_assess",
        "reaction_illustration_request",
        "experiment_request_caution",
        "calculation_request",
        "off_topic",
        "unclear",
    ]
    needs_rag: bool
    needs_wikipedia: bool
    needs_diagnostic_question: bool
    safety_level: Literal["safe", "caution", "unsafe"]
    confidence: float
    reason: str
```

First implementation can use a small LLM with structured output. Later, stored classifications can train a SetFit classifier.

Routing examples:

```text
"When was Mendeleev born?"
→ reference_closed, needs_wikipedia=True

"What is a covalent bond?"
→ theory_learning_open, needs_rag=True, needs_diagnostic_question=True

"Can I get hydrogen from magnesium and acid?"
→ reaction_illustration_request or experiment_request_caution, safety_level=caution

"Na atomic mass?"
→ reference_closed

"I think a chemical bond is when atoms are close."
→ student_answer_to_assess
```

If classification confidence is low, the bot should ask a clarifying question instead of guessing.

## Tutor Moves

The bot should not always answer with a question. It should choose a tutor move.

Proposed moves:

```text
direct_answer
brief_answer_plus_check
diagnostic_question
socratic_hint
correct_misconception
ask_for_example
give_reaction_illustration
safe_refusal_or_redirect
mini_quiz
```

Routing rules for the first version:

```text
reference_closed + high confidence
→ direct_answer

theory_learning_open + unknown mastery
→ diagnostic_question

student_answer_to_assess + misconception found
→ correct_misconception

student_answer_to_assess + partial understanding
→ socratic_hint

student_answer_to_assess + good definition but weak examples
→ ask_for_example

reaction_illustration_request + safety caution
→ give_reaction_illustration

unsafe experiment request
→ safe_refusal_or_redirect
```

## Assessment and Skill Updates

LLM assessment should evaluate the student's answer against concept cards.

Assessment schema:

```python
class StudentAnswerAssessment(BaseModel):
    concept_id: str
    theory_score: float
    example_score: float
    equation_score: float
    balancing_score: float
    safety_score: float
    overall_score: float
    misconceptions: list[str]
    missing_pieces: list[str]
    next_focus: str
    confidence: float
```

The LLM evaluates the current answer. A deterministic update formula updates long-term mastery.

Initial soft Bayesian update:

```python
def soft_bkt_update(
    p_mastery: float,
    score: float,
    learn: float = 0.15,
    guess: float = 0.20,
    slip: float = 0.10,
) -> float:
    p_correct_if_mastered = 1.0 - slip
    p_correct_if_unmastered = guess
    p_wrong_if_mastered = slip
    p_wrong_if_unmastered = 1.0 - guess

    likelihood_if_mastered = (
        score * p_correct_if_mastered
        + (1.0 - score) * p_wrong_if_mastered
    )
    likelihood_if_unmastered = (
        score * p_correct_if_unmastered
        + (1.0 - score) * p_wrong_if_unmastered
    )

    denominator = (
        p_mastery * likelihood_if_mastered
        + (1.0 - p_mastery) * likelihood_if_unmastered
    )
    if denominator == 0:
        return p_mastery

    posterior = p_mastery * likelihood_if_mastered / denominator
    return posterior + (1.0 - posterior) * learn
```

This avoids vague “the bot thinks the student improved” logic. The model scores an answer; the formula updates the stored skill probability.

## RAG and Wikipedia

The current RAG and Wikipedia logic from `rag_bot.py` can be reused, but in the tutor graph they should be graph nodes, not free agent tools at first.

The classifier decides:

```text
needs_rag=True        → retrieve textbook context
needs_wikipedia=True  → retrieve reference context
```

The tutor should use context only when it serves the current tutor move.

## Reaction Illustrations

The tutor needs a safe alternative to full experiment plans.

Schema:

```python
class ReactionIllustration(BaseModel):
    title: str
    equation: str | None
    concept_focus: str
    explanation: str
    not_a_procedure: bool = True
    safety_note: str
```

Purpose:

```text
Show real chemistry as an illustration.
Avoid quantities, concentrations, heating times, apparatus, and step-by-step procedures.
```

This is especially important for prompts like:

```text
"Explain covalent and ionic bonds with examples as experiment reactions."
```

The correct response mode should be reaction illustration, not laboratory protocol.

## Pending Proposals

Durable changes should be staged first.

Proposal types:

```text
concept_card_patch
student_memory_update
learned_knowledge_update
prompt_hint_update
classifier_label_correction
```

Commands later:

```text
/proposals
/proposal show <id>
/proposal approve <id>
/proposal reject <id>
```

This keeps self-improvement safe and reviewable.

## Initial Graph Shape

First useful graph:

```text
START
  ↓
load_recent_history
  ↓
load_session_state
  ↓
classify_question
  ↓
route_by_classification
  ├─ ask_clarifying_question
  ├─ direct_reference_answer
  ├─ learning_dialogue
  ├─ assess_student_answer
  ├─ reaction_illustration
  └─ safety_redirect
  ↓
save_turn
  ↓
save_decision_log
  ↓
END
```

Later graph:

```text
START
  ↓
load_student_snapshot
  ↓
classify_question
  ↓
load_or_create_concept_cards
  ↓
maybe_retrieve_rag
  ↓
maybe_retrieve_wiki
  ↓
if student_answer: assess_student_answer → update_skill_state
  ↓
select_tutor_move
  ↓
generate_response
  ↓
save_learning_event
  ↓
maybe_propose_memory_or_concept_updates
  ↓
END
```

## Debug Commands to Add Later

```text
/why
/debug last
/progress
/concepts
/concept <id>
/proposals
/review-session
/export-classifier-dataset
```

`/why` should show operational decisions, not hidden chain of thought.

Example:

```text
classification: theory_learning_open, confidence 0.82
active concept: covalent_bond
student theory mastery: 0.31
tutor move: diagnostic_question
reason: topic is broad and mastery is unknown
```

## Future SetFit Classifier

Do not start with SetFit as the first classifier. Start with an LLM classifier and save its structured outputs. Add human corrections when needed.

Later:

```text
/export-classifier-dataset
scripts/train_question_classifier_setfit.py
```

Hybrid routing:

```text
SetFit confidence >= 0.75 → use SetFit
SetFit confidence < 0.75  → use LLM classifier
LLM confidence < 0.65     → ask clarification
```

## Non-goals for Early PRs

Do not:

```text
replace rag_bot.py
fine-tune the main LLM
make a fully autonomous self-editing agent
build a web UI
add pyBKT before enough assessment data exists
migrate to Agno before a working LangGraph tutor loop exists
```

## Success Criteria for PR 0

This PR is complete if:

```text
- the architecture document exists;
- it clearly separates the reference bot from the tutor bot;
- it defines storage/memory/knowledge/learning layers;
- it defines safe self-improvement through pending proposals;
- it explains the role of soft Bayesian skill updates;
- it gives enough direction for PR 1 without implementing runtime code.
```

## Next PR

PR 1 should create the empty LangGraph tutor bot skeleton:

```text
src/chemistry_bot/tutor_graph/
  __init__.py
  app.py
  graph.py
  state.py
  schemas.py
```

With a minimal graph:

```text
START → generate_response → END
```

and a CLI entry point:

```powershell
python -m chemistry_bot.tutor_graph.app
```
