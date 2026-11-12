"""State-constrained dialogue-relation classification for follow-up turns.

This module answers one narrow question only: how does the current student
message relate to the tutor's previous question? Python first derives which
relation labels are possible from the previous tutor move. The small LLM then
chooses one label from that reduced set.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz, process, utils

from .schemas import QuestionMainType, TurnRelation

TURN_RELATION_LABELS: tuple[TurnRelation, ...] = (
    "student_answer",
    "clarification_choice",
    "new_question",
    "unclear",
)

DEFAULT_TURN_RELATION_PROMPT = """You are a fast dialogue-relation classifier for a chemistry tutor.

Decide only how CURRENT_STUDENT_MESSAGE relates to LAST_TUTOR_QUESTION.
Python provides ALLOWED_LABELS for the current dialogue state.
Return exactly one label from ALLOWED_LABELS and nothing else.
Do not answer the chemistry question. Do not return JSON, explanations,
confidence, scores, or safety decisions.

Definitions:
- student_answer: the student attempts to answer the tutor's check or diagnostic question.
- clarification_choice: the student selects an option that the tutor explicitly offered.
- new_question: the student asks a different chemistry question or explicitly postpones the old answer.
- unclear: the relation cannot be determined from the two messages.

Important:
- Never return a label that is absent from ALLOWED_LABELS.
- A full chemistry question is new_question, even when the tutor was waiting for an answer.
- "I will answer later" followed by another request is new_question.
"""


class ThinTurnRelation(BaseModel):
    relation: TurnRelation


_GENERIC_CLARIFICATION_CHOICES: dict[QuestionMainType, tuple[str, ...]] = {
    "reference_closed": (
        "quick factual answer",
        "factual answer",
        "quick answer",
        "definition",
        "just the fact",
    ),
    "theory_learning_open": (
        "step by step explanation",
        "step-by-step explanation",
        "detailed explanation",
        "explain the concept",
        "help me understand",
    ),
    "calculation_request": (
        "help with a chemistry calculation",
        "calculation",
        "solve the problem",
        "show the calculation",
    ),
    "reaction_illustration_request": (
        "reaction concept",
        "chemical reaction concept",
        "reaction equation",
        "show a reaction",
    ),
}

_SAFE_REDIRECT_CHOICES: dict[QuestionMainType, tuple[str, ...]] = {
    "reaction_illustration_request": (
        "safe conceptual explanation",
        "safe conceptual explanation of the reaction",
        "conceptual explanation of the reaction",
        "explain the reaction safely",
        "reaction principles",
        "redox roles",
        "oxidation and reduction roles",
    ),
    "theory_learning_open": (
        "why the materials are hazardous",
        "why it is hazardous",
        "why it is dangerous",
        "explain the hazards",
        "chemical risks",
        "safety risks",
    ),
}

_NEW_QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "which ",
    "can ",
    "could ",
    "is ",
    "are ",
    "do ",
    "does ",
    "explain ",
    "please explain ",
    "tell me ",
    "tell me about ",
    "definition of ",
    "simple definition ",
    "give me a definition ",
    "i need to know ",
    "now i need",
    "instead i need",
    "another question",
    "i read that ",
    "новый вопрос",
    "новий вопрос",
    "теперь мне нужно",
    "зараз мені потрібно",
)

_POSTPONE_PHRASES = (
    "i will answer later",
    "i will answer you later",
    "i'll answer later",
    "not now",
    "answer later",
    "отвечу позже",
    "відповім пізніше",
)

_ANSWER_EXPECTING_MOVES = frozenset(
    {
        "brief_answer_plus_check",
        "feedback_on_student_answer",
        "diagnostic_question",
    }
)


def _normalize(text: str) -> str:
    processed = utils.default_process(str(text or ""))
    return " ".join((processed or "").split())


def turn_relation_context(session_state: dict[str, Any] | None) -> str:
    """Return a stable Python-derived description of the previous tutor move."""
    move = str(dict(session_state or {}).get("last_tutor_move") or "")
    if move == "clarification_question":
        return "clarification_selection"
    if move == "safe_redirect":
        return "safe_redirect_selection"
    if move == "off_topic_redirect":
        return "chemistry_topic_selection"
    if move in _ANSWER_EXPECTING_MOVES:
        return "answer_to_check_question"
    return "generic_awaited_reply"


def allowed_turn_relations(
    session_state: dict[str, Any] | None,
) -> tuple[TurnRelation, ...]:
    """Derive labels that are logically possible after the previous tutor move."""
    move = str(dict(session_state or {}).get("last_tutor_move") or "")
    if move in {"clarification_question", "safe_redirect"}:
        return ("clarification_choice", "new_question", "unclear")
    if move == "off_topic_redirect":
        return ("new_question", "unclear")
    if move in _ANSWER_EXPECTING_MOVES:
        return ("student_answer", "new_question", "unclear")
    return ("student_answer", "new_question", "unclear")


def parse_turn_relation_label(
    raw_text: str,
    *,
    allowed_labels: tuple[TurnRelation, ...] | None = None,
) -> ThinTurnRelation:
    """Parse one label and reject labels impossible in the current state."""
    allowed = tuple(allowed_labels or TURN_RELATION_LABELS)
    if not allowed:
        raise ValueError("Turn-relation classifier received no allowed labels.")

    normalized = _normalize(raw_text).replace(" ", "_")
    if normalized in allowed:
        return ThinTurnRelation(relation=normalized)  # type: ignore[arg-type]

    plain = _normalize(raw_text)
    matches = [
        label
        for label in allowed
        if re.search(rf"\b{re.escape(label.replace('_', ' '))}\b", plain)
        or re.search(rf"\b{re.escape(label)}\b", str(raw_text).lower())
    ]
    if len(matches) == 1:
        return ThinTurnRelation(relation=matches[0])

    allowed_text = ", ".join(allowed)
    raise ValueError(
        "Turn-relation classifier did not return one state-allowed label "
        f"({allowed_text}): {raw_text!r}"
    )


def should_classify_turn_relation(session_state: dict[str, Any] | None) -> bool:
    state = dict(session_state or {})
    return bool(
        state.get("awaiting_student_answer")
        and str(state.get("last_bot_question") or "").strip()
    )


def build_turn_relation_payload(
    user_text: str,
    session_state: dict[str, Any] | None,
) -> dict[str, Any]:
    state = dict(session_state or {})
    return {
        "RELATION_CONTEXT": turn_relation_context(state),
        "ALLOWED_LABELS": list(allowed_turn_relations(state)),
        "LAST_TUTOR_MOVE": state.get("last_tutor_move"),
        "LAST_TUTOR_QUESTION": state.get("last_bot_question"),
        "CURRENT_STUDENT_MESSAGE": user_text,
        "excluded_context": [
            "full chat history",
            "RAG excerpts",
            "safety labels",
            "source decisions",
        ],
    }


def turn_relation_input_text(
    user_text: str,
    session_state: dict[str, Any] | None,
) -> str:
    return (
        "Choose exactly one relation label from ALLOWED_LABELS.\n"
        + json.dumps(
            build_turn_relation_payload(user_text, session_state),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _choice_groups_for_move(
    last_tutor_move: str | None,
) -> dict[QuestionMainType, tuple[str, ...]]:
    if last_tutor_move == "safe_redirect":
        return _SAFE_REDIRECT_CHOICES
    return _GENERIC_CLARIFICATION_CHOICES


def resolve_clarification_choice_main_type(
    user_text: str,
    *,
    last_tutor_move: str | None = None,
    score_cutoff: float = 72.0,
) -> tuple[QuestionMainType | None, dict[str, Any]]:
    """Map an offered option choice to an operational main type with RapidFuzz."""
    normalized = _normalize(user_text)
    if not normalized:
        return None, {
            "matched": False,
            "score": None,
            "choice": None,
            "last_tutor_move": last_tutor_move,
        }

    groups = _choice_groups_for_move(last_tutor_move)
    flattened: list[tuple[QuestionMainType, str]] = [
        (main_type, phrase)
        for main_type, phrases in groups.items()
        for phrase in phrases
    ]
    choices = [phrase for _, phrase in flattened]
    match = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        processor=utils.default_process,
        score_cutoff=score_cutoff,
    )
    if match is None:
        return None, {
            "matched": False,
            "score": None,
            "choice": None,
            "last_tutor_move": last_tutor_move,
        }

    matched_phrase, score, index = match
    main_type = flattened[int(index)][0]
    return main_type, {
        "matched": True,
        "score": float(score),
        "choice": matched_phrase,
        "main_type": main_type,
        "last_tutor_move": last_tutor_move,
    }


def _looks_like_new_question(user_text: str) -> bool:
    normalized = _normalize(user_text)
    if not normalized:
        return False
    return (
        "?" in user_text
        or normalized.startswith(_NEW_QUESTION_PREFIXES)
        or any(phrase in normalized for phrase in _POSTPONE_PHRASES)
    )


def rule_based_turn_relation(
    user_text: str,
    session_state: dict[str, Any] | None,
) -> ThinTurnRelation:
    """State-aware fallback used when the tiny relation-model call fails."""
    state = dict(session_state or {})
    if not should_classify_turn_relation(state):
        return ThinTurnRelation(relation="new_question")

    allowed = allowed_turn_relations(state)
    last_tutor_move = str(state.get("last_tutor_move") or "")

    if "clarification_choice" in allowed:
        choice, _ = resolve_clarification_choice_main_type(
            user_text,
            last_tutor_move=last_tutor_move,
        )
        if choice is not None:
            return ThinTurnRelation(relation="clarification_choice")

    if "new_question" in allowed and _looks_like_new_question(user_text):
        return ThinTurnRelation(relation="new_question")

    # After an off-topic redirect, any nonempty chemistry topic is a new request,
    # even when it is only one or two words such as "atoms".
    if last_tutor_move == "off_topic_redirect" and "new_question" in allowed:
        if _normalize(user_text):
            return ThinTurnRelation(relation="new_question")

    word_count = len(re.findall(r"\w+", _normalize(user_text), flags=re.UNICODE))
    if "student_answer" in allowed and 1 <= word_count <= 60:
        return ThinTurnRelation(relation="student_answer")
    return ThinTurnRelation(relation="unclear")


def apply_turn_relation(
    main_type: QuestionMainType,
    *,
    relation: TurnRelation | None,
    user_text: str,
    last_tutor_move: str | None = None,
) -> tuple[QuestionMainType, list[str], dict[str, Any]]:
    """Apply one state-constrained relation label to the chemistry label."""
    changes: list[str] = []
    details: dict[str, Any] = {
        "relation": relation,
        "effective_relation": relation,
        "clarification_choice": None,
        "last_tutor_move": last_tutor_move,
    }

    if relation == "student_answer":
        if main_type != "student_answer_to_assess":
            changes.append("main_type -> student_answer_to_assess from turn_relation")
        return "student_answer_to_assess", changes, details

    if relation == "clarification_choice":
        selected, choice_debug = resolve_clarification_choice_main_type(
            user_text,
            last_tutor_move=last_tutor_move,
        )
        details["clarification_choice"] = choice_debug
        if selected is None:
            # A full chemistry label from the content classifier is stronger than
            # an unresolved option-selection label. Preserve the content route and
            # record the effective relation as a new question.
            if main_type != "unclear":
                details["effective_relation"] = "new_question"
                changes.append(
                    "turn_relation clarification_choice -> new_question because no offered option matched"
                )
                return main_type, changes, details
            changes.append("main_type remains unclear because clarification choice was not resolved")
            return "unclear", changes, details
        if main_type != selected:
            changes.append(f"main_type -> {selected} from clarification_choice")
        return selected, changes, details

    # new_question deliberately preserves the chemistry-content classifier label.
    # unclear also preserves a clear content label; if the content classifier itself
    # returned unclear, the normal clarification route remains in force.
    return main_type, changes, details


__all__ = [
    "DEFAULT_TURN_RELATION_PROMPT",
    "TURN_RELATION_LABELS",
    "ThinTurnRelation",
    "TurnRelation",
    "allowed_turn_relations",
    "apply_turn_relation",
    "build_turn_relation_payload",
    "parse_turn_relation_label",
    "resolve_clarification_choice_main_type",
    "rule_based_turn_relation",
    "should_classify_turn_relation",
    "turn_relation_context",
    "turn_relation_input_text",
]
