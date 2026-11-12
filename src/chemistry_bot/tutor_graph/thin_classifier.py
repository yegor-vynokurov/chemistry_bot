"""Thin classification and deterministic policy derivation.

The content LLM has one job: emit one ``main_type`` label. Dialogue relation,
safety, source flags, and audit reasons are computed by separate narrow components.
"""

from __future__ import annotations

import re
from typing import Any

from .schemas import (
    QuestionClassification,
    QuestionMainType,
    ThinQuestionClassification,
    TurnRelation,
)
from .semantic_safety import SafetyAssessment
from .turn_relation import apply_turn_relation


MAIN_TYPE_LABELS: tuple[QuestionMainType, ...] = (
    "reference_closed",
    "theory_learning_open",
    "student_answer_to_assess",
    "reaction_illustration_request",
    "experiment_request_caution",
    "calculation_request",
    "off_topic",
    "unclear",
)

_EXPLICIT_USER_DOCUMENT_PHRASES = (
    "uploaded book",
    "uploaded books",
    "uploaded document",
    "uploaded documents",
    "uploaded file",
    "uploaded files",
    "attached book",
    "attached books",
    "attached document",
    "attached documents",
    "attached file",
    "attached files",
    "downloaded book",
    "downloaded books",
    "downloaded document",
    "downloaded documents",
    "downloaded file",
    "downloaded files",
    "according to the book",
    "according to the books",
    "according to the textbook",
    "according to the uploaded",
    "according to uploaded",
    "according to downloaded",
    "according to the downloaded",
    "from the book",
    "from the books",
    "from the document",
    "from the documents",
    "from the file",
    "from the files",
    "from the folder",
    "in this book",
    "in these books",
    "in the textbook",
    "in the uploaded book",
    "in the uploaded books",
    "in the downloaded book",
    "in the downloaded books",
    "in the downloaded file",
    "in the downloaded files",
    "in the folder",
    "documents in the folder",
    "books in the folder",
    "files in the folder",
    "what does the book say",
    "what do the books say",
    "what does the document say",
    "what do the documents say",
    "what does the file say",
    "what do the files say",
    "use the book",
    "use the books",
    "use the document",
    "use the documents",
    "use the file",
    "use the files",
    "загруженной книге",
    "загруженной книги",
    "загруженных книгах",
    "загруженном документе",
    "загруженных документах",
    "скачанной книге",
    "скачанных книгах",
    "скачанном документе",
    "скачанных документах",
    "скачанным документам",
    "скачанным книгам",
    "скачанным файлам",
    "по скачанным документам",
    "по скачанным книгам",
    "по скачанным файлам",
    "в папке",
    "книгах в папке",
    "документах в папке",
    "файлах в папке",
    "прикреплённой книге",
    "прикрепленной книге",
    "прикреплённых книгах",
    "прикрепленных книгах",
    "у цій книзі",
    "у цих книгах",
    "у завантаженій книзі",
    "у завантажених книгах",
    "у завантаженому документі",
    "у завантажених документах",
    "у папці",
    "книгах у папці",
    "документах у папці",
    "файлах у папці",
    "за файлами у папці",
    "за документами у папці",
    "за книгами у папці",
    "по завантажених документах",
    "по завантажених книгах",
    "згідно з книгою",
    "згідно з книгами",
    "згідно з документом",
    "згідно з документами",
)

_EXPLOSIVE_TERMS = (
    "black powder",
    "gunpowder",
    "explosive",
    "explosives",
    "bomb",
    "detonation",
    "nitrocellulose",
    "нитроцеллюлоз",
    "чёрный порох",
    "черный порох",
)

_HAZARDOUS_TERMS = (
    "chlorine gas",
    "hydrogen gas",
    "poison",
    "toxic gas",
    "concentrated acid",
    "sulfuric acid",
    "sulphuric acid",
    "nitric acid",
    "hydrochloric acid",
    "potassium nitrate",
    "ammonium nitrate",
)

_PROCEDURAL_TERMS = (
    "how to make",
    "how do i make",
    "how can i make",
    "how to synthesize",
    "how to obtain",
    "how to extract",
    "steps for",
    "step by step",
    "procedure",
    "at home",
    "grams",
    "milliliters",
    " ml",
    "temperature",
    "heat to",
    "collect the gas",
    "как сделать",
    "как изготовить",
    "как получить",
    "пошагово",
)

_REACTION_ILLUSTRATION_TERMS = (
    "reaction example",
    "example reaction",
    "show a reaction",
    "reaction equation",
    "chemical equation",
    "illustrate with a reaction",
)

_CALCULATION_TERMS = (
    "calculate",
    "balance",
    "coefficient",
    "molarity",
    "moles",
    "mass percent",
    "stoichiometry",
    "equation",
    "ph ",
    "calculate ph",
)

_NUMERIC_CALCULATION_COMMANDS = (
    "calculate",
    "compute",
    "solve",
    "determine",
    "find the",
    "how much",
    "what mass",
    "what volume",
    "what concentration",
    "how many grams",
    "how many moles",
    "titration calculation",
    "calculate ph",
    "ph of",
    "balance this equation",
    "balance the equation",
)

_QUANTITATIVE_CALCULATION_TARGETS = (
    "molarity",
    "concentration",
    "mass percent",
    "percent yield",
    "stoichiometry",
    "moles",
    "mole",
    "molar mass",
    "molecular mass",
    "formula mass",
    "mass",
    "volume",
    "ph",
    "poh",
    "ka",
    "kb",
    "equivalence point",
    "endpoint",
    "coefficient",
    "coefficients",
)

_NUMERIC_CALCULATION_UNITS = (
    "g", "mg", "kg", "ml", "l", "mol", "mmol", "m", "molar",
    "%", "ppm", "ph", "ka", "kb", "kw", "atm", "kpa", "°c", "celsius",
)

_QUALITATIVE_NON_CALCULATION_PATTERNS = (
    "why ",
    "why do ",
    "why does ",
    "explain why ",
    "explain the role ",
    "what is the role ",
    "what does ",
    "what is used ",
    "why is ",
    "why are ",
)

_REFERENCE_TERMS = (
    "atomic mass",
    "molar mass",
    "symbol of",
    "formula of",
    "when was",
    "who discovered",
    "who is",
    "density of",
    "melting point",
    "boiling point",
    "how many electrons",
)

_WIKIPEDIA_TERMS = (
    "when was",
    "who discovered",
    "who invented",
    "who is",
    "history of",
    "biography",
    "discovered by",
    "named after",
    "atomic number",
    "melting point",
    "boiling point",
    "density of",
)

_CHEMISTRY_TERMS = (
    "atom",
    "molecule",
    "bond",
    "electron",
    "proton",
    "neutron",
    "ion",
    "ionic",
    "covalent",
    "acid",
    "base",
    "salt",
    "reaction",
    "oxidation",
    "reduction",
    "redox",
    "periodic",
    "titration",
    "hydroxide",
    "carbonate",
)

_OFF_TOPIC_TERMS = (
    "football",
    "movie",
    "weather",
    "politics",
    "recipe for cake",
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _looks_like_question(text: str) -> bool:
    normalized = _normalize(text)
    return "?" in text or normalized.startswith(
        (
            "what ", "why ", "how ", "when ", "where ", "who ", "which ",
            "can ", "could ", "would ", "is ", "are ", "do ", "does ",
            "explain ", "tell ", "напомни", "объясни", "поясни",
        )
    )


def _has_calculation_unit(lower_text: str) -> bool:
    """Return true for unit-like tokens without matching inside words."""
    unit_pattern = "|".join(re.escape(unit) for unit in _NUMERIC_CALCULATION_UNITS)
    return bool(re.search(rf"(?<![A-Za-z])(?:{unit_pattern})(?![A-Za-z])", lower_text))


def _is_qualitative_why_or_role_question(lower_text: str) -> bool:
    """Spot concept/role questions that small models may mistake for calculation."""
    if _contains_any(lower_text, _QUALITATIVE_NON_CALCULATION_PATTERNS):
        return True
    # Common mixed phrasing from users: "calculate why..." should not become a
    # calculation route unless the request also gives a concrete quantitative target.
    return bool(re.search(r"\b(?:calculate|compute|solve|determine)\s+why\b", lower_text))


def has_numeric_calculation_intent(user_text: str) -> bool:
    """Return true only for requests that need quantitative/equation work.

    The thin LLM sometimes overuses ``calculation_request`` for analytical
    chemistry questions about titration, pH, concentration, or equations. Python
    keeps the calculation route only when the text asks for a numerical result,
    equation balancing, quantitative setup, or a clearly mathematical target.
    Qualitative "why/role" questions are not calculation requests just because
    they mention titration, concentration, pH, or the word "calculate".
    """
    lower = _normalize(user_text)
    has_command = _contains_any(lower, _NUMERIC_CALCULATION_COMMANDS)
    has_digit = bool(re.search(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", lower))
    has_unit = _has_calculation_unit(lower)
    has_math_symbol = any(symbol in user_text for symbol in ("=", "→", "->"))
    has_quantitative_target = _contains_any(lower, _QUANTITATIVE_CALCULATION_TARGETS)
    asks_amount = lower.startswith(("how much ", "how many ", "what mass ", "what volume "))
    explicit_setup = "set up" in lower and ("calculation" in lower or has_quantitative_target)

    if "balance" in lower and ("equation" in lower or has_math_symbol):
        return True

    qualitative_question = _is_qualitative_why_or_role_question(lower)
    if qualitative_question and not (has_digit or has_unit or has_math_symbol or asks_amount):
        return False

    if asks_amount and (has_quantitative_target or has_digit or has_unit):
        return True
    if explicit_setup:
        return True
    if has_command and (has_digit or has_unit or has_math_symbol or has_quantitative_target):
        return True
    if has_digit and (has_unit or has_quantitative_target or has_math_symbol):
        return True
    return False


def _non_calculation_fallback_type(user_text: str) -> QuestionMainType:
    lower = _normalize(user_text)
    if lower.startswith(("why ", "how ", "explain ", "tell me why", "help me understand")):
        return "theory_learning_open"
    if _is_qualitative_why_or_role_question(lower):
        return "theory_learning_open"
    if _contains_any(lower, _CHEMISTRY_TERMS):
        return "theory_learning_open" if lower.startswith(("why ", "how ")) else "reference_closed"
    return "reference_closed" if _looks_like_question(user_text) else "unclear"


def parse_main_type_label(raw_text: str) -> ThinQuestionClassification:
    """Parse one label from a small-model response without requiring JSON."""
    normalized = _normalize(raw_text).replace("`", "")
    if normalized in MAIN_TYPE_LABELS:
        return ThinQuestionClassification(main_type=normalized)  # type: ignore[arg-type]

    matches = [label for label in MAIN_TYPE_LABELS if re.search(rf"\b{re.escape(label)}\b", normalized)]
    if len(matches) == 1:
        return ThinQuestionClassification(main_type=matches[0])
    raise ValueError(f"Classifier did not return exactly one supported label: {raw_text!r}")


def rule_based_main_type(
    user_text: str,
    session_state: dict[str, Any] | None = None,
) -> ThinQuestionClassification:
    """Deterministic fallback for chemistry-content type only.

    Dialogue role is intentionally excluded. A separate turn-relation classifier
    decides whether a reply is a student answer, a clarification choice, or a new
    question.
    """
    state = dict(session_state or {})
    text = user_text.strip()
    lower = _normalize(text)

    if not text:
        return ThinQuestionClassification(main_type="unclear")

    if _contains_any(lower, _PROCEDURAL_TERMS):
        return ThinQuestionClassification(main_type="experiment_request_caution")

    if _contains_any(lower, _REACTION_ILLUSTRATION_TERMS):
        return ThinQuestionClassification(main_type="reaction_illustration_request")

    if _contains_any(lower, _CALCULATION_TERMS) and has_numeric_calculation_intent(text):
        return ThinQuestionClassification(main_type="calculation_request")

    if _contains_any(lower, _REFERENCE_TERMS):
        return ThinQuestionClassification(main_type="reference_closed")

    if _contains_any(lower, _CHEMISTRY_TERMS):
        return ThinQuestionClassification(main_type="theory_learning_open")

    if _contains_any(lower, _OFF_TOPIC_TERMS):
        return ThinQuestionClassification(main_type="off_topic")

    return ThinQuestionClassification(main_type="unclear")

def explicit_user_document_request(user_text: str) -> bool:
    return _contains_any(_normalize(user_text), _EXPLICIT_USER_DOCUMENT_PHRASES)


def derive_question_classification(
    thin: ThinQuestionClassification | QuestionMainType | str,
    *,
    user_text: str,
    session_state: dict[str, Any] | None = None,
    source: str = "llm_label",
    needs_documents_dev: bool = False,
    turn_relation: TurnRelation | None = None,
    safety_assessment: SafetyAssessment | None = None,
) -> tuple[QuestionClassification, dict[str, Any]]:
    """Assemble operational policy from thin labels and measurable signals."""
    if isinstance(thin, ThinQuestionClassification):
        main_type: QuestionMainType = thin.main_type
    else:
        main_type = ThinQuestionClassification(main_type=str(thin)).main_type

    state = dict(session_state or {})
    text = user_text.strip()
    lower = _normalize(text)
    words = re.findall(r"\w+", lower, flags=re.UNICODE)

    has_explicit_user_docs = explicit_user_document_request(text)
    main_type, relation_changes, relation_debug = apply_turn_relation(
        main_type,
        relation=turn_relation,
        user_text=text,
        last_tutor_move=str(state.get("last_tutor_move") or "") or None,
    )
    effective_turn_relation = relation_debug.get("effective_relation", turn_relation)
    changes: list[str] = list(relation_changes)

    has_numeric_intent = has_numeric_calculation_intent(text)
    if main_type == "calculation_request" and not has_numeric_intent:
        replacement_type = _non_calculation_fallback_type(text)
        changes.append(f"main_type -> {replacement_type} from no_numeric_calculation_intent")
        main_type = replacement_type

    safety_decision = safety_assessment.decision if safety_assessment is not None else "safe"
    if safety_decision == "unsafe":
        if main_type != "experiment_request_caution":
            changes.append("main_type -> experiment_request_caution from semantic safety gate")
        main_type = "experiment_request_caution"
        safety_level = "unsafe"
        changes.append("safety -> unsafe from hazard plus procedural-intent signals")
    elif main_type == "experiment_request_caution":
        safety_level = "caution"
    elif safety_decision == "caution" or main_type == "reaction_illustration_request":
        safety_level = "caution"
    else:
        safety_level = "safe"

    needs_course_rag = main_type in {
        "theory_learning_open",
        "calculation_request",
        "reaction_illustration_request",
    }
    if main_type in {"student_answer_to_assess", "off_topic", "unclear", "experiment_request_caution"}:
        needs_course_rag = False

    needs_documents_dev = bool(needs_documents_dev)
    needs_user_documents = has_explicit_user_docs or needs_documents_dev

    needs_wikipedia = (
        main_type == "reference_closed"
        and _contains_any(lower, _WIKIPEDIA_TERMS)
        and not needs_user_documents
    )

    broad_learning_request = main_type == "theory_learning_open" and (
        lower.startswith(("i don't understand", "i do not understand", "help me understand"))
        or (len(words) <= 3 and not has_explicit_user_docs)
    )
    needs_diagnostic_question = main_type == "unclear" or broad_learning_request

    reason_parts = [f"main_type={main_type}", f"source={source}"]
    if effective_turn_relation is not None:
        reason_parts.append(f"turn_relation={effective_turn_relation}")
    if has_explicit_user_docs:
        reason_parts.append("explicit_user_document_request")
    if needs_documents_dev:
        reason_parts.append("developer_document_override")
    if needs_course_rag:
        reason_parts.append("course_rag_candidate")
    if needs_wikipedia:
        reason_parts.append("encyclopedic_lookup")
    if safety_level != "safe":
        reason_parts.append(f"safety={safety_level}")
    if safety_assessment is not None:
        reason_parts.append(f"safety_gate={safety_assessment.reason}")
    if needs_diagnostic_question:
        reason_parts.append("diagnostic_or_clarification_needed")

    classification = QuestionClassification(
        main_type=main_type,
        turn_relation=effective_turn_relation,
        needs_course_rag=needs_course_rag,
        needs_user_documents=needs_user_documents,
        needs_documents_dev=needs_documents_dev,
        needs_wikipedia=needs_wikipedia,
        needs_diagnostic_question=needs_diagnostic_question,
        safety_level=safety_level,  # type: ignore[arg-type]
        reason="; ".join(reason_parts),
    )
    debug = {
        "thin_label": thin.main_type if isinstance(thin, ThinQuestionClassification) else str(thin),
        "derived": classification.model_dump(),
        "changes": changes,
        "turn_relation": relation_debug,
        "safety_assessment": (
            safety_assessment.as_dict() if safety_assessment is not None else None
        ),
        "signals": {
            "explicit_user_document_request": has_explicit_user_docs,
            "needs_documents_dev": needs_documents_dev,
            "turn_relation": effective_turn_relation,
            "turn_relation_raw": turn_relation,
            "semantic_safety_decision": safety_decision,
            "hazard_detected": (
                safety_assessment.hazard_detected if safety_assessment is not None else False
            ),
            "procedural_intent_detected": (
                safety_assessment.procedure_detected if safety_assessment is not None else False
            ),
            "broad_learning_request": broad_learning_request,
            "numeric_calculation_intent": has_numeric_intent,
        },
    }
    return classification, debug


__all__ = [
    "MAIN_TYPE_LABELS",
    "derive_question_classification",
    "explicit_user_document_request",
    "has_numeric_calculation_intent",
    "parse_main_type_label",
    "rule_based_main_type",
]
