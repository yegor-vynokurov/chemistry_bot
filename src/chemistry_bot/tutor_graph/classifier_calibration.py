"""Compatibility wrapper for the I0.3.1A thin classifier policy.

Older code imports ``calibrate_question_classification``. The function now
ignores model-generated flags/scores and re-derives the operational
classification from ``main_type`` plus deterministic signals.
"""

from __future__ import annotations

from typing import Any

from .schemas import QuestionClassification, ThinQuestionClassification
from .thin_classifier import derive_question_classification


def calibrate_question_classification(
    classification: QuestionClassification | ThinQuestionClassification,
    *,
    user_text: str,
    session_state: dict[str, Any] | None = None,
) -> tuple[QuestionClassification, dict[str, Any]]:
    thin = ThinQuestionClassification(main_type=classification.main_type)
    derived, debug = derive_question_classification(
        thin,
        user_text=user_text,
        session_state=session_state,
        source="compatibility_calibration",
    )
    debug["before"] = (
        classification.model_dump()
        if hasattr(classification, "model_dump")
        else {"main_type": classification.main_type}
    )
    debug["after"] = derived.model_dump()
    return derived, debug


__all__ = ["calibrate_question_classification"]
