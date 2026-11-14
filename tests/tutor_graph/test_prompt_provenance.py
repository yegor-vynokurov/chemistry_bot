from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from chemistry_bot.integrations.anythingllm_probe.routing_preview import (
    AnythingLLMRoutingPreview,
    RoutingPreview,
)
from chemistry_bot.tutor_graph.graph import TutorGraphConfig, build_tutor_graph
from chemistry_bot.tutor_graph.prompt_catalog import PromptCatalog
from chemistry_bot.tutor_graph.prompt_provenance import PromptInvocationCollector
from chemistry_bot.tutor_graph.schemas import QuestionClassification
from chemistry_bot.tutor_graph.storage import SQLiteTutorStore


class _FakeResponse:
    def __init__(self, content: str, response_metadata: dict[str, object] | None = None) -> None:
        self.content = content
        self.response_metadata = dict(response_metadata or {})


class _FakeModel:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> _FakeResponse:
        self.messages = messages
        if self.failure is not None:
            raise self.failure
        return _FakeResponse("private model output")


class _QueuedModel:
    def __init__(self, responses: list[str | _FakeResponse]) -> None:
        self.responses = list(responses)
        self.seen: list[list[object]] = []

    def invoke(self, messages: list[object]) -> _FakeResponse:
        self.seen.append(messages)
        response = self.responses.pop(0)
        return response if isinstance(response, _FakeResponse) else _FakeResponse(response)


class PromptProvenanceTests(unittest.TestCase):
    def test_runtime_prompt_assets_resolve_by_id_version_and_digest(self) -> None:
        """Verify all packaged prompt releases can be resolved exactly."""
        catalog = PromptCatalog()

        self.assertEqual(catalog.assignment, {"kind": "baseline", "experiment_id": None})
        self.assertGreaterEqual(len(catalog._prompts), 19)
        for revision in catalog._prompts.values():
            resolved = catalog.resolve_reference(revision.as_reference())
            self.assertEqual(resolved, revision)

    def test_collector_records_success_and_failure_without_message_content(self) -> None:
        """Record actual attempts while excluding input and output text."""
        catalog = PromptCatalog()
        prompt = catalog.components("routing.classifier.main_type")[0]
        collector = PromptInvocationCollector()
        succeeded = _FakeModel()
        failed = _FakeModel(failure=RuntimeError("sensitive exception content"))

        collector.invoke(
            succeeded,
            ["private learner text"],
            role="routing.classifier.main_type",
            model_name="phi4-mini:latest",
            prompts=(prompt,),
            assignment=catalog.assignment,
        )
        with self.assertRaises(RuntimeError):
            collector.invoke(
                failed,
                ["another private learner text"],
                role="routing.classifier.main_type",
                model_name="phi4-mini:latest",
                prompts=(prompt,),
                assignment=catalog.assignment,
            )

        records = collector.snapshot()
        self.assertEqual([item["status"] for item in records], ["succeeded", "failed"])
        self.assertEqual(records[1]["error_code"], "RuntimeError")
        serialized = repr(records)
        self.assertNotIn("private learner text", serialized)
        self.assertNotIn("private model output", serialized)
        self.assertNotIn("sensitive exception content", serialized)

    def test_exchange_persists_actual_calls_and_explicit_no_model_state(self) -> None:
        """Link prompt references and deterministic no-model state to the tutor turn."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteTutorStore(Path(temp_dir) / "tutor.db")
            prompt = PromptCatalog().components("tutor.system")[0]
            exchange = store.save_exchange(
                student_id="student",
                session_id="session",
                student_text="question",
                tutor_text="answer",
                prompt_invocations=[
                    {
                        "role": "tutor.answer",
                        "provider": "ollama",
                        "model": "gemma4:e2b-it-q4_K_M",
                        "status": "succeeded",
                        "attempt": 1,
                        "prompts": [prompt.as_reference()],
                        "assignment_kind": "baseline",
                        "experiment_id": None,
                    }
                ],
            )
            readback = store.get_prompt_invocations(exchange["tutor"])
            self.assertEqual(readback["state"], "recorded")
            self.assertEqual(readback["invocations"][0]["model"], "gemma4:e2b-it-q4_K_M")
            self.assertEqual(readback["invocations"][0]["prompts"][0]["prompt_id"], prompt.prompt_id)
            self.assertEqual(readback["invocations"][0]["prompts"][0]["version"], prompt.version)

            no_model_exchange = store.save_exchange(
                student_id="student",
                session_id="session",
                student_text="deterministic question",
                tutor_text="deterministic answer",
                no_model_reason="deterministic_route",
            )
            no_model_readback = store.get_prompt_invocations(no_model_exchange["tutor"])
            self.assertEqual(no_model_readback["state"], "no_model_invocation")
            self.assertEqual(no_model_readback["invocations"][0]["reason"], "deterministic_route")

    def test_turn_and_invocations_roll_back_together(self) -> None:
        """Keep the exchange absent if any provenance insert fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "tutor.db"
            store = SQLiteTutorStore(database_path)
            prompt = PromptCatalog().components("tutor.system")[0]
            with store.connect() as connection:
                connection.execute(
                    """CREATE TRIGGER reject_prompt_invocation
                       BEFORE INSERT ON prompt_invocations
                       BEGIN SELECT RAISE(ABORT, 'blocked by test fixture'); END"""
                )
                connection.commit()

            with self.assertRaises(sqlite3.IntegrityError):
                store.save_exchange(
                    student_id="student",
                    session_id="session",
                    student_text="question",
                    tutor_text="answer",
                    prompt_invocations=[
                        {
                            "role": "tutor.answer",
                            "provider": "ollama",
                            "model": "gemma4:e2b-it-q4_K_M",
                            "status": "succeeded",
                            "attempt": 1,
                            "prompts": [prompt.as_reference()],
                            "assignment_kind": "baseline",
                            "experiment_id": None,
                        }
                    ],
                )
            self.assertEqual(store.count_session_turns("session"), 0)

    def test_legacy_turn_without_provenance_remains_unknown(self) -> None:
        """Distinguish old turn rows from newly recorded no-model turns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteTutorStore(Path(temp_dir) / "tutor.db")
            legacy_id = store.save_turn(
                student_id="student",
                session_id="session",
                role="tutor",
                text="historical answer",
            )
            self.assertEqual(store.get_prompt_invocations(legacy_id)["state"], "unknown")

    def test_provenance_readback_rejects_missing_or_non_tutor_turn_ids(self) -> None:
        """Avoid reporting unknown provenance for an invalid turn reference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteTutorStore(Path(temp_dir) / "tutor.db")
            student_id = store.save_turn(
                student_id="student",
                session_id="session",
                role="student",
                text="current question",
            )

            with self.assertRaises(KeyError):
                store.get_prompt_invocations(999_999)
            with self.assertRaises(ValueError):
                store.get_prompt_invocations(student_id)

    def test_routing_preview_round_trip_keeps_support_calls_without_replay(self) -> None:
        """Preserve main and relation classifier records through the tool interaction log shape."""
        fake_model = _QueuedModel(["reference_closed", "new_question"])
        with patch(
            "chemistry_bot.integrations.anythingllm_probe.routing_preview.ChatOllama",
            return_value=fake_model,
        ):
            service = AnythingLLMRoutingPreview(semantic_safety_enabled=False)
            preview = service.classify(
                user_text="What is an atom?",
                session_state={
                    "awaiting_student_answer": True,
                    "last_bot_question": "What is an atom?",
                    "last_tutor_move": "diagnostic_question",
                },
                rag_tool_available=False,
            )

        restored = RoutingPreview.from_dict(preview.as_dict())
        self.assertEqual(
            [item["role"] for item in restored.prompt_invocations],
            ["routing.classifier.main_type", "routing.classifier.turn_relation"],
        )
        self.assertEqual(len(fake_model.seen), 2)
        self.assertEqual(
            [item["prompt_id"] for item in restored.prompt_invocations[0]["prompts"]],
            ["sys_000016", "sys_000025"],
        )

    def test_graph_records_tutor_components_and_deterministic_no_call(self) -> None:
        """Capture tutor system/route components and leave deterministic routes call-free."""
        classification = QuestionClassification(main_type="reference_closed")
        collector = PromptInvocationCollector()
        fake_model = _QueuedModel(["Oxygen has atomic number eight."])
        with patch("chemistry_bot.tutor_graph.graph.ChatOllama", return_value=fake_model):
            graph = build_tutor_graph(
                TutorGraphConfig(
                    classifier_enabled=False,
                    turn_relation_enabled=False,
                    followup_enabled=False,
                    semantic_safety_enabled=False,
                    model_name="gemma4:e2b-it-q4_K_M",
                )
            )
            result = graph.invoke(
                {
                    "student_id": "student",
                    "session_id": "session",
                    "user_text": "What is the atomic number of oxygen?",
                    "classification": classification,
                    "session_state": {},
                    "messages": [],
                    "prompt_invocation_collector": collector,
                }
            )
        invocation = collector.snapshot()[0]
        self.assertEqual(invocation["role"], "tutor.answer")
        self.assertEqual(
            [item["prompt_id"] for item in invocation["prompts"]],
            ["sys_000015", "sys_000020", "sys_000027", "sys_000029"],
        )
        self.assertEqual(result["response"].visible_answer, "Oxygen has atomic number eight.")

        deterministic_collector = PromptInvocationCollector()
        deterministic = classification.model_copy(update={"main_type": "off_topic"})
        deterministic_result = graph.invoke(
            {
                "student_id": "student",
                "session_id": "session",
                "user_text": "Tell me a joke.",
                "classification": deterministic,
                "session_state": {},
                "messages": [],
                "prompt_invocation_collector": deterministic_collector,
            }
        )
        self.assertEqual(deterministic_collector.snapshot(), ())
        self.assertEqual(deterministic_result["debug"]["model_invoked"], False)

    def test_graph_records_primary_retry_and_followup_as_separate_calls(self) -> None:
        """Keep each tutor retry and follow-up invocation under its own role record."""
        classification = QuestionClassification(main_type="theory_learning_open")
        collector = PromptInvocationCollector()
        tutor_model = _QueuedModel(
            [
                _FakeResponse(
                    "Atoms share electrons to form",
                    {"finish_reason": "length"},
                ),
                _FakeResponse("Atoms share electrons to form a chemical bond."),
            ]
        )
        followup_model = _QueuedModel([_FakeResponse("What do you think happens next?")])
        with patch(
            "chemistry_bot.tutor_graph.graph.ChatOllama",
            side_effect=[tutor_model, followup_model],
        ):
            graph = build_tutor_graph(
                TutorGraphConfig(
                    classifier_enabled=False,
                    turn_relation_enabled=False,
                    followup_enabled=True,
                    semantic_safety_enabled=False,
                    model_name="gemma4:e2b-it-q4_K_M",
                )
            )
            result = graph.invoke(
                {
                    "student_id": "student",
                    "session_id": "session",
                    "user_text": "How do atoms form a chemical bond?",
                    "classification": classification,
                    "session_state": {},
                    "messages": [],
                    "prompt_invocation_collector": collector,
                }
            )

        records = collector.snapshot()
        self.assertEqual(
            [(item["role"], item["attempt"]) for item in records],
            [("tutor.answer", 1), ("tutor.answer.retry", 2), ("tutor.followup", 1)],
        )
        self.assertEqual(
            [item["prompt_id"] for item in records[1]["prompts"]],
            ["sys_000019", "sys_000021", "sys_000027"],
        )
        self.assertEqual(records[2]["prompts"][0]["prompt_id"], "sys_000018")
        self.assertIn("Atoms share electrons", result["response"].visible_answer)


if __name__ == "__main__":
    unittest.main()
