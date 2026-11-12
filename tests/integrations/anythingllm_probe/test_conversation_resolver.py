from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chemistry_bot.integrations.anythingllm_probe.conversation_resolver import (
    AnythingLLMConversationResolver,
    ConversationResolutionError,
    parse_openai_conversation,
)
from chemistry_bot.tutor_graph.storage import SQLiteTutorStore


class ConversationResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteTutorStore(Path(self.temp_dir.name) / "tutor.db")
        ids = iter(["session-a", "session-b", "session-c", "session-d"])
        self.resolver = AnythingLLMConversationResolver(
            self.store,
            student_id="student-1",
            session_id_factory=lambda: next(ids),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _messages(question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": question},
        ]

    def _store_exchange(self, session_id: str, user_text: str, answer: str) -> None:
        self.store.ensure_session(session_id, "student-1", title=user_text)
        self.store.save_turn(
            student_id="student-1",
            session_id=session_id,
            role="student",
            text=user_text,
        )
        self.store.save_turn(
            student_id="student-1",
            session_id=session_id,
            role="tutor",
            text=answer,
        )

    def test_first_message_creates_new_session(self) -> None:
        resolution, parsed = self.resolver.resolve(self._messages("what is an atom"))
        self.assertEqual(resolution.session_id, "session-a")
        self.assertTrue(resolution.created_new_session)
        self.assertEqual(resolution.resolution_method, "new_conversation_no_history")
        self.assertEqual(parsed.current_user_text, "what is an atom")

    def test_exact_history_suffix_reuses_session(self) -> None:
        self._store_exchange("existing", "what is an atom", "An atom is...")
        resolution, _ = self.resolver.resolve(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "what is an atom"},
                {"role": "assistant", "content": "An atom is..."},
                {"role": "user", "content": "and a molecule?"},
            ]
        )
        self.assertEqual(resolution.session_id, "existing")
        self.assertFalse(resolution.created_new_session)
        self.assertEqual(resolution.matched_turn_count, 2)
        self.assertEqual(resolution.resolution_method, "exact_history_suffix")

    def test_unmatched_supplied_history_creates_new_session(self) -> None:
        resolution, _ = self.resolver.resolve(
            [
                {"role": "user", "content": "unknown old question"},
                {"role": "assistant", "content": "unknown old answer"},
                {"role": "user", "content": "continue"},
            ]
        )
        self.assertEqual(resolution.session_id, "session-a")
        self.assertEqual(
            resolution.resolution_method,
            "new_conversation_no_history_match",
        )
        self.assertTrue(resolution.warnings)

    def test_unmatched_history_is_imported_so_the_next_request_can_match(self) -> None:
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ]
        resolution, parsed = self.resolver.resolve(messages)
        persisted = self.resolver.append_exchange(
            resolution=resolution,
            parsed=parsed,
            response_text="current answer",
            request_id="request-1",
            request_fingerprint="fingerprint-1",
        )
        self.assertEqual(len(persisted["imported_history"]), 2)

        next_resolution, _ = self.resolver.resolve(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "current question"},
                {"role": "assistant", "content": "current answer"},
                {"role": "user", "content": "next question"},
            ]
        )
        self.assertEqual(next_resolution.session_id, resolution.session_id)
        self.assertEqual(
            next_resolution.resolution_method,
            "exact_history_suffix",
        )
        self.assertEqual(next_resolution.matched_turn_count, 4)

    def test_same_first_question_still_creates_a_new_chat_without_history(self) -> None:
        self._store_exchange("old", "same question", "old answer")
        resolution, _ = self.resolver.resolve(self._messages("same question"))
        self.assertNotEqual(resolution.session_id, "old")
        self.assertEqual(resolution.resolution_method, "new_conversation_no_history")

    def test_system_messages_do_not_affect_identity(self) -> None:
        parsed = parse_openai_conversation(
            [
                {"role": "system", "content": "workspace prompt"},
                {"role": "user", "content": "question"},
            ]
        )
        self.assertEqual(parsed.current_user_text, "question")
        self.assertEqual(parsed.history, ())
        self.assertEqual(parsed.ignored_roles, ("system",))

    def test_request_must_end_in_user_message(self) -> None:
        with self.assertRaises(ConversationResolutionError):
            parse_openai_conversation(
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            )

    def test_multipart_text_content_is_supported(self) -> None:
        parsed = parse_openai_conversation(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "input_text", "text": "part two"},
                    ],
                }
            ]
        )
        self.assertEqual(parsed.current_user_text, "part one\npart two")


if __name__ == "__main__":
    unittest.main()
