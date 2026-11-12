from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chemistry_bot.tutor_graph.storage import SQLiteTutorStore


class SQLiteDecisionLogTests(unittest.TestCase):
    def test_save_and_read_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteTutorStore(Path(tmp) / "tutor.db")
            turn_id = store.save_turn(
                student_id="student_1",
                session_id="session_1",
                role="student",
                text="What is an atom?",
            )
            first_id = store.save_decision(
                student_id="student_1",
                session_id="session_1",
                turn_id=turn_id,
                node_name="classify_question",
                decision_type="classification",
                decision={"main_type": "theory_learning_open"},
                confidence=0.8,
                reason="Conceptual question.",
            )
            second_id = store.save_decision(
                student_id="student_1",
                session_id="session_1",
                turn_id=turn_id,
                node_name="select_route",
                decision_type="routing",
                decision={"route": "theory"},
                confidence=0.8,
                reason="Use theory route.",
            )

            rows = store.get_recent_decisions(session_id="session_1", limit=10)
            self.assertEqual([row.id for row in rows], [first_id, second_id])
            self.assertEqual(rows[0].turn_id, turn_id)
            self.assertEqual(rows[1].decision, {"route": "theory"})
            self.assertEqual(store.status().decision_log_entries, 2)

            filtered = store.get_recent_decisions(
                session_id="session_1",
                limit=10,
                decision_types=["routing"],
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].decision_type, "routing")


if __name__ == "__main__":
    unittest.main()
