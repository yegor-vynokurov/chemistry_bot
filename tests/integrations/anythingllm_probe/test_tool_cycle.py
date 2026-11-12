from __future__ import annotations

import json
import unittest

from chemistry_bot.integrations.anythingllm_probe.tool_cycle import (
    inspect_tool_cycle,
    interaction_id_from_tool_call_id,
    make_interaction_id,
    make_tool_call_id,
)


class ToolCycleTests(unittest.TestCase):
    def test_interaction_id_round_trip(self) -> None:
        interaction_id = make_interaction_id()
        tool_call_id = make_tool_call_id(interaction_id)
        self.assertEqual(
            interaction_id_from_tool_call_id(tool_call_id), interaction_id
        )

    def test_standard_tool_result_is_detected(self) -> None:
        interaction_id = make_interaction_id()
        call_id = make_tool_call_id(interaction_id)
        body = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "rag-memory", "parameters": {}},
                }
            ],
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "rag-memory",
                                "arguments": json.dumps(
                                    {"action": "search", "content": "question"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "retrieved chunk",
                },
            ],
        }
        inspection = inspect_tool_cycle(body)
        self.assertEqual(inspection.phase, "tool_result_return")
        self.assertEqual(inspection.interaction_id, interaction_id)
        self.assertEqual(len(inspection.assistant_tool_calls), 1)
        self.assertEqual(len(inspection.tool_results), 1)

    def test_nonstandard_message_after_call_is_retained_as_possible_result(self) -> None:
        interaction_id = make_interaction_id()
        call_id = make_tool_call_id(interaction_id)
        inspection = inspect_tool_cycle(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "rag-memory",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {"role": "system", "content": "RAG result: chunk"},
                ]
            }
        )
        self.assertEqual(inspection.phase, "tool_result_return")
        self.assertEqual(
            inspection.tool_results[0].detection_method,
            "message_after_assistant_tool_call",
        )


if __name__ == "__main__":
    unittest.main()
