from __future__ import annotations

import unittest

from chemistry_bot.integrations.anythingllm_probe.context_adapter import (
    adapt_anythingllm_tool_results,
)


class _Result:
    def __init__(self, content: str):
        self.content = content


class ContextAdapterTests(unittest.TestCase):
    def test_parses_repeated_inline_metadata_and_ranks_relevant_chunk(self) -> None:
        content = """Additional context for query:
<document_metadata>
sourceDocument: analytical.pdf
published: 2026-07-14
</document_metadata>
A zwitterion contains both positive and negative charges.
<document_metadata>
sourceDocument: analytical.pdf
published: 2026-07-14
</document_metadata>
Sodium carbonate is titrated with hydrochloric acid. The carbonate reacts with two equivalents of hydrogen ions.
"""
        adapted = adapt_anythingllm_tool_results(
            query="titration of sodium carbonate",
            tool_results=[_Result(content)],
        )
        self.assertEqual(adapted.source_documents, ("analytical.pdf",))
        self.assertEqual(adapted.relevance, "high")
        self.assertGreaterEqual(adapted.relevance_score, 0.60)
        self.assertIn("Sodium carbonate", adapted.chunks[0].text)
        self.assertEqual(adapted.chunks[0].rank, 1)

    def test_marks_irrelevant_retrieval_low(self) -> None:
        content = """<document_metadata>
sourceDocument: analytical.pdf
</document_metadata>
Zwitterions and amino-acid ionization are discussed in this section.
"""
        adapted = adapt_anythingllm_tool_results(
            query="why use zinc uranyl acetate",
            tool_results=[_Result(content)],
        )
        self.assertEqual(adapted.relevance, "low")
        self.assertTrue(any("low lexical relevance" in item for item in adapted.warnings))

    def test_prompt_marks_document_as_untrusted(self) -> None:
        adapted = adapt_anythingllm_tool_results(
            query="weak acid strong base",
            tool_results=[
                _Result(
                    "<document_metadata>\nsourceDocument: book.pdf\n</document_metadata>\n"
                    "Weak-acid titration with a strong base has a buffer region."
                )
            ],
        )
        prompt = adapted.to_system_message_text()
        self.assertIn("untrusted reference material", prompt)
        self.assertIn("book.pdf", prompt)
        self.assertIn("Weak-acid titration", prompt)


if __name__ == "__main__":
    unittest.main()
