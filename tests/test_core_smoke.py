"""Smoke tests for retrieval connectivity and the main QA workflow."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.cli.rag_bot import CliBot  # noqa: E402
from chemistry_bot.retrieval import (  # noqa: E402
    ChromaTheoryRAG,
    INTROCHEM_CHROMA_DIR,
    INTROCHEM_COLLECTION_NAME,
    INTROCHEM_RETRIEVAL_FIXTURES,
    PROMPT_GARDEN_ROOT,
    RAGConfig,
)


FIXTURES_PATH = INTROCHEM_RETRIEVAL_FIXTURES
DB_PATH = INTROCHEM_CHROMA_DIR
COLLECTION_NAME = INTROCHEM_COLLECTION_NAME
BOT_QUESTION = "What is a chemical equation and why must it be balanced?"


def load_retrieval_fixtures(limit: int = 3) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    with FIXTURES_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                continue

            fixtures.append(record)
            if len(fixtures) >= limit:
                break

    if not fixtures:
        raise RuntimeError(
            f"No retrieval fixtures were loaded from {FIXTURES_PATH}."
        )

    return fixtures


def build_smoke_rag_config() -> RAGConfig:
    return RAGConfig(
        enabled=True,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=None,
        ollama_base_url=None,
        chunks_path=None,
        top_k=3,
        candidate_k=6,
        max_per_section=2,
        max_context_chars=3200,
        filter_default_retrieval=True,
        fail_open=False,
    )


class RetrievalSmokeTest(unittest.TestCase):
    """Confirm that the textbook index opens and returns hits."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rag = ChromaTheoryRAG(build_smoke_rag_config())
        cls.fixtures = load_retrieval_fixtures(limit=3)

    def test_fixture_queries_return_hits(self) -> None:
        for fixture in self.fixtures:
            query_id = fixture.get("id", "unknown")
            query = fixture["query"]

            with self.subTest(query_id=query_id):
                hits = self.rag.retrieve(query)
                self.assertGreater(len(hits), 0)
                self.assertLessEqual(len(hits), self.rag.config.top_k)
                self.assertTrue(all(hit.chunk_id for hit in hits))
                self.assertEqual(
                    len({hit.chunk_id for hit in hits}),
                    len(hits),
                )

                formatted_context = self.rag.format_context(hits)
                self.assertIn("[S1]", formatted_context)


class BotSmokeTest(unittest.TestCase):
    """Confirm that the RAG-enabled bot returns a structured answer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bot = CliBot(
            model_name="phi4-mini",
            garden_root=PROMPT_GARDEN_ROOT,
            combo_id="combo_000014",
            fewshot_id="fsh_000002",
            max_history_messages=6,
            materialize_context=False,
            rag_config=build_smoke_rag_config(),
        )
        cls.bot.garden.add_run = lambda **_: None

    def test_school_question_returns_structured_answer(self) -> None:
        answer = self.bot.invoke_once(
            user_text=BOT_QUESTION,
            session_id="core_smoke_test",
            silent=True,
        )

        self.assertIsNotNone(answer)
        assert answer is not None

        self.assertIn(answer.request_type, {"theory", "mixed"})
        self.assertTrue(answer.short_answer.strip())
        self.assertTrue(answer.explanation.strip())
        self.assertGreater(len(self.bot.last_rag_hits), 0)

        allowed_source_ids = {
            hit.label for hit in self.bot.last_rag_hits
        }
        self.assertTrue(
            set(answer.source_ids).issubset(allowed_source_ids)
        )
