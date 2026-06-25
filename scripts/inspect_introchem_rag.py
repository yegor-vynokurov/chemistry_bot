"""Quick retrieval inspection tool for the Introductory Chemistry index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.retrieval import (  # noqa: E402
    ChromaTheoryRAG,
    INTROCHEM_CHROMA_DIR,
    INTROCHEM_COLLECTION_NAME,
    RAGConfig,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview retrieval hits from the local Introductory Chemistry RAG index."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to retrieve against the Introductory Chemistry index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of final retrieval hits to keep.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Number of retrieval candidates before filtering.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3500,
        help="Context budget for formatted retrieval blocks.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the formatted context block in addition to source previews.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rag = ChromaTheoryRAG(
        RAGConfig(
            enabled=True,
            db_path=INTROCHEM_CHROMA_DIR,
            collection_name=INTROCHEM_COLLECTION_NAME,
            embedding_model=None,
            ollama_base_url=None,
            chunks_path=None,
            top_k=args.top_k,
            candidate_k=max(args.top_k, args.candidate_k),
            max_per_section=2,
            max_context_chars=args.max_context_chars,
            filter_default_retrieval=True,
            fail_open=True,
        )
    )

    context, hits = rag.retrieve_context(args.query)
    print(rag.format_sources(hits))
    if args.show_context:
        print()
        print(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
