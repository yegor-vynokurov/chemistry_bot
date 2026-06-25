"""Public launcher for the current RAG-enabled chemistry bot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.cli.rag_bot import CliBot, StudentContext, TeacherContext  # noqa: E402
from chemistry_bot.retrieval import (  # noqa: E402
    INTROCHEM_CHROMA_DIR,
    INTROCHEM_COLLECTION_NAME,
    PROMPT_GARDEN_ROOT,
    RAGConfig,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_teacher_context() -> TeacherContext:
    return TeacherContext(
        teacher_profile=(
            "an experienced chemistry teacher "
            "with university-level knowledge"
        ),
        personality_traits=(
            "patient, curious, calm, observant, and honest"
        ),
        tone=(
            "warm and encouraging, but never childish or overly enthusiastic"
        ),
        teaching_style=(
            "build explanations step by step; "
            "use one concrete example before introducing terminology"
        ),
        correction_style=(
            "correct misconceptions directly but gently"
        ),
        language_style=(
            "plain English for a ninth-grade student; "
            "no slang and no unnecessary jargon"
        ),
    )


def make_bot(args: argparse.Namespace) -> CliBot:
    top_k = int(args.rag_k)
    candidate_k = max(int(args.candidate_k), top_k)

    return CliBot(
        model_name=args.model,
        garden_root=PROMPT_GARDEN_ROOT,
        combo_id=args.combo_id,
        fewshot_id=None if args.no_fewshot else args.fewshot_id,
        max_history_messages=args.max_history_messages,
        materialize_context=True,
        rag_config=RAGConfig(
            enabled=not args.no_rag,
            db_path=INTROCHEM_CHROMA_DIR,
            collection_name=INTROCHEM_COLLECTION_NAME,
            embedding_model=None,
            ollama_base_url=None,
            chunks_path=None,
            top_k=top_k,
            candidate_k=candidate_k,
            max_per_section=2,
            max_context_chars=args.max_context_chars,
            filter_default_retrieval=True,
            fail_open=True,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current RAG-enabled Chemistry Bot for school-level "
            "question answering."
        )
    )
    parser.add_argument(
        "--model",
        default="phi4-mini",
        help="Local Ollama chat model to use.",
    )
    parser.add_argument(
        "--session-id",
        default="local_user",
        help="Conversation session identifier.",
    )
    parser.add_argument(
        "--combo-id",
        default="combo_000014",
        help="Base Prompt Garden combo to use.",
    )
    parser.add_argument(
        "--fewshot-id",
        default="fsh_000002",
        help="Few-shot Prompt Garden node id.",
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Disable few-shot examples.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG for this run.",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=4,
        help="Number of retrieved sources to keep.",
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
        default=6500,
        help="Maximum character budget for formatted RAG context.",
    )
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=12,
        help="Maximum number of chat-history messages to keep.",
    )
    parser.add_argument(
        "--single-question",
        help=(
            "Ask one question, print the structured result, and exit "
            "instead of starting the interactive loop."
        ),
    )
    parser.add_argument(
        "--protocol-context",
        default="No verified experimental protocol was retrieved.",
        help="Initial protocol context for the student profile.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source previews after a single-question run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bot = make_bot(args)

    student = StudentContext(protocol_context=args.protocol_context)
    teacher = build_teacher_context()

    if args.single_question:
        answer = bot.invoke_once(
            user_text=args.single_question,
            session_id=args.session_id,
            silent=True,
        )
        if answer is None:
            print("Bot invocation failed.")
            return 1

        print(answer.model_dump_json(indent=2))
        if args.show_sources:
            print()
            print(bot.rag.format_sources(bot.last_rag_hits))
        return 0

    bot(session_id=args.session_id, context=student, teacher_context=teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
