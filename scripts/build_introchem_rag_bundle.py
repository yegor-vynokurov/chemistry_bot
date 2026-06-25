"""One-command wrapper for the current Introductory Chemistry RAG build flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.retrieval import (  # noqa: E402
    INTROCHEM_BUILD_CHUNKS_SCRIPT,
    INTROCHEM_CHROMA_DIR,
    INTROCHEM_COLLECTION_NAME,
    INTROCHEM_EMBEDDING_MODEL,
    INTROCHEM_NORMALIZED_ROOT,
    INTROCHEM_PARSE_SCRIPT,
    INTROCHEM_PRIMARY_HTML,
    INTROCHEM_RAG_ROOT,
    INTROCHEM_VECTOR_SEARCH_SCRIPT,
    introchem_chapter_output_dir,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_INPUT = INTROCHEM_PRIMARY_HTML
DEFAULT_NORMALIZED_ROOT = INTROCHEM_NORMALIZED_ROOT
DEFAULT_RAG_OUTPUT = INTROCHEM_RAG_ROOT
DEFAULT_DB = INTROCHEM_CHROMA_DIR
DEFAULT_COLLECTION = INTROCHEM_COLLECTION_NAME
DEFAULT_EMBEDDING_MODEL = INTROCHEM_EMBEDDING_MODEL
DEFAULT_BASE_URL = "http://localhost:11434"


def parse_chapter_spec(value: str) -> list[int]:
    chapters: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid chapter range: {part}")
            chapters.update(range(start, end + 1))
        else:
            chapters.add(int(part))

    result = sorted(chapters)
    if not result:
        raise ValueError("No chapters were selected.")
    return result


def run_command(command: list[str], dry_run: bool) -> None:
    rendered = subprocess.list2cmdline(command)
    print(f"> {rendered}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current textbook-to-RAG bundle for Introductory Chemistry "
            "through one public wrapper command."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the textbook XHTML/HTML export.",
    )
    parser.add_argument(
        "--chapters",
        default="1-18",
        help="Chapter selection, for example '1-18' or '1,4,7-9'.",
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=DEFAULT_NORMALIZED_ROOT,
        help="Root directory for normalized chapter outputs.",
    )
    parser.add_argument(
        "--rag-output",
        type=Path,
        default=DEFAULT_RAG_OUTPUT,
        help="Output directory for rag_chunks.jsonl and chunk reports.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model used for index build.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Ollama base URL for embeddings.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=1800,
        help="Chunk target size in characters.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Chunk maximum size in characters.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=180,
        help="Chunk overlap size in characters.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Chunk minimum size in characters.",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip textbook parsing and reuse existing normalized outputs.",
    )
    parser.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Skip chunk generation and reuse existing rag_chunks.jsonl.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector index build.",
    )
    parser.add_argument(
        "--only-default",
        action="store_true",
        help="Index only default retrieval chunks.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete the existing Chroma collection before rebuilding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    normalized_root = args.normalized_root.resolve()
    rag_output = args.rag_output.resolve()
    db_path = args.db.resolve()

    chapters = parse_chapter_spec(args.chapters)

    if not args.skip_parse and not input_path.exists():
        raise FileNotFoundError(f"Textbook input file not found: {input_path}")

    python_exe = sys.executable

    if not args.skip_parse:
        for chapter in chapters:
            if normalized_root == INTROCHEM_NORMALIZED_ROOT.resolve():
                chapter_dir = introchem_chapter_output_dir(chapter).resolve()
            else:
                chapter_dir = normalized_root / f"chapter_{chapter:02d}_v3"
            command = [
                python_exe,
                str(INTROCHEM_PARSE_SCRIPT),
                "--input",
                str(input_path),
                "--chapter",
                str(chapter),
                "--output",
                str(chapter_dir),
            ]
            run_command(command, dry_run=args.dry_run)

    if not args.skip_chunks:
        command = [
            python_exe,
            str(INTROCHEM_BUILD_CHUNKS_SCRIPT),
            "--normalized-root",
            str(normalized_root),
            "--output",
            str(rag_output),
            "--target-chars",
            str(args.target_chars),
            "--max-chars",
            str(args.max_chars),
            "--overlap-chars",
            str(args.overlap_chars),
            "--min-chars",
            str(args.min_chars),
        ]
        run_command(command, dry_run=args.dry_run)

    if not args.skip_index:
        command = [
            python_exe,
            str(INTROCHEM_VECTOR_SEARCH_SCRIPT),
            "build",
            "--chunks",
            str(rag_output / "rag_chunks.jsonl"),
            "--db",
            str(db_path),
            "--collection",
            args.collection,
            "--model",
            args.embedding_model,
            "--base-url",
            args.base_url,
        ]
        if args.only_default:
            command.append("--only-default")
        if args.rebuild_index:
            command.append("--rebuild")
        run_command(command, dry_run=args.dry_run)

    print()
    print("Bundle workflow summary")
    print(f"- input: {input_path}")
    print(f"- chapters: {chapters}")
    print(f"- normalized root: {normalized_root}")
    print(f"- rag output: {rag_output}")
    print(f"- db: {db_path}")
    print(f"- collection: {args.collection}")
    print(f"- dry run: {args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
