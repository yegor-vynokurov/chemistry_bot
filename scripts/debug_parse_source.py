from __future__ import annotations

import argparse
from pathlib import Path

from wiki_ingest.parsers import parse_source


def main() -> int:
    p = argparse.ArgumentParser(description="Parse one source and print title/text preview without calling LLM.")
    p.add_argument("--input", required=True)
    p.add_argument("--chars", type=int, default=800)
    args = p.parse_args()

    parsed = parse_source(Path(args.input))
    print(f"parser_kind: {parsed.parser_kind}")
    print(f"title_guess: {parsed.title_guess}")
    print(f"text_chars: {parsed.char_count}")
    print("")
    print(parsed.text[: args.chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
