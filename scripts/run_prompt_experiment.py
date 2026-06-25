"""Run a Prompt Garden experiment outside the notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.promptops.runner import (  # noqa: E402
    ExperimentRunConfig,
    run_prompt_experiment,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Prompt Garden experiment with tracked raw and "
            "normalized run artifacts."
        )
    )
    parser.add_argument(
        "--garden-root",
        default=str(REPO_ROOT / "prompt_garden"),
        help="Prompt Garden workspace root.",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Prompt Garden experiment id to execute.",
    )
    parser.add_argument(
        "--model",
        default="phi4-mini",
        help="Local Ollama chat model to use.",
    )
    parser.add_argument(
        "--bot-variant",
        choices=("rag", "legacy"),
        default="rag",
        help="Bot implementation to evaluate.",
    )
    parser.add_argument(
        "--fewshot-id",
        default="fsh_000002",
        help="Few-shot Prompt Garden node id.",
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Disable few-shot examples for this run.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG when bot-variant=rag.",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=4,
        help="Number of retrieved sources to keep in rag mode.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Number of retrieval candidates before filtering in rag mode.",
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
        "--case-set",
        default="default_chemistry_school_cases_v1",
        help=(
            "Case-set id inside prompt_garden/cases/ or a direct path to a "
            "case-set JSON file."
        ),
    )
    parser.add_argument(
        "--only-case-id",
        action="append",
        default=[],
        help="Run only the specified case id. Repeat as needed.",
    )
    parser.add_argument(
        "--skip-case-id",
        action="append",
        default=[],
        help="Skip the specified case id. Repeat as needed.",
    )
    parser.add_argument(
        "--only-combo",
        action="append",
        default=[],
        help="Run only the specified attached combo id. Repeat as needed.",
    )
    parser.add_argument(
        "--skip-combo",
        action="append",
        default=[],
        help="Skip the specified attached combo id. Repeat as needed.",
    )
    parser.add_argument(
        "--run-mode",
        choices=("missing", "failed", "all"),
        default="missing",
        help=(
            "missing: resume and run only missing artifacts; "
            "failed: rerun only previously failed artifacts; "
            "all: rerun every selected case."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without invoking the model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    summary = run_prompt_experiment(
        ExperimentRunConfig(
            garden_root=Path(args.garden_root).resolve(),
            experiment_id=args.experiment_id,
            model=args.model,
            bot_variant=args.bot_variant,
            fewshot_id=None if args.no_fewshot else args.fewshot_id,
            use_rag=not args.no_rag,
            rag_k=args.rag_k,
            candidate_k=args.candidate_k,
            max_context_chars=args.max_context_chars,
            max_history_messages=args.max_history_messages,
            case_set=args.case_set,
            only_case_ids=tuple(args.only_case_id),
            skip_case_ids=tuple(args.skip_case_id),
            only_combo_ids=tuple(args.only_combo),
            skip_combo_ids=tuple(args.skip_combo),
            run_mode=args.run_mode,
            dry_run=args.dry_run,
        )
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\nOK: Prompt Garden runner dry-run completed.")
        return 0

    print("\nOK: Prompt Garden runner completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
