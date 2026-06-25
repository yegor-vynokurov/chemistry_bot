"""Run the repository's Prompt Garden tooling smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_prompt_garden_smoke"
    )
    result = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
    ).run(suite)

    if result.wasSuccessful():
        print("\nOK: Prompt Garden smoke tests passed.")
        return 0

    print("\nFAIL: Prompt Garden smoke tests failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

