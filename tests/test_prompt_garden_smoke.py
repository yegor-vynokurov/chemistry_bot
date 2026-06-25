"""Smoke tests for Prompt Garden review and runner tooling."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.promptops.garden import PromptGarden  # noqa: E402
from chemistry_bot.promptops.review_compare import (  # noqa: E402
    compare_record_fields,
    field_segment_alignment,
    unified_text_diff,
)
from chemistry_bot.promptops.review_store import (  # noqa: E402
    build_review_rows,
    load_normalized_scope,
    summary_metrics,
)
from chemistry_bot.promptops.runner import (  # noqa: E402
    ExperimentRunConfig,
    run_prompt_experiment,
)


FIXTURE_GARDEN_ROOT = (
    REPO_ROOT / "tests" / "fixtures" / "prompt_garden_smoke"
)
FIXTURE_SCOPE = "exp_fixture_prompt_garden_smoke"
FIXTURE_CASE_SET = "fixture_school_cases_v1"


def _load_review_app_module():
    module_path = REPO_ROOT / "apps" / "prompt_garden_review.py"
    spec = importlib.util.spec_from_file_location(
        "prompt_garden_review_smoke",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load review app module from {module_path}."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptGardenLoaderSmokeTest(unittest.TestCase):
    """Confirm that tracked normalized artifacts load into review tables."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.garden = PromptGarden(FIXTURE_GARDEN_ROOT)
        cls.artifacts = load_normalized_scope(
            garden_root=FIXTURE_GARDEN_ROOT,
            scope=FIXTURE_SCOPE,
        )
        cls.review_rows = build_review_rows(
            cls.garden,
            cls.artifacts,
        )

    def test_fixture_scope_loads_review_rows(self) -> None:
        self.assertEqual(len(self.artifacts), 2)
        self.assertEqual(len(self.review_rows), 2)

        metrics = summary_metrics(self.review_rows)
        self.assertEqual(metrics["combo_count"], 2)
        self.assertEqual(metrics["case_count"], 1)
        self.assertAlmostEqual(metrics["average_score"], 0.975)
        self.assertAlmostEqual(metrics["pass_rate"], 1.0)

        combo_ids = {row["combo_id"] for row in self.review_rows}
        self.assertEqual(
            combo_ids,
            {"combo_fixture_baseline", "combo_fixture_challenger"},
        )


class PromptGardenCompareSmokeTest(unittest.TestCase):
    """Confirm that review diffs detect meaningful answer changes."""

    @classmethod
    def setUpClass(cls) -> None:
        artifacts = load_normalized_scope(
            garden_root=FIXTURE_GARDEN_ROOT,
            scope=FIXTURE_SCOPE,
        )
        artifact_by_combo = {
            artifact["combo_id"]: artifact for artifact in artifacts
        }
        cls.baseline = artifact_by_combo["combo_fixture_baseline"]
        cls.challenger = artifact_by_combo["combo_fixture_challenger"]

    def test_field_comparison_and_diff_helpers(self) -> None:
        comparison = compare_record_fields(
            self.baseline,
            self.challenger,
            field_paths=(
                "short_answer",
                "explanation",
                "source_ids",
            ),
        )

        self.assertEqual(comparison["field_count"], 3)
        self.assertEqual(comparison["changed_field_count"], 3)
        self.assertEqual(
            comparison["changed_fields"],
            ["short_answer", "explanation", "source_ids"],
        )

        diff_text = unified_text_diff(
            self.baseline["normalized_text_blocks"]["explanation"][
                "normalized"
            ],
            self.challenger["normalized_text_blocks"]["explanation"][
                "normalized"
            ],
            from_label="baseline",
            to_label="challenger",
        )
        self.assertIn("--- baseline", diff_text)
        self.assertIn("+++ challenger", diff_text)
        self.assertIn("electrons", diff_text)

        alignment = field_segment_alignment(
            self.baseline,
            self.challenger,
            field_path="explanation",
            granularity="sentence",
        )
        self.assertEqual(alignment["granularity"], "sentence")
        self.assertGreaterEqual(alignment["left_count"], 2)
        self.assertGreaterEqual(alignment["right_count"], 2)
        self.assertTrue(alignment["alignment"])


class PromptGardenRunnerSmokeTest(unittest.TestCase):
    """Confirm that the scripted runner can plan from a fixture workspace."""

    def test_runner_dry_run_uses_fixture_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_smoke_copy"
            shutil.copytree(
                FIXTURE_GARDEN_ROOT,
                temp_root,
                dirs_exist_ok=True,
            )

            result = run_prompt_experiment(
                ExperimentRunConfig(
                    garden_root=temp_root,
                    experiment_id=FIXTURE_SCOPE,
                    model="phi4-mini",
                    bot_variant="rag",
                    fewshot_id=None,
                    use_rag=False,
                    case_set=FIXTURE_CASE_SET,
                    run_mode="all",
                    dry_run=True,
                )
            )

        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(result["experiment_id"], FIXTURE_SCOPE)
        self.assertEqual(result["case_set_id"], FIXTURE_CASE_SET)
        self.assertEqual(result["case_count"], 1)
        self.assertEqual(result["combo_count"], 2)
        self.assertEqual(result["target_count"], 2)
        self.assertTrue(
            result["execution"]["signature"],
        )


class PromptGardenStreamlitSmokeTest(unittest.TestCase):
    """Confirm that the Streamlit review app imports and loads fixture data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.review_app = _load_review_app_module()

    def test_review_app_discovers_and_loads_fixture_scope(self) -> None:
        scopes = self.review_app.discover_review_scopes(
            str(FIXTURE_GARDEN_ROOT)
        )
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0]["scope"], FIXTURE_SCOPE)

        bundle = self.review_app.load_scope_bundle(
            str(FIXTURE_GARDEN_ROOT),
            FIXTURE_SCOPE,
        )
        self.assertEqual(len(bundle["artifacts"]), 2)
        self.assertEqual(len(bundle["review_rows"]), 2)
        self.assertEqual(bundle["summary_metrics"]["combo_count"], 2)
        self.assertEqual(
            bundle["experiment"]["id"],
            FIXTURE_SCOPE,
        )
        self.assertEqual(
            set(bundle["signatures"]),
            {
                "sig_fixture_gemma4_fewshot",
                "sig_fixture_phi4",
            },
        )
