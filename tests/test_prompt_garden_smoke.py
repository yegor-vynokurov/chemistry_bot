"""Smoke tests for Prompt Garden review and runner tooling."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from typing import Any
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from chemistry_bot.promptops.garden import PromptGarden  # noqa: E402
from chemistry_bot.promptops.garden_index import (  # noqa: E402
    combo_usage_rows,
    experiment_composition,
    experiment_composition_rows,
    experiment_summary_rows,
    prompt_usage_rows,
)
from chemistry_bot.promptops.review_app_data import (  # noqa: E402
    load_combo_explorer_bundle,
    load_experiment_composition_bundle,
    load_garden_index_bundle,
    load_prompt_explorer_bundle,
    load_prompt_similarity_items,
    load_prompt_workspace_bundle,
)
from chemistry_bot.promptops.review_app_control import (  # noqa: E402
    _matches_search,
    _normalize_control_section,
    _parse_tag_text,
    _queue_prompt_workspace_refresh,
    build_workspace_status_summary,
)
from chemistry_bot.promptops.review_app_analysis import (  # noqa: E402
    build_review_score_summary,
    build_scope_selector_maps,
)
from chemistry_bot.promptops.review_compare import (  # noqa: E402
    compare_record_fields,
    field_segment_alignment,
    unified_text_diff,
)
from chemistry_bot.promptops.review_embeddings import (  # noqa: E402
    embedding_cache_filename,
    nearest_neighbor_rows_for_text_items,
    similarity_bundle,
    text_item_similarity_bundle,
)
from chemistry_bot.promptops.review_store import (  # noqa: E402
    NORMALIZED_REVIEW_ARTIFACT_VERSION,
    build_prompt_snapshot,
    build_review_rows,
    load_normalized_scope,
    relative_artifact_paths,
    summary_metrics,
    write_json,
)
from chemistry_bot.promptops.runner import (  # noqa: E402
    ExperimentRunConfig,
    build_runner_command,
    plan_prompt_experiment,
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
            f"Could not load control-panel app module from {module_path}."
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

    def test_answer_similarity_bundle_still_runs_for_fixture_scope(self) -> None:
        bundle = similarity_bundle(
            records=self.artifacts,
            field_path="explanation",
            same_case_only=True,
            latest_only=False,
            duplicate_threshold=0.92,
        )
        self.assertEqual(bundle["item_kind"], "review_records")
        self.assertEqual(bundle["record_count"], 2)
        self.assertEqual(bundle["pair_count"], 1)
        self.assertEqual(len(bundle["pairwise_rows"]), 1)
        self.assertEqual(
            bundle["pairwise_rows"][0]["case_id"],
            "ionic_bond_intro",
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


class PromptGardenManagementSmokeTest(unittest.TestCase):
    """Confirm that management helpers support safe workspace operations."""

    def _build_management_workspace(
        self,
        temp_root: Path,
    ) -> dict[str, object]:
        garden = PromptGarden(temp_root)
        garden.init()

        system_root = garden.create_root(
            prompt_type="system",
            tree_id="system_management_main",
            title="Management system root",
            text="You are a precise chemistry teacher.",
            tags=["management", "system"],
        )
        system_child = garden.create_child(
            parent_id=system_root["id"],
            title="Management system child",
            text="You are a concise chemistry teacher.",
            tags=["management", "branch"],
            branch="concise",
        )
        user_root = garden.create_root(
            prompt_type="user",
            tree_id="user_management_main",
            title="Management user root",
            text="Answer the student question: {question}",
            tags=["management", "user"],
        )
        orphan_prompt = garden.create_root(
            prompt_type="assistant",
            tree_id="assistant_orphan_main",
            title="Orphan prompt",
            text="This prompt is intentionally unused.",
            tags=["management", "orphan"],
        )
        combo = garden.create_combo(
            title="Management combo",
            prompt_ids={
                "system": system_root["id"],
                "user": user_root["id"],
            },
            tags=["management", "candidate"],
            notes="Fresh combo for management smoke tests.",
        )
        experiment = garden.create_experiment(
            name="management_smoke_experiment",
            goal="Exercise Prompt Garden management helpers.",
            hypothesis="The management helpers should preserve storage integrity.",
            notes="Fresh experiment for safe detach and delete checks.",
            tags=["management", "smoke-test"],
            combo_ids=[combo["id"]],
        )

        return {
            "garden": garden,
            "system_root": system_root,
            "system_child": system_child,
            "user_root": user_root,
            "orphan_prompt": orphan_prompt,
            "combo": combo,
            "experiment": experiment,
        }

    def test_root_prompt_creation_is_reflected_in_prompt_index_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_authoring"
            garden = PromptGarden(temp_root)
            garden.init()

            root_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_authoring_main",
                title="Authoring root prompt",
                text="You are a structured chemistry teacher.",
                tags=["authoring", "root"],
                description="Long and detailed prompt with many restrictions.",
                keywords=["strict", "chemistry", "school"],
            )

            index_bundle = load_garden_index_bundle(str(temp_root))
            self.assertEqual(index_bundle["summary"]["prompt_count"], 1)

            prompt_row = next(
                row for row in index_bundle["prompt_rows"]
                if row["id"] == root_prompt["id"]
            )
            self.assertEqual(prompt_row["title"], "Authoring root prompt")
            self.assertEqual(
                prompt_row["display_title"],
                "Authoring root prompt",
            )
            self.assertEqual(
                prompt_row["description"],
                "Long and detailed prompt with many restrictions.",
            )
            self.assertEqual(
                prompt_row["keywords"],
                ["strict", "chemistry", "school"],
            )
            self.assertEqual(prompt_row["type"], "system")
            self.assertEqual(prompt_row["tree_id"], "system_authoring_main")
            self.assertEqual(prompt_row["branch"], "main")
            self.assertEqual(prompt_row["depth"], 0)
            self.assertEqual(prompt_row["lineage_ids"], [root_prompt["id"]])
            self.assertEqual(prompt_row["child_prompt_ids"], [])
            self.assertEqual(prompt_row["combo_count"], 0)
            self.assertEqual(prompt_row["experiment_count"], 0)
            self.assertTrue(prompt_row["file_exists"])
            self.assertTrue(_matches_search(prompt_row, "restrictions"))
            self.assertTrue(_matches_search(prompt_row, "chemistry"))

    def test_branch_prompt_creation_is_reflected_in_lineage_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_authoring"
            garden = PromptGarden(temp_root)
            garden.init()

            root_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_authoring_main",
                title="Authoring system root",
                text="You are a detailed chemistry teacher.",
                tags=["authoring", "root"],
            )
            branch_prompt = garden.create_child(
                parent_id=root_prompt["id"],
                title="Authoring system branch",
                text="You are a concise chemistry teacher.",
                tags=["authoring", "branch"],
                branch="concise",
            )

            child_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                branch_prompt["id"],
            )
            self.assertEqual(
                child_bundle["prompt"]["parent_id"],
                root_prompt["id"],
            )
            self.assertEqual(
                [row["id"] for row in child_bundle["lineage_rows"]],
                [root_prompt["id"], branch_prompt["id"]],
            )

            root_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                root_prompt["id"],
            )
            self.assertEqual(
                [row["id"] for row in root_bundle["child_rows"]],
                [branch_prompt["id"]],
            )

            index_bundle = load_garden_index_bundle(str(temp_root))
            branch_row = next(
                row for row in index_bundle["prompt_rows"]
                if row["id"] == branch_prompt["id"]
            )
            self.assertEqual(branch_row["parent_id"], root_prompt["id"])
            self.assertEqual(branch_row["branch"], "concise")
            self.assertEqual(branch_row["depth"], 1)

    def test_combo_creation_is_reflected_in_combo_index_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_authoring"
            garden = PromptGarden(temp_root)
            garden.init()

            system_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_authoring_main",
                title="Authoring combo system",
                text="You are a chemistry tutor.",
                tags=["authoring", "system"],
            )
            user_prompt = garden.create_root(
                prompt_type="user",
                tree_id="user_authoring_main",
                title="Authoring combo user",
                text="Answer the question: {question}",
                tags=["authoring", "user"],
            )
            fewshot_prompt = garden.create_root(
                prompt_type="fewshot",
                tree_id="fewshot_authoring_main",
                title="Authoring combo few-shot",
                text=json.dumps(
                    [
                        {
                            "input": "What is an atom?",
                            "output": "An atom is the smallest unit of an element that keeps its chemical identity.",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                tags=["authoring", "fewshot"],
            )

            combo = garden.create_combo(
                title="Authoring combo candidate",
                prompt_ids={
                    "system": system_prompt["id"],
                    "user": user_prompt["id"],
                    "fewshot": fewshot_prompt["id"],
                },
                tags=["authoring", "combo"],
                notes="Created during smoke coverage for Streamlit authoring.",
            )

            index_bundle = load_garden_index_bundle(str(temp_root))
            self.assertEqual(index_bundle["summary"]["combo_count"], 1)

            combo_row = next(
                row for row in index_bundle["combo_rows"]
                if row["id"] == combo["id"]
            )
            self.assertEqual(
                combo_row["prompt_ids"],
                combo["prompt_ids"],
            )
            self.assertEqual(
                set(combo_row["prompt_roles"]),
                {"system", "user", "fewshot"},
            )
            self.assertEqual(combo_row["missing_prompt_ids"], [])
            self.assertEqual(combo_row["experiment_count"], 0)

            combo_bundle = load_combo_explorer_bundle(
                str(temp_root),
                combo["id"],
            )
            self.assertEqual(combo_bundle["combo"]["title"], combo["title"])
            self.assertEqual(
                len(combo_bundle["prompt_role_rows"]),
                3,
            )
            self.assertEqual(
                combo_bundle["summary"]["missing_prompt_ids"],
                [],
            )

    def test_update_experiment_metadata_and_safe_detach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            updated = garden.update_experiment_metadata(
                experiment["id"],
                name="management_smoke_experiment_v2",
                goal="Updated goal",
                hypothesis="Updated hypothesis",
                notes="Updated notes",
                tags=["management", "edited"],
                status="running",
            )

            self.assertEqual(updated["name"], "management_smoke_experiment_v2")
            self.assertEqual(updated["goal"], "Updated goal")
            self.assertEqual(updated["hypothesis"], "Updated hypothesis")
            self.assertEqual(updated["notes"], "Updated notes")
            self.assertEqual(updated["tags"], ["management", "edited"])
            self.assertEqual(updated["status"], "running")

            experiment_index_row = next(
                row for row in garden.list_experiments()
                if row["id"] == experiment["id"]
            )
            self.assertEqual(
                experiment_index_row["name"],
                "management_smoke_experiment_v2",
            )
            self.assertEqual(experiment_index_row["status"], "running")

            preview = garden.preview_detach_combo_from_experiment(
                experiment["id"],
                combo["id"],
            )
            self.assertTrue(preview["safe_to_detach"])
            self.assertEqual(preview["blockers"], [])

            detached = garden.detach_combo_from_experiment(
                experiment["id"],
                combo["id"],
            )
            self.assertEqual(detached["combo_ids"], [])
            self.assertEqual(garden.combo_experiment_ids(combo["id"]), [])
            self.assertFalse(any(
                edge.get("from") == experiment["id"]
                and edge.get("to") == combo["id"]
                and edge.get("kind") == "experiment_uses_combo"
                for edge in garden.list_edges()
            ))

    def test_find_combo_by_prompt_ids_detects_existing_combo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            garden = PromptGarden(temp_root)
            garden.init()

            system_root = garden.create_root(
                prompt_type="system",
                tree_id="system_authoring_main",
                title="Duplicate check system",
                text="You are a careful chemistry teacher.",
                tags=["authoring", "system"],
            )
            user_root = garden.create_root(
                prompt_type="user",
                tree_id="user_authoring_main",
                title="Duplicate check user",
                text="Answer carefully: {question}",
                tags=["authoring", "user"],
            )
            fewshot_a = garden.create_root(
                prompt_type="fewshot",
                tree_id="fewshot_authoring_main",
                title="Duplicate check few-shot A",
                text=json.dumps(
                    [{"input": "What is pH?", "output": "pH measures acidity or basicity."}],
                    ensure_ascii=False,
                    indent=2,
                ),
                tags=["authoring", "fewshot"],
            )
            fewshot_b = garden.create_root(
                prompt_type="fewshot",
                tree_id="fewshot_authoring_main",
                title="Duplicate check few-shot B",
                text=json.dumps(
                    [{"input": "What is a molecule?", "output": "A molecule is a bonded group of atoms."}],
                    ensure_ascii=False,
                    indent=2,
                ),
                tags=["authoring", "fewshot"],
            )

            combo = garden.create_combo(
                title="Duplicate check combo",
                prompt_ids={
                    "system": system_root["id"],
                    "user": user_root["id"],
                    "fewshot": fewshot_a["id"],
                },
                tags=["authoring", "combo"],
                notes="Baseline combo for duplicate lookup coverage.",
            )

            existing_combo = garden.find_combo_by_prompt_ids({
                "fewshot": fewshot_a["id"],
                "user": user_root["id"],
                "system": system_root["id"],
            })
            self.assertIsNotNone(existing_combo)
            self.assertEqual(existing_combo["id"], combo["id"])

            self.assertIsNone(
                garden.find_combo_by_prompt_ids({
                    "system": system_root["id"],
                    "user": user_root["id"],
                })
            )
            self.assertIsNone(
                garden.find_combo_by_prompt_ids({
                    "system": system_root["id"],
                    "user": user_root["id"],
                    "fewshot": fewshot_b["id"],
                })
            )

    def test_attach_combo_updates_experiment_composition_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            system_child = workspace["system_child"]
            user_root = workspace["user_root"]

            attach_target_combo = garden.create_combo(
                title="Attach target combo",
                prompt_ids={
                    "system": system_child["id"],
                    "user": user_root["id"],
                },
                tags=["management", "attach-target"],
                notes="Combo attached after experiment creation.",
            )
            experiment = garden.create_experiment(
                name="attach_management_experiment",
                goal="Verify combo attachment flows.",
                hypothesis="The editor should attach combos without duplicates.",
                notes="Created empty, then populated later.",
                tags=["management", "attach"],
                combo_ids=[],
            )

            self.assertEqual(experiment["combo_ids"], [])

            attached = garden.attach_combo_to_experiment(
                experiment["id"],
                attach_target_combo["id"],
                role="candidate",
                notes="Attached from smoke test.",
            )
            self.assertEqual(
                attached["combo_ids"],
                [attach_target_combo["id"]],
            )

            attached_again = garden.attach_combo_to_experiment(
                experiment["id"],
                attach_target_combo["id"],
            )
            self.assertEqual(
                attached_again["combo_ids"],
                [attach_target_combo["id"]],
            )

            composition_bundle = load_experiment_composition_bundle(
                str(temp_root),
                experiment["id"],
            )
            self.assertEqual(composition_bundle["combo_count"], 1)
            self.assertEqual(composition_bundle["missing_combo_count"], 0)
            self.assertEqual(
                composition_bundle["prompt_id_count"],
                2,
            )

            composition_row = composition_bundle["combo_rows"][0]
            self.assertEqual(
                composition_row["combo_id"],
                attach_target_combo["id"],
            )
            self.assertTrue(composition_row["combo_exists"])
            self.assertEqual(
                composition_row["system_prompt_id"],
                system_child["id"],
            )
            self.assertEqual(
                composition_row["user_prompt_id"],
                user_root["id"],
            )
            self.assertEqual(composition_row["prompt_count"], 2)

    def test_experiment_notes_and_finalization_after_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            result_record = garden.record_experiment_combo_result(
                experiment_id=experiment["id"],
                combo_id=combo["id"],
                score=0.91,
                result_text="This combo handled the tested cases reliably.",
                subject_score=0.84,
                subjective_notes="Strong clarity, but still a bit verbose.",
                metrics={"case_count": 2},
                case_results=[
                    {"case_id": "case_alpha", "score": 0.9, "passed": True},
                    {"case_id": "case_beta", "score": 0.92, "passed": True},
                ],
            )
            self.assertEqual(result_record["combo_id"], combo["id"])

            updated = garden.update_experiment_metadata(
                experiment["id"],
                notes="Post-run interpretation: keep this direction and simplify wording.",
            )
            self.assertIn(
                "Post-run interpretation",
                updated["notes"],
            )

            finalized = garden.finalize_experiment(
                experiment["id"],
                result_text="Promote this combo as the current baseline candidate.",
                subject_score=0.88,
            )
            self.assertEqual(finalized["status"], "completed")
            self.assertEqual(
                finalized["final_result_text"],
                "Promote this combo as the current baseline candidate.",
            )
            self.assertAlmostEqual(
                finalized["final_subject_score"],
                0.88,
            )

            reloaded = garden.get_experiment(experiment["id"])
            self.assertEqual(
                reloaded["summary"]["tested_combo_count"],
                1,
            )
            self.assertAlmostEqual(
                reloaded["summary"]["average_score"],
                0.91,
            )
            self.assertEqual(
                reloaded["subjective_summary"]["subject_score_count"],
                1,
            )
            self.assertAlmostEqual(
                reloaded["subjective_summary"]["average_subject_score"],
                0.84,
            )
            self.assertIn(
                "Post-run interpretation",
                reloaded["notes"],
            )

    def test_dependency_reports_archive_and_safe_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            system_root = workspace["system_root"]
            orphan_prompt = workspace["orphan_prompt"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            prompt_report = garden.inspect_prompt_dependencies(system_root["id"])
            self.assertIn("has_child_prompts", prompt_report["blockers"])
            self.assertIn("used_by_combos", prompt_report["blockers"])
            self.assertIn(combo["id"], prompt_report["combo_ids"])
            self.assertIn(experiment["id"], prompt_report["experiment_ids"])
            prompt_summary = garden.describe_prompt_dependencies(
                system_root["id"]
            )
            self.assertEqual(
                prompt_summary["delete_safety"]["status"],
                "blocked",
            )
            self.assertEqual(
                prompt_summary["usage"]["rows"][0]["count"],
                1,
            )
            self.assertIn(
                "Archive first",
                [
                    row["label"]
                    for row in prompt_summary["delete_safety"]["blocker_rows"]
                ],
            )
            self.assertIn(
                "Archive this prompt before attempting permanent delete.",
                prompt_summary["delete_safety"]["recommended_actions"],
            )
            blocker_messages = [
                row["message"]
                for row in prompt_summary["delete_safety"]["blocker_rows"]
            ]
            self.assertIn(
                "This prompt is not archived yet. Permanent delete stays disabled until it is archived.",
                blocker_messages,
            )
            self.assertTrue(
                any(
                    "Remove or replace this prompt in 1 combo before deleting it."
                    == action
                    for action in prompt_summary["delete_safety"]["recommended_actions"]
                )
            )

            orphan_node = garden.get_node(orphan_prompt["id"])
            orphan_path = temp_root / orphan_node["path"]
            archived_prompt = garden.archive_prompt(
                orphan_prompt["id"],
                reason="Cleanup orphan prompt after review.",
            )
            self.assertTrue(archived_prompt["metadata"]["archived"])
            self.assertIn("archived", archived_prompt["tags"])
            orphan_summary = garden.describe_prompt_dependencies(
                orphan_prompt["id"]
            )
            self.assertEqual(
                orphan_summary["delete_safety"]["status"],
                "safe",
            )
            self.assertEqual(
                orphan_summary["delete_safety"]["recommended_actions"],
                [],
            )
            garden.delete_prompt(orphan_prompt["id"])
            self.assertFalse(orphan_path.exists())
            with self.assertRaises(KeyError):
                garden.get_node(orphan_prompt["id"])

            combo_report = garden.inspect_combo_dependencies(combo["id"])
            self.assertIn("not_archived", combo_report["blockers"])
            self.assertIn("used_by_experiments", combo_report["blockers"])

            archived_combo = garden.archive_combo(
                combo["id"],
                reason="Archive combo before deletion.",
            )
            self.assertEqual(archived_combo["status"], "archived")
            self.assertEqual(archived_combo["test_status"], "archived")
            self.assertTrue(archived_combo["metadata"]["archived"])
            with self.assertRaises(ValueError):
                garden.delete_combo(combo["id"])

            experiment_report = garden.inspect_experiment_dependencies(
                experiment["id"]
            )
            self.assertIn("not_archived", experiment_report["blockers"])
            self.assertEqual(experiment_report["combo_ids"], [combo["id"]])

            archived_experiment = garden.archive_experiment(
                experiment["id"],
                reason="Archive experiment before cleanup.",
            )
            self.assertEqual(archived_experiment["status"], "archived")
            self.assertTrue(
                archived_experiment["metadata"]["archived"]
            )

            safe_experiment_report = garden.inspect_experiment_dependencies(
                experiment["id"]
            )
            self.assertTrue(safe_experiment_report["safe_to_delete"])
            garden.delete_experiment(experiment["id"])
            with self.assertRaises(KeyError):
                garden.get_experiment(experiment["id"])

            safe_combo_report = garden.inspect_combo_dependencies(combo["id"])
            self.assertTrue(safe_combo_report["safe_to_delete"])
            garden.delete_combo(combo["id"])
            with self.assertRaises(KeyError):
                garden.get_combo(combo["id"])

    def test_fixture_artifacts_block_delete_and_detach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_smoke_copy"
            shutil.copytree(
                FIXTURE_GARDEN_ROOT,
                temp_root,
                dirs_exist_ok=True,
            )
            garden = PromptGarden(temp_root)

            experiment_report = garden.inspect_experiment_dependencies(
                FIXTURE_SCOPE
            )
            self.assertFalse(experiment_report["safe_to_delete"])
            self.assertIn("not_archived", experiment_report["blockers"])
            self.assertIn(
                "has_normalized_artifacts",
                experiment_report["blockers"],
            )
            with self.assertRaises(ValueError):
                garden.delete_experiment(FIXTURE_SCOPE)

            experiment = garden.get_experiment(FIXTURE_SCOPE)
            combo_id = experiment["combo_ids"][0]
            detach_preview = garden.preview_detach_combo_from_experiment(
                FIXTURE_SCOPE,
                combo_id,
            )
            self.assertFalse(detach_preview["safe_to_detach"])
            self.assertIn(
                "has_normalized_artifacts",
                detach_preview["blockers"],
            )
            with self.assertRaises(ValueError):
                garden.detach_combo_from_experiment(
                    FIXTURE_SCOPE,
                    combo_id,
                )

    def test_relationship_index_rows_from_management_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            system_root = workspace["system_root"]
            system_child = workspace["system_child"]
            user_root = workspace["user_root"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            prompt_rows = prompt_usage_rows(garden)
            combo_rows = combo_usage_rows(garden)
            experiment_rows = experiment_summary_rows(garden)
            composition_rows = experiment_composition_rows(
                garden,
                experiment["id"],
            )
            composition_bundle = experiment_composition(
                garden,
                experiment["id"],
            )

            prompt_row_by_id = {
                row["id"]: row
                for row in prompt_rows
            }
            combo_row_by_id = {
                row["id"]: row
                for row in combo_rows
            }
            experiment_row_by_id = {
                row["id"]: row
                for row in experiment_rows
            }

            system_prompt_row = prompt_row_by_id[system_root["id"]]
            self.assertEqual(system_prompt_row["combo_ids"], [combo["id"]])
            self.assertEqual(
                system_prompt_row["experiment_ids"],
                [experiment["id"]],
            )
            self.assertEqual(
                system_prompt_row["child_prompt_ids"],
                [system_child["id"]],
            )
            self.assertEqual(system_prompt_row["combo_count"], 1)
            self.assertEqual(system_prompt_row["experiment_count"], 1)

            user_prompt_row = prompt_row_by_id[user_root["id"]]
            self.assertEqual(user_prompt_row["combo_ids"], [combo["id"]])
            self.assertEqual(user_prompt_row["child_count"], 0)

            combo_row = combo_row_by_id[combo["id"]]
            self.assertEqual(combo_row["experiment_ids"], [experiment["id"]])
            self.assertEqual(combo_row["experiment_count"], 1)
            self.assertEqual(combo_row["prompt_ids"]["system"], system_root["id"])
            self.assertEqual(combo_row["prompt_ids"]["user"], user_root["id"])
            self.assertEqual(
                combo_row["prompt_titles_by_role"]["system"],
                system_root["title"],
            )
            self.assertEqual(
                combo_row["prompt_titles_by_role"]["user"],
                user_root["title"],
            )
            self.assertEqual(combo_row["missing_prompt_ids"], [])

            experiment_row = experiment_row_by_id[experiment["id"]]
            self.assertEqual(experiment_row["combo_ids"], [combo["id"]])
            self.assertEqual(experiment_row["combo_count"], 1)
            self.assertEqual(experiment_row["tested_combo_count"], 0)
            self.assertEqual(experiment_row["untested_combo_count"], 1)
            self.assertEqual(experiment_row["missing_combo_count"], 0)

            self.assertEqual(len(composition_rows), 1)
            composition_row = composition_rows[0]
            self.assertTrue(composition_row["combo_exists"])
            self.assertEqual(composition_row["combo_id"], combo["id"])
            self.assertEqual(composition_row["system_prompt_id"], system_root["id"])
            self.assertEqual(composition_row["user_prompt_id"], user_root["id"])
            self.assertEqual(composition_row["prompt_count"], 2)
            self.assertEqual(composition_row["missing_prompt_ids"], [])
            self.assertEqual(composition_bundle["combo_count"], 1)
            self.assertEqual(composition_bundle["prompt_id_count"], 2)

    def test_cached_relationship_bundles_and_fixture_missing_combo_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            experiment = workspace["experiment"]

            index_bundle = load_garden_index_bundle(str(temp_root))
            self.assertEqual(index_bundle["summary"]["prompt_count"], 4)
            self.assertEqual(index_bundle["summary"]["combo_count"], 1)
            self.assertEqual(index_bundle["summary"]["experiment_count"], 1)

            composition_bundle = load_experiment_composition_bundle(
                str(temp_root),
                experiment["id"],
            )
            self.assertEqual(composition_bundle["combo_count"], 1)
            self.assertEqual(composition_bundle["missing_combo_count"], 0)

        fixture_index_bundle = load_garden_index_bundle(
            str(FIXTURE_GARDEN_ROOT)
        )
        self.assertEqual(
            fixture_index_bundle["summary"]["experiment_count"],
            1,
        )
        self.assertEqual(
            fixture_index_bundle["summary"]["combo_count"],
            0,
        )

        fixture_composition = load_experiment_composition_bundle(
            str(FIXTURE_GARDEN_ROOT),
            FIXTURE_SCOPE,
        )
        self.assertEqual(fixture_composition["combo_count"], 2)
        self.assertEqual(fixture_composition["missing_combo_count"], 2)
        self.assertEqual(
            fixture_composition["missing_combo_ids"],
            [
                "combo_fixture_baseline",
                "combo_fixture_challenger",
            ],
        )
        self.assertTrue(all(
            not row["combo_exists"]
            for row in fixture_composition["combo_rows"]
        ))

    def test_prompt_explorer_bundle_includes_text_lineage_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            system_root = workspace["system_root"]
            system_child = workspace["system_child"]
            experiment = workspace["experiment"]
            combo = workspace["combo"]

            fewshot_prompt = garden.create_root(
                prompt_type="fewshot",
                tree_id="fewshot_management_main",
                title="Management few-shot root",
                text=json.dumps(
                    [
                        {
                            "input": "What is an ion?",
                            "output": "An ion is an atom or group with a net electric charge.",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                tags=["management", "fewshot"],
            )

            root_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                system_root["id"],
            )
            self.assertEqual(root_bundle["prompt"]["id"], system_root["id"])
            self.assertIn(
                "precise chemistry teacher",
                root_bundle["prompt"]["text"],
            )
            self.assertEqual(
                [row["id"] for row in root_bundle["lineage_rows"]],
                [system_root["id"]],
            )
            self.assertEqual(
                [row["id"] for row in root_bundle["child_rows"]],
                [system_child["id"]],
            )
            self.assertEqual(
                [row["id"] for row in root_bundle["dependent_combo_rows"]],
                [combo["id"]],
            )
            self.assertEqual(
                [row["id"] for row in root_bundle["dependent_experiment_rows"]],
                [experiment["id"]],
            )

            child_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                system_child["id"],
            )
            self.assertEqual(
                [row["id"] for row in child_bundle["lineage_rows"]],
                [system_root["id"], system_child["id"]],
            )
            self.assertEqual(child_bundle["child_rows"], [])

            fewshot_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                fewshot_prompt["id"],
            )
            self.assertEqual(
                fewshot_bundle["prompt"]["type"],
                "fewshot",
            )
            self.assertIsNone(fewshot_bundle["parsed_fewshot_error"])
            self.assertEqual(
                len(fewshot_bundle["parsed_fewshot_examples"] or []),
                1,
            )
            self.assertEqual(
                fewshot_bundle["parsed_fewshot_examples"][0]["input"],
                "What is an ion?",
            )

    def test_contextual_prompt_workspace_uses_generic_display_title(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_contextual"
            garden = PromptGarden(temp_root)
            garden.init()

            system_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_contextual_main",
                title="Base contextual system prompt",
                text="You are a careful chemistry teacher.",
                tags=["contextual", "system"],
            )
            user_prompt = garden.create_root(
                prompt_type="user",
                tree_id="user_contextual_main",
                title="Base contextual user prompt",
                text="Explain the concept: {question}",
                tags=["contextual", "user"],
            )
            combo = garden.create_combo(
                title="Contextual base combo",
                prompt_ids={
                    "system": system_prompt["id"],
                    "user": user_prompt["id"],
                },
                tags=["contextual", "combo"],
            )

            contextual_combo = garden.get_or_create_context_combo(
                base_combo_id=combo["id"],
                student_context={"grade": "8"},
                teacher_context={"tone": "structured"},
            )
            contextual_system_id = contextual_combo["prompt_ids"]["system"]

            bundle = load_prompt_workspace_bundle(
                str(temp_root),
                contextual_system_id,
            )
            prompt_payload = bundle["prompt"]

            self.assertEqual(
                prompt_payload["display_title"],
                "Contextual system prompt",
            )
            self.assertNotIn(
                "combo_",
                prompt_payload["display_title"],
            )
            self.assertIn(
                system_prompt["id"],
                prompt_payload["description"],
            )
            self.assertEqual(
                prompt_payload["keywords"],
                ["context", "system", "auto-generated"],
            )

            contextual_row = next(
                row for row in prompt_usage_rows(garden)
                if row["id"] == contextual_system_id
            )
            self.assertEqual(
                contextual_row["display_title"],
                "Contextual system prompt",
            )
            self.assertTrue(
                _matches_search(contextual_row, "auto-generated")
            )

    def test_prompt_workspace_bundle_aggregates_usage_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_workspace"
            garden = PromptGarden(temp_root)
            garden.init()

            system_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_workspace_main",
                title="Workspace system prompt",
                text="You are a careful chemistry teacher.",
                tags=["workspace", "system"],
            )
            user_prompt_a = garden.create_root(
                prompt_type="user",
                tree_id="user_workspace_main",
                title="Workspace user prompt A",
                text="Answer question A: {question}",
                tags=["workspace", "user"],
            )
            user_prompt_b = garden.create_child(
                parent_id=user_prompt_a["id"],
                title="Workspace user prompt B",
                text="Answer question B: {question}",
                tags=["workspace", "user", "branch"],
                branch="b",
            )
            combo_a = garden.create_combo(
                title="Workspace combo A",
                prompt_ids={
                    "system": system_prompt["id"],
                    "user": user_prompt_a["id"],
                },
                tags=["workspace", "combo"],
            )
            combo_b = garden.create_combo(
                title="Workspace combo B",
                prompt_ids={
                    "system": system_prompt["id"],
                    "user": user_prompt_b["id"],
                },
                tags=["workspace", "combo"],
            )
            experiment = garden.create_experiment(
                name="workspace_bundle_experiment",
                goal="Check prompt workspace bundle aggregation.",
                hypothesis="The prompt workspace bundle should expose usage and review-derived performance together.",
                tags=["workspace", "bundle"],
                combo_ids=[combo_a["id"], combo_b["id"]],
            )
            orphan_prompt = garden.create_root(
                prompt_type="assistant",
                tree_id="assistant_workspace_main",
                title="Workspace orphan prompt",
                text="This prompt is intentionally unused.",
                tags=["workspace", "orphan"],
            )

            _, normalized_scope_dir = garden.ensure_run_scope_dirs(
                experiment["id"]
            )

            def _write_artifact(
                *,
                run_id: str,
                combo: dict[str, Any],
                model: str,
                score: float,
                created_at: str,
                case_id: str,
            ) -> None:
                filename = f"{run_id}__{combo['id']}__{case_id}.json"
                snapshot = build_prompt_snapshot(
                    garden,
                    base_combo_id=combo["id"],
                    active_combo_id=combo["id"],
                    fewshot_id=None,
                )
                artifact = {
                    "schema_version": NORMALIZED_REVIEW_ARTIFACT_VERSION,
                    "scope": experiment["id"],
                    "id": run_id,
                    "raw_run_id": run_id,
                    "combo_id": combo["id"],
                    "active_combo_id": combo["id"],
                    "experiment_id": experiment["id"],
                    "model": model,
                    "task": "prompt_experiment_case",
                    "created_at": created_at,
                    "question": "Why do ionic bonds form?",
                    "case_set_id": "workspace_cases_v1",
                    "case_id": case_id,
                    "parsed_answer": {
                        "request_type": "answer",
                        "certainty": "high",
                        "experiment": {"kind": "explanation"},
                        "source_ids": [],
                    },
                    "normalized_text_blocks": {
                        "short_answer": {
                            "normalized": "Opposite charges attract."
                        },
                        "explanation": {
                            "normalized": "The ions lower energy by electrostatic attraction."
                        },
                        "comparison_text": (
                            "Opposite charges attract.\n\n"
                            "The ions lower energy by electrostatic attraction."
                        ),
                        "comparison_hash": f"hash_{run_id}",
                    },
                    "metrics": {
                        "score": score,
                        "passed": score >= 0.9,
                        "passed_count": 1 if score >= 0.9 else 0,
                        "total_count": 1,
                        "duration_seconds": 0.42,
                        "validation_ok": True,
                    },
                    "tags": ["workspace", "normalized"],
                    "source_ids": [],
                    "source_usage": {"count": 0},
                    "request_type": "answer",
                    "experiment_kind": "explanation",
                    "answer_lengths": {"example_count": 0},
                    "review_flags": {
                        "parse_error": False,
                        "validation_ok": True,
                        "has_sources": False,
                    },
                    "case_result": {
                        "score": score,
                        "passed": score >= 0.9,
                        "passed_count": 1 if score >= 0.9 else 0,
                        "total_count": 1,
                    },
                    "prompt_snapshot": snapshot,
                    "artifact_paths": relative_artifact_paths(
                        scope=experiment["id"],
                        filename=filename,
                    ),
                    "execution": {
                        "signature": f"sig_{model.replace(':', '_')}",
                        "fewshot_id": None,
                    },
                }
                write_json(normalized_scope_dir / filename, artifact)

            _write_artifact(
                run_id="run_workspace_000001",
                combo=combo_a,
                model="phi4-mini",
                score=1.0,
                created_at="2026-06-25T12:00:01",
                case_id="ionic_bond_intro",
            )
            _write_artifact(
                run_id="run_workspace_000002",
                combo=combo_b,
                model="gemma4:12b",
                score=0.8,
                created_at="2026-06-25T12:00:02",
                case_id="ionic_bond_intro",
            )
            _write_artifact(
                run_id="run_workspace_000003",
                combo=combo_b,
                model="phi4-mini",
                score=0.9,
                created_at="2026-06-25T12:00:03",
                case_id="ionic_bond_followup",
            )

            bundle = load_prompt_workspace_bundle(
                str(temp_root),
                system_prompt["id"],
            )
            self.assertEqual(bundle["prompt"]["id"], system_prompt["id"])
            self.assertEqual(
                bundle["usage_summary"]["child_prompt_count"],
                0,
            )
            self.assertEqual(
                bundle["usage_summary"]["dependent_combo_count"],
                2,
            )
            self.assertEqual(
                bundle["usage_summary"]["dependent_experiment_count"],
                1,
            )
            self.assertEqual(
                bundle["usage_summary"]["recorded_run_count"],
                3,
            )
            self.assertEqual(
                bundle["usage_summary"]["scored_run_count"],
                3,
            )
            self.assertEqual(
                bundle["usage_summary"]["model_count"],
                2,
            )
            self.assertTrue(bundle["usage_summary"]["has_review_history"])
            self.assertEqual(
                bundle["usage_summary"]["latest_run_at"],
                "2026-06-25T12:00:03",
            )
            self.assertEqual(
                bundle["usage_summary"]["models"],
                ["gemma4:12b", "phi4-mini"],
            )
            self.assertEqual(
                bundle["dependency_summary"]["delete_safety"]["status"],
                "blocked",
            )
            self.assertIn(
                "combos",
                bundle["dependency_summary"]["usage"]["headline"].casefold(),
            )
            self.assertEqual(
                [row["combo_id"] for row in bundle["combo_performance_rows"]],
                [combo_a["id"], combo_b["id"]],
            )
            self.assertAlmostEqual(
                bundle["combo_performance_rows"][0]["average_score"],
                1.0,
            )
            self.assertEqual(
                bundle["combo_performance_rows"][0]["best_model"],
                "phi4-mini",
            )
            self.assertAlmostEqual(
                bundle["combo_performance_rows"][1]["average_score"],
                0.85,
            )
            self.assertEqual(
                bundle["combo_performance_rows"][1]["run_count"],
                2,
            )
            self.assertEqual(
                bundle["model_performance_rows"][0]["model"],
                "phi4-mini",
            )
            self.assertAlmostEqual(
                bundle["model_performance_rows"][0]["average_score"],
                0.95,
            )
            self.assertEqual(
                bundle["model_performance_rows"][0]["best_combo_id"],
                combo_a["id"],
            )
            self.assertEqual(
                bundle["top_combo_rows"][0]["combo_id"],
                combo_a["id"],
            )
            self.assertEqual(
                bundle["top_model_rows"][0]["model"],
                "phi4-mini",
            )

            orphan_bundle = load_prompt_workspace_bundle(
                str(temp_root),
                orphan_prompt["id"],
            )
            self.assertFalse(
                orphan_bundle["usage_summary"]["has_review_history"]
            )
            self.assertEqual(
                orphan_bundle["dependency_summary"]["delete_safety"]["status"],
                "blocked",
            )
            self.assertEqual(orphan_bundle["review_rows"], [])
            self.assertEqual(
                orphan_bundle["combo_performance_rows"],
                [],
            )
            self.assertEqual(
                orphan_bundle["model_performance_rows"],
                [],
            )

    def test_prompt_workspace_bundle_for_used_prompt_without_runs_keeps_usage_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_workspace_no_runs"
            garden = PromptGarden(temp_root)
            garden.init()

            system_prompt = garden.create_root(
                prompt_type="system",
                tree_id="system_workspace_no_runs",
                title="No-runs system prompt",
                text="Stay precise and concise.",
                tags=["workspace", "system", "no-runs"],
            )
            user_prompt = garden.create_root(
                prompt_type="user",
                tree_id="user_workspace_no_runs",
                title="No-runs user prompt",
                text="Explain {question}.",
                tags=["workspace", "user", "no-runs"],
            )
            combo = garden.create_combo(
                title="No-runs combo",
                prompt_ids={
                    "system": system_prompt["id"],
                    "user": user_prompt["id"],
                },
                tags=["workspace", "combo", "no-runs"],
            )
            experiment = garden.create_experiment(
                name="workspace_no_runs_experiment",
                goal="Expose prompt usage without any recorded runs.",
                hypothesis="Prompt Workspace should still show combo and experiment dependencies even before the first run.",
                tags=["workspace", "no-runs"],
                combo_ids=[combo["id"]],
            )

            bundle = load_prompt_workspace_bundle(
                str(temp_root),
                system_prompt["id"],
            )

            self.assertEqual(
                bundle["usage_summary"]["dependent_combo_count"],
                1,
            )
            self.assertEqual(
                bundle["usage_summary"]["dependent_experiment_count"],
                1,
            )
            self.assertEqual(
                bundle["usage_summary"]["recorded_run_count"],
                0,
            )
            self.assertFalse(bundle["usage_summary"]["has_review_history"])
            self.assertIsNone(bundle["usage_summary"]["latest_run_at"])
            self.assertEqual(bundle["usage_summary"]["models"], [])
            self.assertEqual(
                [row["id"] for row in bundle["dependent_combo_rows"]],
                [combo["id"]],
            )
            self.assertEqual(
                [row["id"] for row in bundle["dependent_experiment_rows"]],
                [experiment["id"]],
            )
            self.assertEqual(bundle["review_rows"], [])
            self.assertEqual(bundle["combo_performance_rows"], [])
            self.assertEqual(bundle["model_performance_rows"], [])
            self.assertEqual(bundle["top_combo_rows"], [])
            self.assertEqual(bundle["top_model_rows"], [])
            self.assertEqual(
                bundle["dependency_summary"]["delete_safety"]["status"],
                "blocked",
            )
            self.assertIn(
                "Archive this prompt before attempting permanent delete.",
                bundle["dependency_summary"]["delete_safety"]["recommended_actions"],
            )

    def test_stale_fewshot_registry_path_uses_canonical_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            garden = PromptGarden(temp_root)
            garden.init()

            fewshot_prompt = garden.create_root(
                prompt_type="fewshot",
                tree_id="fewshot_management_main",
                title="Fallback few-shot root",
                text=json.dumps(
                    [
                        {
                            "input": "What is an ion?",
                            "output": "An ion is a charged particle.",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                tags=["management", "fewshot"],
            )

            nodes = garden.list_nodes()
            for node in nodes:
                if node["id"] == fewshot_prompt["id"]:
                    node["path"] = "prompts/fewshot/few_000001.md"
                    break
            garden._save_nodes(nodes)

            index_bundle = load_garden_index_bundle(str(temp_root))
            prompt_row = next(
                row for row in index_bundle["prompt_rows"]
                if row["id"] == fewshot_prompt["id"]
            )
            self.assertTrue(prompt_row["file_exists"])
            self.assertTrue(prompt_row["used_path_fallback"])
            self.assertEqual(
                prompt_row["resolved_path"],
                fewshot_prompt["path"],
            )
            self.assertEqual(
                index_bundle["summary"]["prompt_path_fallback_count"],
                1,
            )

            prompt_bundle = load_prompt_explorer_bundle(
                str(temp_root),
                fewshot_prompt["id"],
            )
            self.assertTrue(prompt_bundle["prompt"]["file_exists"])
            self.assertTrue(
                prompt_bundle["prompt"]["used_path_fallback"]
            )
            self.assertEqual(
                prompt_bundle["prompt"]["resolved_path"],
                fewshot_prompt["path"],
            )
            self.assertIn(
                "charged particle",
                prompt_bundle["prompt"]["text"],
            )

    def test_combo_explorer_bundle_includes_prompt_roles_and_experiment_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            system_root = workspace["system_root"]
            user_root = workspace["user_root"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            combo_bundle = load_combo_explorer_bundle(
                str(temp_root),
                combo["id"],
            )
            self.assertEqual(combo_bundle["combo"]["id"], combo["id"])
            self.assertEqual(
                combo_bundle["combo"]["notes"],
                "Fresh combo for management smoke tests.",
            )
            self.assertEqual(
                combo_bundle["combo"]["prompt_ids"]["system"],
                system_root["id"],
            )
            self.assertEqual(
                combo_bundle["combo"]["prompt_ids"]["user"],
                user_root["id"],
            )
            self.assertEqual(len(combo_bundle["prompt_role_rows"]), 2)

            prompt_role_by_role = {
                row["role"]: row
                for row in combo_bundle["prompt_role_rows"]
            }
            self.assertEqual(
                prompt_role_by_role["system"]["prompt_title"],
                system_root["title"],
            )
            self.assertEqual(
                prompt_role_by_role["user"]["prompt_title"],
                user_root["title"],
            )
            self.assertEqual(
                [row["id"] for row in combo_bundle["dependent_experiment_rows"]],
                [experiment["id"]],
            )
            self.assertEqual(
                combo_bundle["summary"]["experiment_count"],
                1,
            )
            self.assertEqual(
                combo_bundle["summary"]["missing_prompt_ids"],
                [],
            )
            self.assertEqual(
                combo_bundle["derived_combo_rows"],
                [],
            )
            self.assertIn(
                "used_by_experiments",
                combo_bundle["dependency_report"]["blockers"],
            )

    def test_index_bundle_reflects_archived_cleanup_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            garden = workspace["garden"]
            orphan_prompt = workspace["orphan_prompt"]
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            garden.archive_prompt(
                orphan_prompt["id"],
                reason="Archive unused prompt before cleanup.",
            )
            garden.archive_combo(
                combo["id"],
                reason="Archive combo before cleanup review.",
            )
            garden.archive_experiment(
                experiment["id"],
                reason="Archive experiment before cleanup review.",
            )

            index_bundle = load_garden_index_bundle(str(temp_root))
            self.assertEqual(
                index_bundle["summary"]["archived_prompt_count"],
                1,
            )
            self.assertEqual(
                index_bundle["summary"]["archived_combo_count"],
                1,
            )
            self.assertEqual(
                index_bundle["summary"]["archived_experiment_count"],
                1,
            )

            archived_prompt_row = next(
                row for row in index_bundle["prompt_rows"]
                if row["id"] == orphan_prompt["id"]
            )
            archived_combo_row = next(
                row for row in index_bundle["combo_rows"]
                if row["id"] == combo["id"]
            )
            archived_experiment_row = next(
                row for row in index_bundle["experiment_rows"]
                if row["id"] == experiment["id"]
            )
            self.assertTrue(archived_prompt_row["is_archived"])
            self.assertTrue(archived_combo_row["is_archived"])
            self.assertTrue(archived_experiment_row["is_archived"])

    def test_prompt_similarity_items_and_nearest_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            system_root = workspace["system_root"]
            system_child = workspace["system_child"]

            prompt_items = load_prompt_similarity_items(str(temp_root))
            self.assertEqual(len(prompt_items), 4)

            prompt_item_by_id = {
                item["item_id"]: item
                for item in prompt_items
            }
            self.assertIn(
                "precise chemistry teacher",
                prompt_item_by_id[system_root["id"]]["text"],
            )
            self.assertEqual(
                prompt_item_by_id[system_root["id"]]["group_id"],
                system_root["tree_id"],
            )

            prompt_bundle = text_item_similarity_bundle(
                prompt_items,
                same_group_only=False,
                duplicate_threshold=0.92,
                item_kind="prompt_items",
            )
            self.assertEqual(prompt_bundle["item_count"], 4)
            self.assertTrue(prompt_bundle["pair_count"] >= 1)

            nearest_rows = nearest_neighbor_rows_for_text_items(
                prompt_items,
                selected_item_id=system_root["id"],
                same_group_only=False,
                top_k=3,
                item_kind="prompt_items",
            )
            self.assertTrue(nearest_rows)
            self.assertEqual(
                nearest_rows[0]["neighbor_item_id"],
                system_child["id"],
            )

            answer_cache_name = embedding_cache_filename(
                scope="exp_fixture_prompt_garden_smoke",
                field_path="comparison_text",
                item_kind="review_records",
            )
            prompt_cache_name = embedding_cache_filename(
                scope="prompt_similarity_workspace",
                field_path="prompt_text",
                item_kind="prompt_items",
            )
            self.assertNotEqual(answer_cache_name, prompt_cache_name)
            self.assertIn("prompt_items", prompt_cache_name)

    def test_runner_command_builder_and_plan_preview_respect_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir) / "prompt_garden_management"
            workspace = self._build_management_workspace(temp_root)
            combo = workspace["combo"]
            experiment = workspace["experiment"]

            case_set_path = temp_root / "cases" / "management_case_set.json"
            case_set_path.write_text(
                json.dumps(
                    {
                        "id": "management_case_set",
                        "name": "Management smoke test cases",
                        "cases": [
                            {
                                "id": "case_alpha",
                                "question": "What is an atom?",
                            },
                            {
                                "id": "case_beta",
                                "question": "What is a molecule?",
                            },
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            config = ExperimentRunConfig(
                garden_root=temp_root,
                experiment_id=experiment["id"],
                model="phi4-mini",
                bot_variant="rag",
                fewshot_id=None,
                use_rag=False,
                case_set="management_case_set",
                only_case_ids=("case_alpha",),
                only_combo_ids=(combo["id"],),
                run_mode="all",
                dry_run=True,
            )

            plan = plan_prompt_experiment(config)
            self.assertEqual(plan["experiment_id"], experiment["id"])
            self.assertEqual(plan["case_set_id"], "management_case_set")
            self.assertEqual(plan["case_count"], 1)
            self.assertEqual(plan["combo_count"], 1)
            self.assertEqual(plan["target_count"], 1)
            self.assertEqual(
                plan["targets_preview"][0]["case_id"],
                "case_alpha",
            )
            self.assertEqual(
                plan["targets_preview"][0]["combo_id"],
                combo["id"],
            )

            command = build_runner_command(config)
            self.assertIn("--experiment-id", command)
            self.assertIn(experiment["id"], command)
            self.assertIn("--case-set 'management_case_set'", command)
            self.assertIn("--only-case-id 'case_alpha'", command)
            self.assertIn(f"--only-combo '{combo['id']}'", command)
            self.assertIn("--run-mode 'all'", command)
            self.assertIn("--no-fewshot", command)
            self.assertIn("--no-rag", command)
            self.assertIn("--dry-run", command)


class PromptGardenStreamlitSmokeTest(unittest.TestCase):
    """Confirm that the Streamlit control panel imports and loads fixture data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.review_app = _load_review_app_module()

    def test_control_and_analysis_loaders_align_for_fixture_scope(self) -> None:
        index_bundle = load_garden_index_bundle(
            str(FIXTURE_GARDEN_ROOT)
        )
        composition_bundle = load_experiment_composition_bundle(
            str(FIXTURE_GARDEN_ROOT),
            FIXTURE_SCOPE,
        )
        scope_bundle = self.review_app.load_scope_bundle(
            str(FIXTURE_GARDEN_ROOT),
            FIXTURE_SCOPE,
        )

        self.assertEqual(
            index_bundle["summary"]["experiment_count"],
            1,
        )
        self.assertEqual(
            composition_bundle["experiment"]["id"],
            FIXTURE_SCOPE,
        )
        self.assertEqual(
            composition_bundle["combo_count"],
            scope_bundle["summary_metrics"]["combo_count"],
        )
        self.assertEqual(
            composition_bundle["missing_combo_count"],
            2,
        )
        self.assertEqual(
            composition_bundle["summary"]["missing_combo_count"],
            2,
        )

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

    def test_workspace_status_summary_helper(self) -> None:
        scopes = self.review_app.discover_review_scopes(
            str(FIXTURE_GARDEN_ROOT)
        )
        index_bundle = load_garden_index_bundle(
            str(FIXTURE_GARDEN_ROOT)
        )
        summary = build_workspace_status_summary(
            garden_root=str(FIXTURE_GARDEN_ROOT),
            index_bundle=index_bundle,
            scopes=scopes,
        )
        self.assertEqual(summary["garden_root"], str(FIXTURE_GARDEN_ROOT))
        self.assertEqual(summary["review_scope_count"], 1)
        self.assertEqual(summary["experiment_count"], 1)
        self.assertEqual(summary["combo_count"], 0)
        self.assertEqual(summary["prompt_count"], 0)
        self.assertEqual(summary["active_experiment_count"], 1)
        self.assertEqual(summary["archived_total"], 0)

    def test_prompt_workspace_section_normalizes_old_label(self) -> None:
        self.assertEqual(
            _normalize_control_section("Workspace"),
            "Prompt Workspace",
        )
        self.assertEqual(
            _normalize_control_section("Prompt Explorer"),
            "Prompt Workspace",
        )
        self.assertEqual(
            _normalize_control_section("Prompt Workspace"),
            "Prompt Workspace",
        )

    def test_prompt_workspace_refresh_helper_keeps_prompt_context(self) -> None:
        st.session_state.clear()
        st.session_state["prompt_explorer_include_archived"] = False
        with patch(
            "chemistry_bot.promptops.review_app_control.st.cache_data.clear"
        ) as cache_clear_mock, patch(
            "chemistry_bot.promptops.review_app_control.st.rerun"
        ) as rerun_mock:
            _queue_prompt_workspace_refresh(
                "Prompt `sys_000001` archived.",
                selected_prompt_id="sys_000001",
                ensure_archived_visible=True,
            )

        self.assertEqual(
            st.session_state.get("prompt_garden_cleanup_flash_message"),
            "Prompt `sys_000001` archived.",
        )
        self.assertEqual(
            st.session_state.get("prompt_garden_selected_prompt_id"),
            "sys_000001",
        )
        self.assertEqual(
            st.session_state.get("prompt_garden_control_section_redirect"),
            "Prompt Workspace",
        )
        self.assertTrue(
            st.session_state.get("prompt_explorer_include_archived")
        )
        cache_clear_mock.assert_called_once()
        rerun_mock.assert_called_once()
        st.session_state.clear()

    def test_prompt_workspace_refresh_helper_clears_deleted_selection(self) -> None:
        st.session_state.clear()
        st.session_state["prompt_garden_control_section"] = "Cleanup"
        st.session_state["prompt_garden_selected_prompt_id"] = "sys_000009"
        st.session_state["prompt_explorer_selected_prompt"] = (
            "sys_000009 | prompt slated for deletion"
        )
        with patch(
            "chemistry_bot.promptops.review_app_control.st.cache_data.clear"
        ) as cache_clear_mock, patch(
            "chemistry_bot.promptops.review_app_control.st.rerun"
        ) as rerun_mock:
            _queue_prompt_workspace_refresh(
                "Prompt `sys_000009` deleted permanently.",
                clear_selected_prompt=True,
            )

        self.assertEqual(
            st.session_state.get("prompt_garden_cleanup_flash_message"),
            "Prompt `sys_000009` deleted permanently.",
        )
        self.assertEqual(
            st.session_state.get("prompt_garden_control_section_redirect"),
            "Prompt Workspace",
        )
        self.assertIsNone(
            st.session_state.get("prompt_garden_selected_prompt_id")
        )
        self.assertIsNone(
            st.session_state.get("prompt_explorer_selected_prompt")
        )
        cache_clear_mock.assert_called_once()
        rerun_mock.assert_called_once()
        st.session_state.clear()

    def test_authoring_tag_parser_normalizes_duplicates(self) -> None:
        tags = _parse_tag_text(
            "chemistry, safety, chemistry,  branch , Safety, ,"
        )
        self.assertEqual(
            tags,
            ["chemistry", "safety", "branch"],
        )

    def test_analysis_scope_selector_maps_and_score_summary(self) -> None:
        scopes = self.review_app.discover_review_scopes(
            str(FIXTURE_GARDEN_ROOT)
        )
        selector_maps = build_scope_selector_maps(scopes)
        self.assertIn("by_id", selector_maps)
        self.assertIn("by_name", selector_maps)
        self.assertIn(
            FIXTURE_SCOPE,
            selector_maps["by_id"].values(),
        )
        self.assertIn(
            FIXTURE_SCOPE,
            selector_maps["by_name"].values(),
        )

        bundle = self.review_app.load_scope_bundle(
            str(FIXTURE_GARDEN_ROOT),
            FIXTURE_SCOPE,
        )
        score_summary = build_review_score_summary(bundle["review_rows"])
        self.assertEqual(score_summary["row_count"], 2)
        self.assertEqual(score_summary["scored_row_count"], 2)
        self.assertAlmostEqual(score_summary["min_score"], 0.95)
        self.assertAlmostEqual(score_summary["max_score"], 1.0)
        self.assertAlmostEqual(score_summary["average_score"], 0.975)
