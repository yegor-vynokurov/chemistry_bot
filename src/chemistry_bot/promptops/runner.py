"""Script-oriented Prompt Garden experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
import json
import re

from chemistry_bot.cli.legacy_bot import CliBot as LegacyCliBot
from chemistry_bot.cli.rag_bot import CliBot as RagCliBot
from chemistry_bot.promptops.eval import (
    DEFAULT_CHEMISTRY_CASE_SET_ID,
    default_case_set_payload,
    evaluate_case,
    summarize_results,
)
from chemistry_bot.promptops.garden import PromptGarden
from chemistry_bot.promptops.review_store import (
    NORMALIZED_REVIEW_ARTIFACT_VERSION,
    RAW_RUN_ARTIFACT_VERSION,
    build_answer_lengths,
    build_normalized_text_blocks,
    build_prompt_snapshot,
    read_json,
    relative_artifact_paths,
    write_json,
    write_summary_report,
)
from chemistry_bot.retrieval import (
    INTROCHEM_CHROMA_DIR,
    INTROCHEM_COLLECTION_NAME,
    RAGConfig,
)


RUNNER_VERSION = "prompt-garden-runner-v1"
CASE_TASK_NAME = "prompt_experiment_case"


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Configuration for one scripted Prompt Garden experiment run."""

    garden_root: Path
    experiment_id: str
    model: str = "phi4-mini"
    bot_variant: Literal["legacy", "rag"] = "rag"
    fewshot_id: str | None = "fsh_000002"
    use_rag: bool = True
    rag_k: int = 4
    candidate_k: int = 12
    max_context_chars: int = 6500
    max_history_messages: int = 12
    case_set: str | None = DEFAULT_CHEMISTRY_CASE_SET_ID
    only_case_ids: tuple[str, ...] = ()
    skip_case_ids: tuple[str, ...] = ()
    only_combo_ids: tuple[str, ...] = ()
    skip_combo_ids: tuple[str, ...] = ()
    run_mode: Literal["missing", "failed", "all"] = "missing"
    dry_run: bool = False


def _now_iso() -> str:
    return PromptGarden._now()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash_data(value: Any) -> str:
    return PromptGarden._hash_data(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def _case_set_payload_copy() -> dict[str, Any]:
    return json.loads(
        json.dumps(
            default_case_set_payload(),
            ensure_ascii=False,
        )
    )


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def resolve_case_set_path(
    garden_root: Path,
    reference: str | None,
) -> Path | None:
    """Resolve a case-set reference against one Prompt Garden workspace."""

    if not reference:
        return None

    direct_path = Path(reference)
    if direct_path.exists():
        return direct_path

    case_filename = reference
    if not case_filename.endswith(".json"):
        case_filename = f"{case_filename}.json"

    candidate = Path(garden_root) / "cases" / case_filename
    if candidate.exists():
        return candidate

    return None


def load_case_set_payload(
    garden_root: Path,
    reference: str | None,
) -> dict[str, Any]:
    """Load one case-set payload, falling back to the built-in default."""

    resolved_path = resolve_case_set_path(garden_root, reference)

    if resolved_path is None:
        return _case_set_payload_copy()

    payload = read_json(resolved_path)
    if "cases" not in payload or not isinstance(payload["cases"], list):
        raise ValueError(
            f"Case set {resolved_path} must contain a top-level 'cases' list."
        )

    return payload


def list_case_set_rows(
    garden_root: Path,
) -> list[dict[str, Any]]:
    """Describe the built-in and local case-set options for one workspace."""

    rows: list[dict[str, Any]] = []
    default_payload = _case_set_payload_copy()
    rows.append({
        "reference": str(default_payload.get("id") or DEFAULT_CHEMISTRY_CASE_SET_ID),
        "id": str(default_payload.get("id") or DEFAULT_CHEMISTRY_CASE_SET_ID),
        "title": default_payload.get("name") or "Built-in default case set",
        "case_count": len(default_payload.get("cases", [])),
        "source": "built_in_default",
        "path": None,
    })

    cases_dir = Path(garden_root) / "cases"
    if not cases_dir.exists():
        return rows

    for path in sorted(cases_dir.glob("*.json")):
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        cases = payload.get("cases")
        if not isinstance(cases, list):
            continue
        reference = path.stem
        rows.append({
            "reference": reference,
            "id": str(payload.get("id") or reference),
            "title": payload.get("name") or reference,
            "case_count": len(cases),
            "source": "workspace_file",
            "path": str(path),
        })

    return rows


def build_runner_command(
    config: ExperimentRunConfig,
    *,
    python_executable: str = r".\.venv\Scripts\python.exe",
    script_path: str = "scripts/run_prompt_experiment.py",
    include_filters: bool = True,
    dry_run: bool | None = None,
) -> str:
    """Render a copy-pasteable PowerShell command for one runner config."""

    effective_dry_run = config.dry_run if dry_run is None else dry_run
    command_lines = [
        f"& {_powershell_quote(python_executable)} {_powershell_quote(script_path)}",
        f"--garden-root {_powershell_quote(str(config.garden_root))}",
        f"--experiment-id {_powershell_quote(config.experiment_id)}",
        f"--model {_powershell_quote(config.model)}",
        f"--bot-variant {_powershell_quote(config.bot_variant)}",
    ]

    if config.fewshot_id is None:
        command_lines.append("--no-fewshot")
    else:
        command_lines.append(
            f"--fewshot-id {_powershell_quote(config.fewshot_id)}"
        )

    if not config.use_rag:
        command_lines.append("--no-rag")

    command_lines.extend([
        f"--rag-k {config.rag_k}",
        f"--candidate-k {config.candidate_k}",
        f"--max-context-chars {config.max_context_chars}",
        f"--max-history-messages {config.max_history_messages}",
        f"--run-mode {_powershell_quote(config.run_mode)}",
    ])

    if config.case_set:
        command_lines.append(
            f"--case-set {_powershell_quote(str(config.case_set))}"
        )

    if include_filters:
        for case_id in config.only_case_ids:
            command_lines.append(
                f"--only-case-id {_powershell_quote(case_id)}"
            )
        for case_id in config.skip_case_ids:
            command_lines.append(
                f"--skip-case-id {_powershell_quote(case_id)}"
            )
        for combo_id in config.only_combo_ids:
            command_lines.append(
                f"--only-combo {_powershell_quote(combo_id)}"
            )
        for combo_id in config.skip_combo_ids:
            command_lines.append(
                f"--skip-combo {_powershell_quote(combo_id)}"
            )

    if effective_dry_run:
        command_lines.append("--dry-run")

    return " `\n  ".join(command_lines)


def plan_prompt_experiment(
    config: ExperimentRunConfig,
) -> dict[str, Any]:
    """Build the same execution preview used by runner dry-runs."""

    runner = PromptExperimentRunner(config)
    return runner.plan()


class PromptExperimentRunner:
    """Execute Prompt Garden experiments outside the notebook."""

    def __init__(self, config: ExperimentRunConfig) -> None:
        self.config = config
        self.garden = PromptGarden(config.garden_root)
        self.garden.init()
        self.experiment = self.garden.get_experiment(config.experiment_id)
        self.case_set_payload = self._load_case_set_payload()
        self.case_set_id = str(self.case_set_payload["id"])
        self.case_set_hash = _hash_data(self.case_set_payload["cases"])
        self.selected_cases = self._select_cases(
            self.case_set_payload["cases"]
        )
        self.case_filters_active = bool(
            self.config.only_case_ids
            or self.config.skip_case_ids
        )
        self.selected_case_ids = [
            case["id"]
            for case in self.selected_cases
        ]
        self.selected_combo_ids = self._select_combo_ids()
        self.scope = self.config.experiment_id or "_adhoc"
        self.raw_scope_dir, self.normalized_scope_dir = (
            self.garden.ensure_run_scope_dirs(self.scope)
        )
        self.report_scope_dir = self.garden.ensure_report_scope_dir(
            self.scope
        )
        self.execution_context = self._build_execution_context()
        self.execution_signature = _hash_data(self.execution_context)
        self._prompt_snapshot_cache: dict[
            tuple[str, str, str | None],
            dict[str, Any],
        ] = {}

    def _resolve_case_set_path(self) -> Path | None:
        return resolve_case_set_path(
            self.garden.root,
            self.config.case_set,
        )

    def _load_case_set_payload(self) -> dict[str, Any]:
        return load_case_set_payload(
            self.garden.root,
            self.config.case_set,
        )

    def _select_cases(
        self,
        cases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen_ids: set[str] = set()
        ordered_cases: list[dict[str, Any]] = []

        for case in cases:
            case_id = case.get("id")
            if not case_id:
                raise ValueError("Each Prompt Garden case must have an 'id'.")
            if case_id in seen_ids:
                raise ValueError(f"Duplicate case id in case set: {case_id}")
            seen_ids.add(case_id)
            ordered_cases.append(case)

        available_ids = {case["id"] for case in ordered_cases}
        unknown_only = set(self.config.only_case_ids) - available_ids
        unknown_skip = set(self.config.skip_case_ids) - available_ids

        if unknown_only:
            raise ValueError(
                "Unknown --only-case-id value(s): "
                + ", ".join(sorted(unknown_only))
            )

        if unknown_skip:
            raise ValueError(
                "Unknown --skip-case-id value(s): "
                + ", ".join(sorted(unknown_skip))
            )

        if self.config.only_case_ids:
            selected = [
                case
                for case in ordered_cases
                if case["id"] in set(self.config.only_case_ids)
            ]
        else:
            selected = [
                case
                for case in ordered_cases
                if case["id"] not in set(self.config.skip_case_ids)
            ]

        if not selected:
            raise ValueError("No Prompt Garden cases were selected for this run.")

        return selected

    def _select_combo_ids(self) -> list[str]:
        experiment_combo_ids = list(self.experiment.get("combo_ids", []))
        if not experiment_combo_ids:
            raise ValueError(
                f"Experiment {self.config.experiment_id} has no attached combos."
            )

        available_ids = set(experiment_combo_ids)
        unknown_only = set(self.config.only_combo_ids) - available_ids
        unknown_skip = set(self.config.skip_combo_ids) - available_ids

        if unknown_only:
            raise ValueError(
                "Unknown or unattached --only-combo value(s): "
                + ", ".join(sorted(unknown_only))
            )

        if unknown_skip:
            raise ValueError(
                "Unknown or unattached --skip-combo value(s): "
                + ", ".join(sorted(unknown_skip))
            )

        if self.config.only_combo_ids:
            selected = [
                combo_id
                for combo_id in experiment_combo_ids
                if combo_id in set(self.config.only_combo_ids)
            ]
        else:
            selected = [
                combo_id
                for combo_id in experiment_combo_ids
                if combo_id not in set(self.config.skip_combo_ids)
            ]

        if not selected:
            raise ValueError("No Prompt Garden combos were selected for this run.")

        return selected

    def _build_execution_context(self) -> dict[str, Any]:
        rag_payload = {
            "enabled": (
                self.config.bot_variant == "rag"
                and self.config.use_rag
            ),
            "top_k": self.config.rag_k,
            "candidate_k": self.config.candidate_k,
            "max_context_chars": self.config.max_context_chars,
        }

        return {
            "runner_version": RUNNER_VERSION,
            "bot_variant": self.config.bot_variant,
            "model": self.config.model,
            "fewshot_id": self.config.fewshot_id,
            "case_set_id": self.case_set_id,
            "case_set_hash": self.case_set_hash,
            "max_history_messages": self.config.max_history_messages,
            "rag": rag_payload,
        }

    def _make_bot(self, combo_id: str) -> LegacyCliBot | RagCliBot:
        shared_kwargs = {
            "model_name": self.config.model,
            "garden_root": self.garden.root,
            "combo_id": combo_id,
            "fewshot_id": self.config.fewshot_id,
            "max_history_messages": self.config.max_history_messages,
            "materialize_context": True,
        }

        if self.config.bot_variant == "legacy":
            return LegacyCliBot(**shared_kwargs)

        return RagCliBot(
            **shared_kwargs,
            rag_config=RAGConfig(
                enabled=self.config.use_rag,
                db_path=INTROCHEM_CHROMA_DIR,
                collection_name=INTROCHEM_COLLECTION_NAME,
                embedding_model=None,
                ollama_base_url=None,
                chunks_path=None,
                top_k=self.config.rag_k,
                candidate_k=max(self.config.candidate_k, self.config.rag_k),
                max_per_section=2,
                max_context_chars=self.config.max_context_chars,
                filter_default_retrieval=True,
                fail_open=True,
            ),
        )

    def _load_existing_normalized_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        index: dict[tuple[str, str], dict[str, Any]] = {}

        if not self.normalized_scope_dir.exists():
            return index

        for path in sorted(self.normalized_scope_dir.glob("*.json")):
            try:
                artifact = read_json(path)
            except json.JSONDecodeError:
                continue

            execution = artifact.get("execution") or {}
            if execution.get("signature") != self.execution_signature:
                continue

            combo_id = artifact.get("combo_id")
            case_id = artifact.get("case_id")
            created_at = artifact.get("created_at", "")
            if not combo_id or not case_id:
                continue

            key = (combo_id, case_id)
            current = index.get(key)
            if current is None or created_at >= current.get("created_at", ""):
                artifact["_path"] = str(path)
                index[key] = artifact

        return index

    def _should_run_case(
        self,
        existing_artifact: dict[str, Any] | None,
    ) -> bool:
        if self.config.run_mode == "all":
            return True

        if self.config.run_mode == "missing":
            return existing_artifact is None

        if self.config.run_mode == "failed":
            if existing_artifact is None:
                return False
            metrics = existing_artifact.get("metrics") or {}
            return not bool(metrics.get("passed"))

        raise ValueError(f"Unsupported run mode: {self.config.run_mode}")

    def _artifact_filename(
        self,
        run_id: str,
        combo_id: str,
        case_id: str,
    ) -> str:
        return (
            f"{run_id}__{combo_id}__{_slugify(case_id)}.json"
        )

    def _artifact_tags(
        self,
        case_result: dict[str, Any],
        validation_ok: bool,
        raw_output_text: str | None,
    ) -> list[str]:
        tags = [
            self.config.bot_variant,
            "rag" if self.config.bot_variant == "rag" and self.config.use_rag else "no_rag",
        ]

        if validation_ok:
            tags.append("validation_ok")
        else:
            tags.append("validation_failed")

        if case_result.get("passed"):
            tags.append("case_passed")
        else:
            tags.append("case_failed")

        if not raw_output_text:
            tags.append("empty_output")

        if case_result.get("answer") is None:
            tags.append("parse_error")

        return tags

    def _prompt_snapshot(
        self,
        base_combo_id: str,
        active_combo_id: str,
        bot: LegacyCliBot | RagCliBot,
    ) -> dict[str, Any]:
        cache_key = (
            base_combo_id,
            active_combo_id,
            self.config.fewshot_id,
        )
        cached = self._prompt_snapshot_cache.get(cache_key)
        if cached is not None:
            return cached

        snapshot = build_prompt_snapshot(
            garden=self.garden,
            base_combo_id=base_combo_id,
            active_combo_id=active_combo_id,
            fewshot_id=self.config.fewshot_id,
            selected_fewshot_example_ids=list(
                getattr(bot, "selected_fewshot_example_ids", [])
            ),
        )
        self._prompt_snapshot_cache[cache_key] = snapshot
        return snapshot

    def _build_raw_artifact(
        self,
        run_record: dict[str, Any],
        case: dict[str, Any],
        case_result: dict[str, Any],
        duration_seconds: float,
        raw_output_text: str | None,
        parsed_answer: dict[str, Any] | None,
        error_text: str | None,
        bot: LegacyCliBot | RagCliBot,
        prompt_snapshot: dict[str, Any],
        artifact_paths: dict[str, str],
    ) -> dict[str, Any]:
        return {
            **run_record,
            "schema_version": RAW_RUN_ARTIFACT_VERSION,
            "scope": self.scope,
            "case_set_id": self.case_set_id,
            "case_id": case["id"],
            "question": case["question"],
            "prompt_snapshot": prompt_snapshot,
            "artifact_paths": artifact_paths,
            "request_params": {
                "bot_variant": self.config.bot_variant,
                "max_history_messages": self.config.max_history_messages,
            },
            "timings": {
                "duration_seconds": round(duration_seconds, 4),
            },
            "runner_version": RUNNER_VERSION,
            "execution": {
                **self.execution_context,
                "signature": self.execution_signature,
            },
            "raw_output_text": raw_output_text,
            "parsed_answer": parsed_answer,
            "case_result": case_result,
            "error": error_text,
            "rag_context": (
                (bot.last_input_data or {}).get("rag")
                if isinstance(bot.last_input_data, dict)
                else None
            ),
        }

    def _build_normalized_artifact(
        self,
        run_record: dict[str, Any],
        case: dict[str, Any],
        case_result: dict[str, Any],
        duration_seconds: float,
        raw_output_text: str | None,
        parsed_answer: dict[str, Any] | None,
        prompt_snapshot: dict[str, Any],
        artifact_paths: dict[str, str],
    ) -> dict[str, Any]:
        experiment_block = parsed_answer.get("experiment") if parsed_answer else {}
        if not isinstance(experiment_block, dict):
            experiment_block = {}

        validation_ok = bool(run_record["validation_ok"])
        source_ids = []
        if parsed_answer:
            source_ids = list(parsed_answer.get("source_ids", []))
        normalized_text_blocks = build_normalized_text_blocks(
            parsed_answer=parsed_answer,
            raw_output_text=raw_output_text,
        )

        return {
            "schema_version": NORMALIZED_REVIEW_ARTIFACT_VERSION,
            "scope": self.scope,
            "id": run_record["id"],
            "raw_run_id": run_record["id"],
            "combo_id": run_record["combo_id"],
            "active_combo_id": (
                prompt_snapshot.get("active_combo") or {}
            ).get("id"),
            "experiment_id": self.config.experiment_id,
            "model": self.config.model,
            "task": CASE_TASK_NAME,
            "created_at": run_record["created_at"],
            "question": case["question"],
            "case_set_id": self.case_set_id,
            "case_id": case["id"],
            "parsed_answer": parsed_answer,
            "normalized_text_blocks": normalized_text_blocks,
            "metrics": {
                "score": case_result["score"],
                "passed": case_result["passed"],
                "passed_count": case_result["passed_count"],
                "total_count": case_result["total_count"],
                "duration_seconds": round(duration_seconds, 4),
                "validation_ok": validation_ok,
            },
            "tags": self._artifact_tags(
                case_result=case_result,
                validation_ok=validation_ok,
                raw_output_text=raw_output_text,
            ),
            "source_ids": source_ids,
            "source_usage": {
                "count": len(source_ids),
            },
            "request_type": parsed_answer.get("request_type") if parsed_answer else None,
            "experiment_kind": experiment_block.get("kind"),
            "answer_lengths": build_answer_lengths(
                parsed_answer=parsed_answer,
                normalized_text_blocks=normalized_text_blocks,
            ),
            "review_flags": {
                "parse_error": parsed_answer is None,
                "validation_ok": validation_ok,
                "has_sources": bool(source_ids),
            },
            "case_result": case_result,
            "prompt_snapshot": prompt_snapshot,
            "artifact_paths": artifact_paths,
            "execution": {
                **self.execution_context,
                "signature": self.execution_signature,
            },
        }

    def _run_case(
        self,
        bot: LegacyCliBot | RagCliBot,
        combo: dict[str, Any],
        case: dict[str, Any],
    ) -> dict[str, Any]:
        bot.student_context.protocol_context = case.get(
            "protocol_context",
            "No verified experimental protocol was retrieved.",
        )
        if getattr(bot, "materialize_context", False):
            bot.configure_contexts(
                context=bot.student_context,
                teacher_context=bot.teacher_context,
            )

        session_id = (
            f"{self.config.experiment_id}_{combo['id']}_{case['id']}"
        )
        started = perf_counter()
        answer = bot.invoke_once(
            user_text=case["question"],
            session_id=session_id,
            experiment_id=self.config.experiment_id,
            silent=True,
            record_run=False,
        )
        duration_seconds = perf_counter() - started

        parsed_answer = answer.model_dump() if answer is not None else None
        raw_output_text = getattr(bot, "last_raw_response_text", None)
        error_text = (
            getattr(bot, "last_parsing_error", None)
            or getattr(bot, "last_invoke_error", None)
        )
        case_result = evaluate_case(answer, case)
        prompt_snapshot = self._prompt_snapshot(
            base_combo_id=combo["id"],
            active_combo_id=bot.combo_id,
            bot=bot,
        )

        run_record = self.garden.add_run(
            combo_id=combo["id"],
            experiment_id=self.config.experiment_id,
            task=CASE_TASK_NAME,
            model=self.config.model,
            input_data=bot.last_input_data or {"question": case["question"]},
            output_data=parsed_answer if parsed_answer is not None else raw_output_text,
            validation_ok=answer is not None,
            error=error_text,
            metrics={
                "case_id": case["id"],
                "case_score": case_result["score"],
                "case_passed": case_result["passed"],
                "duration_seconds": round(duration_seconds, 4),
            },
        )
        filename = self._artifact_filename(
            run_id=run_record["id"],
            combo_id=combo["id"],
            case_id=case["id"],
        )
        artifact_paths = relative_artifact_paths(
            scope=self.scope,
            filename=filename,
        )

        raw_artifact = self._build_raw_artifact(
            run_record=run_record,
            case=case,
            case_result=case_result,
            duration_seconds=duration_seconds,
            raw_output_text=raw_output_text,
            parsed_answer=parsed_answer,
            error_text=error_text,
            bot=bot,
            prompt_snapshot=prompt_snapshot,
            artifact_paths=artifact_paths,
        )
        normalized_artifact = self._build_normalized_artifact(
            run_record=run_record,
            case=case,
            case_result=case_result,
            duration_seconds=duration_seconds,
            raw_output_text=raw_output_text,
            parsed_answer=parsed_answer,
            prompt_snapshot=prompt_snapshot,
            artifact_paths=artifact_paths,
        )

        raw_path = self.raw_scope_dir / filename
        normalized_path = self.normalized_scope_dir / filename
        write_json(raw_path, raw_artifact)
        write_json(normalized_path, normalized_artifact)
        normalized_artifact["_path"] = str(normalized_path)
        return normalized_artifact

    def _combo_case_results_from_index(
        self,
        combo_id: str,
        normalized_index: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        case_results: list[dict[str, Any]] = []
        missing_case_ids: list[str] = []

        for case in self.selected_cases:
            artifact = normalized_index.get((combo_id, case["id"]))
            if artifact is None:
                missing_case_ids.append(case["id"])
                continue
            case_results.append(artifact["case_result"])

        return case_results, missing_case_ids

    def _update_combo_summary(
        self,
        combo_id: str,
        case_results: list[dict[str, Any]],
    ) -> None:
        summary = summarize_results(case_results)
        self.garden.record_experiment_combo_result(
            experiment_id=self.config.experiment_id,
            combo_id=combo_id,
            score=summary["average_score"],
            result_text=(
                "Scripted Prompt Garden run completed. "
                f"pass_rate={summary['pass_rate']}; "
                f"case_set={self.case_set_id}"
            ),
            subject_score=None,
            subjective_notes=(
                "Generated by scripts/run_prompt_experiment.py."
            ),
            metrics={
                **summary,
                "case_set_id": self.case_set_id,
                "case_set_hash": self.case_set_hash,
                "model": self.config.model,
                "bot_variant": self.config.bot_variant,
                "fewshot_id": self.config.fewshot_id,
                "run_mode": self.config.run_mode,
                "selected_case_count": len(self.selected_cases),
                "artifact_coverage": 1.0,
                "runner_version": RUNNER_VERSION,
            },
            case_results=case_results,
        )

    def _artifacts_for_report(
        self,
        normalized_index: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []

        for combo_id in self.selected_combo_ids:
            for case in self.selected_cases:
                artifact = normalized_index.get((combo_id, case["id"]))
                if artifact is None:
                    continue
                execution = artifact.get("execution") or {}
                if execution.get("signature") != self.execution_signature:
                    continue
                artifacts.append(artifact)

        return sorted(
            artifacts,
            key=lambda artifact: (
                artifact.get("combo_id") or "",
                artifact.get("case_id") or "",
                artifact.get("created_at") or "",
            ),
        )

    def plan(self) -> dict[str, Any]:
        normalized_index = self._load_existing_normalized_index()
        targets = []
        skipped_existing = 0

        for combo_id in self.selected_combo_ids:
            for case in self.selected_cases:
                existing = normalized_index.get((combo_id, case["id"]))
                should_run = self._should_run_case(existing)
                if not should_run:
                    skipped_existing += 1
                targets.append({
                    "combo_id": combo_id,
                    "case_id": case["id"],
                    "should_run": should_run,
                    "has_existing_artifact": existing is not None,
                })

        return {
            "experiment_id": self.config.experiment_id,
            "case_set_id": self.case_set_id,
            "case_count": len(self.selected_cases),
            "case_filters_active": self.case_filters_active,
            "combo_count": len(self.selected_combo_ids),
            "target_count": sum(
                1 for row in targets
                if row["should_run"]
            ),
            "skipped_existing_count": skipped_existing,
            "raw_scope_dir": str(self.raw_scope_dir),
            "normalized_scope_dir": str(self.normalized_scope_dir),
            "report_scope_dir": str(self.report_scope_dir),
            "execution": {
                **self.execution_context,
                "signature": self.execution_signature,
            },
            "targets_preview": targets[:20],
        }

    def run(self) -> dict[str, Any]:
        plan = self.plan()
        if self.config.dry_run:
            return {
                **plan,
                "mode": "dry_run",
            }

        normalized_index = self._load_existing_normalized_index()
        executed_count = 0
        skipped_existing = 0
        updated_combo_ids: list[str] = []
        partial_combo_ids: list[str] = []
        filtered_case_combo_ids: list[str] = []

        for combo_id in self.selected_combo_ids:
            combo = self.garden.get_combo(combo_id)
            bot = self._make_bot(combo_id)

            for case in self.selected_cases:
                existing = normalized_index.get((combo_id, case["id"]))
                if not self._should_run_case(existing):
                    skipped_existing += 1
                    continue

                artifact = self._run_case(
                    bot=bot,
                    combo=combo,
                    case=case,
                )
                normalized_index[(combo_id, case["id"])] = artifact
                executed_count += 1

            case_results, missing_case_ids = self._combo_case_results_from_index(
                combo_id=combo_id,
                normalized_index=normalized_index,
            )
            if missing_case_ids:
                partial_combo_ids.append(combo_id)
                continue

            if self.case_filters_active:
                filtered_case_combo_ids.append(combo_id)
                continue

            self._update_combo_summary(
                combo_id=combo_id,
                case_results=case_results,
            )
            updated_combo_ids.append(combo_id)

        report_artifacts = self._artifacts_for_report(
            normalized_index=normalized_index,
        )
        report_path = write_summary_report(
            garden=self.garden,
            scope=self.scope,
            artifacts=report_artifacts,
            filters={
                "combo_ids": self.selected_combo_ids,
                "case_ids": self.selected_case_ids,
                "run_mode": self.config.run_mode,
                "case_set_id": self.case_set_id,
                "bot_variant": self.config.bot_variant,
                "use_rag": self.config.use_rag,
                "model": self.config.model,
            },
            context={
                "execution": {
                    **self.execution_context,
                    "signature": self.execution_signature,
                },
                "case_filters_active": self.case_filters_active,
                "updated_combo_ids": updated_combo_ids,
                "partial_combo_ids": partial_combo_ids,
                "filtered_case_combo_ids": filtered_case_combo_ids,
                "executed_count": executed_count,
                "skipped_existing_count": skipped_existing,
            },
        )

        return {
            **plan,
            "mode": "executed",
            "executed_count": executed_count,
            "skipped_existing_count": skipped_existing,
            "updated_combo_ids": updated_combo_ids,
            "partial_combo_ids": partial_combo_ids,
            "filtered_case_combo_ids": filtered_case_combo_ids,
            "report_path": str(report_path),
        }


def run_prompt_experiment(
    config: ExperimentRunConfig,
) -> dict[str, Any]:
    """Run or plan a Prompt Garden experiment from a structured config."""

    runner = PromptExperimentRunner(config)
    return runner.run()
