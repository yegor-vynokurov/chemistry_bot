"""Small SQLite storage layer for the experimental tutor graph.

PR 2 intentionally keeps this file boring and explicit. The graph can change,
the RAG bot can change, but durable tutor data should live behind this small
stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable




class _ClosingSQLiteConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path("data") / "tutor_graph" / "tutor.db"


def utc_now_iso() -> str:
    """Return an ISO timestamp that sorts lexicographically."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default




@dataclass(slots=True)
class StoredSession:
    """One persisted tutor conversation session."""

    id: str
    student_id: str
    title: str | None
    status: str
    started_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredSession":
        return cls(
            id=str(row["id"]),
            student_id=str(row["student_id"]),
            title=str(row["title"]) if row["title"] is not None else None,
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
        )


@dataclass(slots=True)
class StoredTurn:
    """One persisted conversation turn."""

    id: int
    session_id: str
    student_id: str
    role: str
    text: str
    response_json: dict[str, Any] | None
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredTurn":
        response = _json_loads(row["response_json"], None)
        metadata = _json_loads(row["metadata_json"], {})
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            student_id=str(row["student_id"]),
            role=str(row["role"]),
            text=str(row["text"]),
            response_json=response if isinstance(response, dict) else None,
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=str(row["created_at"]),
        )


@dataclass(slots=True)
class StoredDecision:
    """One persisted classifier/router decision."""

    id: int
    session_id: str
    student_id: str
    turn_id: int | None
    node_name: str
    decision_type: str
    decision: dict[str, Any]
    confidence: float | None
    reason: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredDecision":
        decision = _json_loads(row["decision_json"], {})
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            student_id=str(row["student_id"]),
            turn_id=int(row["turn_id"]) if row["turn_id"] is not None else None,
            node_name=str(row["node_name"]),
            decision_type=str(row["decision_type"]),
            decision=decision if isinstance(decision, dict) else {},
            confidence=(
                float(row["confidence"]) if row["confidence"] is not None else None
            ),
            reason=str(row["reason"]) if row["reason"] is not None else None,
            created_at=str(row["created_at"]),
        )


@dataclass(slots=True)
class TutorStoreStatus:
    """Compact storage status for CLI/debug output."""

    db_path: str
    students: int
    sessions: int
    turns: int
    decision_log_entries: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "students": self.students,
            "sessions": self.sessions,
            "turns": self.turns,
            "decision_log_entries": self.decision_log_entries,
        }


class SQLiteTutorStore:
    """Durable storage for the standalone tutor graph.

    This class is intentionally not a LangChain tool yet. PR 3 can wrap these
    methods as tools without changing the database schema.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), factory=_ClosingSQLiteConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)
            conn.commit()

    def ensure_student(self, student_id: str, display_name: str | None = None) -> None:
        if not student_id.strip():
            raise ValueError("student_id must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO students (id, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = COALESCE(excluded.display_name, students.display_name),
                    updated_at = excluded.updated_at
                """,
                (student_id, display_name, now, now),
            )
            conn.commit()

    def ensure_session(
        self,
        session_id: str,
        student_id: str,
        title: str | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        self.ensure_student(student_id)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, student_id, title, status, started_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    student_id = excluded.student_id,
                    title = COALESCE(excluded.title, sessions.title),
                    updated_at = excluded.updated_at
                """,
                (session_id, student_id, title, now, now),
            )
            conn.execute(
                """
                INSERT INTO session_state (session_id, state_json, updated_at)
                VALUES (?, '{}', ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now),
            )
            conn.commit()


    def get_recent_sessions(
        self,
        *,
        student_id: str,
        limit: int = 100,
        status: str | None = None,
    ) -> list[StoredSession]:
        """Return a student's sessions, newest activity first."""
        if limit < 1:
            return []
        where = ["student_id = ?"]
        params: list[Any] = [student_id]
        if status is not None:
            where.append("status = ?")
            params.append(status)
        params.append(limit)
        query = f"""
            SELECT *
            FROM sessions
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredSession.from_row(row) for row in rows]

    def get_session(self, session_id: str) -> StoredSession | None:
        """Return one session or ``None`` when it does not exist."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return StoredSession.from_row(row) if row is not None else None

    def count_session_turns(self, session_id: str) -> int:
        """Return the number of stored turns in a session."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    def save_turn(
        self,
        *,
        student_id: str,
        session_id: str,
        role: str,
        text: str,
        response_json: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist one turn and return its integer id."""
        if role not in {"student", "tutor", "system", "tool"}:
            raise ValueError(f"Unsupported role: {role!r}")
        if not text.strip():
            raise ValueError("turn text must not be empty")

        self.ensure_session(session_id=session_id, student_id=student_id)
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO turns (
                    session_id,
                    student_id,
                    role,
                    text,
                    response_json,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    role,
                    text,
                    _json_dumps(response_json) if response_json is not None else None,
                    _json_dumps(metadata or {}),
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def save_exchange(
        self,
        *,
        student_id: str,
        session_id: str,
        student_text: str,
        tutor_text: str,
        student_metadata: dict[str, Any] | None = None,
        tutor_metadata: dict[str, Any] | None = None,
        response_json: dict[str, Any] | None = None,
        prompt_invocations: Iterable[dict[str, Any]] = (),
        no_model_reason: str = "deterministic_or_model_roles_disabled",
    ) -> dict[str, int]:
        """Atomically persist a learner/tutor exchange and its invocation provenance."""
        if not student_text.strip() or not tutor_text.strip():
            raise ValueError("Student and tutor turn text must not be empty")
        records = [dict(record) for record in prompt_invocations]
        if not records:
            from chemistry_bot.tutor_graph.prompt_provenance import no_model_invocation

            records = [no_model_invocation(no_model_reason)]
        self._validate_prompt_invocations(records)
        provenance_state = (
            "no_model_invocation"
            if len(records) == 1 and records[0]["status"] == "no_model_invocation"
            else "recorded"
        )

        self.ensure_session(session_id=session_id, student_id=student_id)
        now = utc_now_iso()
        student_meta = dict(student_metadata or {})
        tutor_meta = {
            **(tutor_metadata or {}),
            "prompt_provenance_state": provenance_state,
        }
        with self.connect() as conn:
            try:
                student_cursor = conn.execute(
                    """INSERT INTO turns (
                        session_id, student_id, role, text, response_json, metadata_json, created_at
                    ) VALUES (?, ?, 'student', ?, NULL, ?, ?)""",
                    (session_id, student_id, student_text, _json_dumps(student_meta), now),
                )
                student_turn_id = int(student_cursor.lastrowid)
                tutor_meta["reply_to_turn_id"] = student_turn_id
                tutor_cursor = conn.execute(
                    """INSERT INTO turns (
                        session_id, student_id, role, text, response_json, metadata_json, created_at
                    ) VALUES (?, ?, 'tutor', ?, ?, ?, ?)""",
                    (
                        session_id,
                        student_id,
                        tutor_text,
                        _json_dumps(response_json) if response_json is not None else None,
                        _json_dumps(tutor_meta),
                        now,
                    ),
                )
                tutor_turn_id = int(tutor_cursor.lastrowid)
                for sequence_index, record in enumerate(records):
                    cursor = conn.execute(
                        """INSERT INTO prompt_invocations (
                            tutor_turn_id, sequence_index, role, provider, model, status,
                            attempt, assignment_kind, experiment_id, route, error_code,
                            reason, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            tutor_turn_id,
                            sequence_index,
                            record["role"],
                            record.get("provider"),
                            record.get("model"),
                            record["status"],
                            record["attempt"],
                            record["assignment_kind"],
                            record.get("experiment_id"),
                            record.get("route"),
                            record.get("error_code"),
                            record.get("reason"),
                            now,
                        ),
                    )
                    invocation_id = int(cursor.lastrowid)
                    for component_index, prompt in enumerate(record.get("prompts", [])):
                        conn.execute(
                            """INSERT INTO prompt_invocation_prompts (
                                invocation_id, component_index, prompt_id, prompt_version,
                                prompt_sha256, component_role
                            ) VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                invocation_id,
                                component_index,
                                prompt["prompt_id"],
                                prompt["version"],
                                prompt["sha256"],
                                prompt["role"],
                            ),
                        )
                conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"student": student_turn_id, "tutor": tutor_turn_id}

    @staticmethod
    def _validate_prompt_invocations(records: list[dict[str, Any]]) -> None:
        """Reject incomplete or contradictory call records before persistence."""
        no_model_records = [record for record in records if record.get("status") == "no_model_invocation"]
        if no_model_records and len(records) != 1:
            raise ValueError("A no-model marker cannot be mixed with model invocation records")
        for record in records:
            status = record.get("status")
            if status == "no_model_invocation":
                if record.get("role") != "no_model_invocation" or not str(record.get("reason", "")).strip():
                    raise ValueError("A no-model invocation requires its explicit role and reason")
                if record.get("prompts") or record.get("assignment_kind") != "not_applicable":
                    raise ValueError("A no-model marker cannot contain prompt or assignment data")
                continue
            if status not in {"succeeded", "failed"}:
                raise ValueError(f"Unsupported prompt invocation status: {status!r}")
            error_code = record.get("error_code")
            if status == "failed" and not str(error_code or "").strip():
                raise ValueError("A failed model invocation requires a safe error code")
            if status == "succeeded" and error_code is not None:
                raise ValueError("A successful model invocation cannot have an error code")
            if not all(str(record.get(key, "")).strip() for key in ("role", "provider", "model")):
                raise ValueError("A model invocation requires role, provider, and model")
            if int(record.get("attempt", 0)) < 1:
                raise ValueError("A model invocation attempt must be at least 1")
            assignment_kind = record.get("assignment_kind")
            experiment_id = record.get("experiment_id")
            if assignment_kind == "baseline" and experiment_id is not None:
                raise ValueError("A baseline invocation cannot have an experiment ID")
            if assignment_kind == "experiment" and not str(experiment_id or "").strip():
                raise ValueError("An experiment invocation requires its experiment ID")
            if assignment_kind not in {"baseline", "experiment"}:
                raise ValueError("A model invocation requires baseline or experiment assignment")
            prompts = record.get("prompts")
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("A model invocation must reference at least one prompt revision")
            for prompt in prompts:
                if not isinstance(prompt, dict) or not all(
                    str(prompt.get(key, "")).strip()
                    for key in ("prompt_id", "version", "sha256", "role")
                ):
                    raise ValueError("Each prompt reference requires ID, version, hash, and role")

    def get_prompt_invocations(self, tutor_turn_id: int) -> dict[str, Any]:
        """Read prompt provenance linked to one tutor turn; distinguish legacy unknown."""
        with self.connect() as conn:
            turn = conn.execute(
                "SELECT role FROM turns WHERE id = ?",
                (tutor_turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"Tutor turn {tutor_turn_id} does not exist.")
            if str(turn["role"]) != "tutor":
                raise ValueError(f"Turn {tutor_turn_id} is not a tutor turn.")
            rows = conn.execute(
                """SELECT * FROM prompt_invocations
                   WHERE tutor_turn_id = ? ORDER BY sequence_index""",
                (tutor_turn_id,),
            ).fetchall()
            invocations: list[dict[str, Any]] = []
            for row in rows:
                prompts = conn.execute(
                    """SELECT prompt_id, prompt_version AS version, prompt_sha256 AS sha256,
                              component_role AS role
                       FROM prompt_invocation_prompts
                       WHERE invocation_id = ? ORDER BY component_index""",
                    (int(row["id"]),),
                ).fetchall()
                invocations.append(
                    {
                        "role": str(row["role"]),
                        "provider": row["provider"],
                        "model": row["model"],
                        "status": str(row["status"]),
                        "attempt": int(row["attempt"]),
                        "assignment_kind": str(row["assignment_kind"]),
                        "experiment_id": row["experiment_id"],
                        "route": row["route"],
                        "error_code": row["error_code"],
                        "reason": row["reason"],
                        "created_at": str(row["created_at"]),
                        "prompts": [dict(prompt) for prompt in prompts],
                    }
                )
        if not invocations:
            state = "unknown"
        elif len(invocations) == 1 and invocations[0]["status"] == "no_model_invocation":
            state = "no_model_invocation"
        else:
            state = "recorded"
        return {"tutor_turn_id": tutor_turn_id, "state": state, "invocations": invocations}

    def get_recent_turns(
        self,
        *,
        student_id: str | None = None,
        session_id: str | None = None,
        limit: int = 10,
        before_turn_id: int | None = None,
        roles: list[str] | None = None,
    ) -> list[StoredTurn]:
        """Load recent turns for a student or a session.

        Results are returned in chronological order, even though the SQL query
        selects the newest rows first. ``before_turn_id`` is exclusive and is
        useful when the current student turn has already been saved but should
        not be duplicated in the model context.
        """
        if limit < 1:
            return []
        if session_id is None and student_id is None:
            raise ValueError("Either student_id or session_id must be provided")

        where: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if student_id is not None:
            where.append("student_id = ?")
            params.append(student_id)
        if before_turn_id is not None:
            where.append("id < ?")
            params.append(before_turn_id)
        if roles:
            unsupported = sorted(set(roles) - {"student", "tutor", "system", "tool"})
            if unsupported:
                raise ValueError(f"Unsupported roles: {unsupported!r}")
            placeholders = ", ".join("?" for _ in roles)
            where.append(f"role IN ({placeholders})")
            params.extend(roles)

        params.append(limit)
        query = f"""
            SELECT *
            FROM turns
            WHERE {' AND '.join(where)}
            ORDER BY id DESC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredTurn.from_row(row) for row in reversed(rows)]

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {}
        state = _json_loads(row["state_json"], {})
        return state if isinstance(state, dict) else {}

    def update_session_state(
        self,
        session_id: str,
        patch: dict[str, Any],
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        current = {} if replace else self.get_session_state(session_id)
        current.update(patch)
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO session_state (session_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, _json_dumps(current), now),
            )
            conn.commit()
        return current

    def save_decision(
        self,
        *,
        student_id: str,
        session_id: str,
        node_name: str,
        decision_type: str,
        decision: dict[str, Any] | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        turn_id: int | None = None,
    ) -> int:
        self.ensure_session(session_id=session_id, student_id=student_id)
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO decision_log (
                    session_id,
                    student_id,
                    turn_id,
                    node_name,
                    decision_type,
                    decision_json,
                    confidence,
                    reason,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    turn_id,
                    node_name,
                    decision_type,
                    _json_dumps(decision or {}),
                    confidence,
                    reason,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_recent_decisions(
        self,
        *,
        student_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        before_decision_id: int | None = None,
        decision_types: list[str] | None = None,
    ) -> list[StoredDecision]:
        """Load recent decision-log entries in chronological order."""
        if limit < 1:
            return []
        if session_id is None and student_id is None:
            raise ValueError("Either student_id or session_id must be provided")

        where: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if student_id is not None:
            where.append("student_id = ?")
            params.append(student_id)
        if before_decision_id is not None:
            where.append("id < ?")
            params.append(before_decision_id)
        if decision_types:
            clean_types = [item.strip() for item in decision_types if item.strip()]
            if clean_types:
                placeholders = ", ".join("?" for _ in clean_types)
                where.append(f"decision_type IN ({placeholders})")
                params.extend(clean_types)

        params.append(limit)
        query = f"""
            SELECT *
            FROM decision_log
            WHERE {' AND '.join(where)}
            ORDER BY id DESC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredDecision.from_row(row) for row in reversed(rows)]

    def count_table(self, table_name: str) -> int:
        allowed = {"students", "sessions", "turns", "decision_log"}
        if table_name not in allowed:
            raise ValueError(f"Unsupported table name: {table_name!r}")
        with self.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
        return int(row["count"])

    def status(self) -> TutorStoreStatus:
        return TutorStoreStatus(
            db_path=str(self.db_path),
            students=self.count_table("students"),
            sessions=self.count_table("sessions"),
            turns=self.count_table("turns"),
            decision_log_entries=self.count_table("decision_log"),
        )

    def iter_recent_turns_text(
        self,
        *,
        session_id: str,
        limit: int = 10,
        before_turn_id: int | None = None,
    ) -> Iterable[str]:
        for turn in self.get_recent_turns(
            session_id=session_id,
            limit=limit,
            before_turn_id=before_turn_id,
        ):
            yield f"[{turn.created_at}] {turn.role}: {turn.text}"
