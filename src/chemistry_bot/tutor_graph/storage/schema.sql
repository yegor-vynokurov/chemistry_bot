-- SQLite schema for the experimental LangGraph chemistry tutor.
--
-- PR 2 adds storage only. Later PRs will expose these operations as LangChain
-- tools and LangGraph nodes. The schema already separates raw chat history from
-- session state and decision logs, so future learning features do not have to
-- rewrite the foundation.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student', 'tutor', 'system', 'tool')),
    text TEXT NOT NULL,
    response_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session_created
ON turns(session_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_turns_student_created
ON turns(student_id, created_at, id);

CREATE TABLE IF NOT EXISTS prompt_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tutor_turn_id INTEGER NOT NULL,
    sequence_index INTEGER NOT NULL CHECK (sequence_index >= 0),
    role TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed', 'no_model_invocation')),
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    assignment_kind TEXT NOT NULL CHECK (assignment_kind IN ('baseline', 'experiment', 'not_applicable')),
    experiment_id TEXT,
    route TEXT,
    error_code TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (tutor_turn_id) REFERENCES turns(id) ON DELETE CASCADE,
    UNIQUE (tutor_turn_id, sequence_index),
    CHECK (
        (status = 'no_model_invocation'
            AND role = 'no_model_invocation'
            AND provider IS NULL AND model IS NULL AND attempt = 0
            AND assignment_kind = 'not_applicable' AND experiment_id IS NULL
            AND reason IS NOT NULL AND length(trim(reason)) > 0
            AND error_code IS NULL)
        OR
        (status IN ('succeeded', 'failed')
            AND role <> 'no_model_invocation'
            AND provider IS NOT NULL AND length(trim(provider)) > 0
            AND model IS NOT NULL AND length(trim(model)) > 0 AND attempt >= 1
            AND reason IS NULL
            AND ((status = 'succeeded' AND error_code IS NULL)
                OR (status = 'failed' AND error_code IS NOT NULL
                    AND length(trim(error_code)) > 0))
            AND ((assignment_kind = 'baseline' AND experiment_id IS NULL)
                OR (assignment_kind = 'experiment' AND experiment_id IS NOT NULL
                    AND length(trim(experiment_id)) > 0)))
    )
);

CREATE INDEX IF NOT EXISTS idx_prompt_invocations_turn_sequence
ON prompt_invocations(tutor_turn_id, sequence_index);

CREATE TABLE IF NOT EXISTS prompt_invocation_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invocation_id INTEGER NOT NULL,
    component_index INTEGER NOT NULL CHECK (component_index >= 0),
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    component_role TEXT NOT NULL,
    FOREIGN KEY (invocation_id) REFERENCES prompt_invocations(id) ON DELETE CASCADE,
    UNIQUE (invocation_id, component_index)
);

CREATE INDEX IF NOT EXISTS idx_prompt_invocation_prompts_identity
ON prompt_invocation_prompts(prompt_id, prompt_version);

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    turn_id INTEGER,
    node_name TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    decision_json TEXT,
    confidence REAL,
    reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_log_session_created
ON decision_log(session_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_decision_log_session_type_created
ON decision_log(session_id, decision_type, created_at, id);
