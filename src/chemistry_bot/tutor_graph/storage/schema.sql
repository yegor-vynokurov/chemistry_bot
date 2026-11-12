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
