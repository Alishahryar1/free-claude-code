-- Frozen pre-version-1 Code schema, independent of production migrations.

CREATE TABLE IF NOT EXISTS code_sessions(
    id TEXT PRIMARY KEY NOT NULL, cwd TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    harness TEXT NOT NULL CHECK(harness = 'codex'), title TEXT NOT NULL, title_search TEXT NOT NULL,
    cwd_search TEXT NOT NULL, auto_title INTEGER NOT NULL CHECK(auto_title IN (0,1)), native_thread_id TEXT,
    native_may_have_input INTEGER NOT NULL CHECK(native_may_have_input IN (0,1)),
    revision INTEGER NOT NULL CHECK(revision > 0), status TEXT NOT NULL CHECK(status IN ('ready','deleting','delete_uncertain')),
    error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS code_sessions_recent ON code_sessions(updated_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS code_runs(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal > 0), text TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    status TEXT NOT NULL CHECK(status IN ('preparing','running','stopping','completed','interrupted','failed')),
    submission_started INTEGER NOT NULL CHECK(submission_started IN (0,1)), native_turn_id TEXT,
    stop_requested INTEGER NOT NULL CHECK(stop_requested IN (0,1)), error TEXT, error_details TEXT NOT NULL,
    created_at INTEGER NOT NULL, finished_at INTEGER,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,ordinal), UNIQUE(session_id,native_turn_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS code_one_active_run ON code_runs(session_id)
    WHERE status IN ('preparing','running','stopping');
CREATE TABLE IF NOT EXISTS code_items(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence > 0), native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, detail TEXT NOT NULL,
    complete INTEGER NOT NULL CHECK(complete IN (0,1)), raw TEXT NOT NULL,
    PRIMARY KEY(session_id,id), FOREIGN KEY(session_id,run_id) REFERENCES code_runs(session_id,id) ON DELETE CASCADE,
    UNIQUE(session_id,sequence), UNIQUE(session_id,native_turn_id,native_item_id)
);
CREATE INDEX IF NOT EXISTS code_items_run ON code_items(session_id,run_id,sequence);
CREATE TABLE IF NOT EXISTS code_deleted(id TEXT PRIMARY KEY NOT NULL);
CREATE TABLE IF NOT EXISTS code_prompts(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    generation TEXT NOT NULL, request_id TEXT NOT NULL, native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, form TEXT NOT NULL, raw TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','answering','resolved','expired')), response_id TEXT, error TEXT,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,generation,request_id), UNIQUE(session_id,response_id)
);