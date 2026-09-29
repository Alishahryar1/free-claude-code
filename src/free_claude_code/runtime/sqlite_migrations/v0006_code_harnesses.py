"""Allow fixed Claude harness sessions and native permission modes."""

import sqlite3

_TABLES = (
    (
        "code_sessions",
        "id,cwd,model,reasoning_effort,harness,title,title_search,cwd_search,auto_title,native_thread_id,native_may_have_input,revision,status,error,created_at,updated_at,mode,native_permission_defaults,context_used_tokens",
        """CREATE TABLE code_sessions(
    native_permission_defaults TEXT CHECK(native_permission_defaults IS NULL OR (json_valid(native_permission_defaults) AND json_type(native_permission_defaults) = 'object')),
    context_used_tokens INTEGER CHECK(context_used_tokens IS NULL OR context_used_tokens >= 0),
    mode TEXT NOT NULL DEFAULT 'config' CHECK(mode IN ('config','ask','auto_review','full_access','default','acceptEdits','plan','dontAsk','auto','bypassPermissions')),
    id TEXT PRIMARY KEY NOT NULL, cwd TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    harness TEXT NOT NULL CHECK(harness IN ('codex','claude')), title TEXT NOT NULL, title_search TEXT NOT NULL,
    cwd_search TEXT NOT NULL, auto_title INTEGER NOT NULL CHECK(auto_title IN (0,1)), native_thread_id TEXT,
    native_may_have_input INTEGER NOT NULL CHECK(native_may_have_input IN (0,1)),
    revision INTEGER NOT NULL CHECK(revision > 0), status TEXT NOT NULL CHECK(status IN ('ready','deleting','delete_uncertain')),
    error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
)""",
    ),
    (
        "code_runs",
        "session_id,id,ordinal,text,model,reasoning_effort,status,submission_started,native_turn_id,stop_requested,error,error_details,created_at,finished_at,mode",
        """CREATE TABLE code_runs(
    mode TEXT NOT NULL DEFAULT 'config' CHECK(mode IN ('config','ask','auto_review','full_access','default','acceptEdits','plan','dontAsk','auto','bypassPermissions')),
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal > 0), text TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
    status TEXT NOT NULL CHECK(status IN ('preparing','running','stopping','completed','interrupted','failed')),
    submission_started INTEGER NOT NULL CHECK(submission_started IN (0,1)), native_turn_id TEXT,
    stop_requested INTEGER NOT NULL CHECK(stop_requested IN (0,1)), error TEXT, error_details TEXT NOT NULL,
    created_at INTEGER NOT NULL, finished_at INTEGER,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,ordinal), UNIQUE(session_id,native_turn_id)
)""",
    ),
    (
        "code_items",
        "session_id,id,run_id,sequence,native_turn_id,native_item_id,kind,title,text,detail,complete,raw",
        """CREATE TABLE code_items(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence > 0), native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, detail TEXT NOT NULL,
    complete INTEGER NOT NULL CHECK(complete IN (0,1)), raw TEXT NOT NULL,
    PRIMARY KEY(session_id,id), FOREIGN KEY(session_id,run_id) REFERENCES code_runs(session_id,id) ON DELETE CASCADE,
    UNIQUE(session_id,sequence), UNIQUE(session_id,native_turn_id,native_item_id)
)""",
    ),
    (
        "code_prompts",
        "session_id,id,generation,request_id,native_turn_id,native_item_id,kind,form,raw,status,response_id,error",
        """CREATE TABLE code_prompts(
    session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, id TEXT NOT NULL,
    generation TEXT NOT NULL, request_id TEXT NOT NULL, native_turn_id TEXT, native_item_id TEXT,
    kind TEXT NOT NULL, form TEXT NOT NULL, raw TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','answering','resolved','expired')), response_id TEXT, error TEXT,
    PRIMARY KEY(session_id,id), UNIQUE(session_id,generation,request_id), UNIQUE(session_id,response_id),
    FOREIGN KEY(session_id,id) REFERENCES code_items(session_id,id) ON DELETE CASCADE
)""",
    ),
)

_INDEXES = (
    "CREATE INDEX code_sessions_recent ON code_sessions(updated_at DESC, id DESC)",
    "CREATE UNIQUE INDEX code_one_active_run ON code_runs(session_id)\n    WHERE status IN ('preparing','running','stopping')",
    "CREATE INDEX code_items_run ON code_items(session_id,run_id,sequence)",
    "CREATE INDEX code_prompts_active ON code_prompts(session_id, id) WHERE status IN ('pending', 'answering')",
)


def upgrade(connection: sqlite3.Connection) -> None:
    for name, columns, _ in _TABLES:
        connection.execute(
            f"CREATE TEMP TABLE saved_{name} AS SELECT {columns} FROM {name}"
        )
    for name, _, _ in reversed(_TABLES):
        connection.execute(f"DROP TABLE {name}")
    for name, columns, schema in _TABLES:
        connection.execute(schema)
        connection.execute(
            f"INSERT INTO {name} ({columns}) SELECT {columns} FROM saved_{name}"
        )
        restored = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        previous = connection.execute(f"SELECT COUNT(*) FROM saved_{name}").fetchone()[
            0
        ]
        if restored != previous:
            raise sqlite3.DatabaseError(
                "Code history could not be migrated completely."
            )
        connection.execute(f"DROP TABLE saved_{name}")
    for statement in _INDEXES:
        connection.execute(statement)
