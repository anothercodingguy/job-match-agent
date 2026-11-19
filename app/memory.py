"""SQLite storage for what the agent remembers about each session.

Tables:
  sessions     one row per visitor session, with the structured profile as JSON
  messages     the user/assistant turns of the chat, in order; an assistant
               row also keeps the jobs it showed (id, title, company) as JSON
  preferences  key/value facts the agent saved (remote only, location, ...)
  feedback     jobs the user saved or dismissed; dismissed jobs are not shown again

The raw resume text is never stored. Only the structured profile is.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import PREFERENCE_KEYS, Preferences, Profile

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    profile_json TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    jobs_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
CREATE TABLE IF NOT EXISTS preferences (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);
CREATE TABLE IF NOT EXISTS feedback (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id     TEXT NOT NULL,
    verdict    TEXT NOT NULL CHECK (verdict IN ('saved', 'dismissed')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, job_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory:
    def __init__(self, db_path: str = ":memory:"):
        self._shared = None
        if db_path == ":memory:":
            # One shared connection, otherwise every connect() would see an empty database.
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # A database made before jobs_json existed: add the column in place.
            if "jobs_json" not in {row[1] for row in conn.execute("PRAGMA table_info(messages)")}:
                conn.execute("ALTER TABLE messages ADD COLUMN jobs_json TEXT")

    @contextmanager
    def _conn(self):
        conn = self._shared or sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            if self._shared is None:
                conn.close()

    # sessions and profile
    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (session_id, _now()))
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None

    def save_profile(self, session_id: str, profile: Profile) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET profile_json = ? WHERE id = ?", (profile.model_dump_json(), session_id)
            )

    def get_profile(self, session_id: str) -> Profile | None:
        with self._conn() as conn:
            row = conn.execute("SELECT profile_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return Profile.model_validate_json(row[0]) if row and row[0] else None

    # chat history
    def add_message(self, session_id: str, role: str, content: str, jobs: list[dict] | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, jobs_json) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, _now(), json.dumps(jobs) if jobs else None),
            )

    def last_shown_jobs(self, session_id: str) -> list[dict]:
        """The jobs from the most recent assistant message that showed any."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT jobs_json FROM messages WHERE session_id = ? AND jobs_json IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row[0]) if row else []

    def recent_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    # preferences
    def set_preference(self, session_id: str, key: str, value) -> None:
        if key not in PREFERENCE_KEYS:
            raise ValueError(f"unknown preference '{key}'. Allowed: {', '.join(PREFERENCE_KEYS)}")
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO preferences (session_id, key, value, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT (session_id, key) DO UPDATE SET value = excluded.value,
                                                               updated_at = excluded.updated_at""",
                (session_id, key, json.dumps(value), _now()),
            )

    def get_preferences(self, session_id: str) -> Preferences:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE session_id = ?", (session_id,)
            ).fetchall()
        return Preferences.model_validate({key: json.loads(value) for key, value in rows})

    # feedback
    def set_feedback(self, session_id: str, job_id: str, verdict: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback (session_id, job_id, verdict, created_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT (session_id, job_id) DO UPDATE SET verdict = excluded.verdict,
                                                                  created_at = excluded.created_at""",
                (session_id, job_id, verdict, _now()),
            )

    def job_ids(self, session_id: str, verdict: str) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT job_id FROM feedback WHERE session_id = ? AND verdict = ?", (session_id, verdict)
            ).fetchall()
        return {row[0] for row in rows}
