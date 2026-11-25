"""Conversation history and answer feedback.

Stored apart from the document registry: the registry describes the corpus and
can be rebuilt from Sources/, whereas conversations and feedback are user data
that must survive a re-index.

Every answer is stored as the full JSON response the UI received, so a thumbs-down
can later be examined together with the exact citations, scores and warnings
that produced it. That makes the feedback table the seed of the Part 4
evaluation set.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question         TEXT NOT NULL,
    category         TEXT,
    response_json    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    message_id  TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    rating      TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment     TEXT,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 60 else title[:57].rstrip() + "..."


class HistoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        question: str,
        response: dict[str, Any],
        category: str | None,
        conversation_id: str | None,
    ) -> tuple[str, str]:
        """Store one question/answer turn. Returns (conversation_id, message_id)."""
        now = _now()
        message_id = uuid.uuid4().hex
        with self._connect() as conn:
            exists = conversation_id and conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                # An unknown id (e.g. a conversation deleted in another tab)
                # starts a fresh conversation rather than failing the question.
                conversation_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO conversations(id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (conversation_id, _title_from(question), now, now),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
            conn.execute(
                "INSERT INTO messages(id, conversation_id, question, category, "
                "response_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, question, category,
                 json.dumps(response, default=str), now),
            )
        return conversation_id, message_id

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, "
                "COUNT(m.id) AS n_messages FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id = c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not conv:
                return None
            rows = conn.execute(
                "SELECT m.*, f.rating FROM messages m "
                "LEFT JOIN feedback f ON f.message_id = m.id "
                "WHERE m.conversation_id = ? ORDER BY m.created_at, m.rowid",
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            response = json.loads(row["response_json"])
            response["message_id"] = row["id"]
            response["conversation_id"] = row["conversation_id"]
            response["feedback"] = row["rating"]
            messages.append(response)
        return {**dict(conv), "messages": messages}

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def set_feedback(self, message_id: str, rating: str, comment: str | None) -> bool:
        with self._connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM messages WHERE id = ?", (message_id,)
            ).fetchone():
                return False
            conn.execute(
                "INSERT INTO feedback(message_id, rating, comment, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(message_id) DO UPDATE SET "
                "rating = excluded.rating, comment = excluded.comment, "
                "created_at = excluded.created_at",
                (message_id, rating, comment, _now()),
            )
        return True

    def rated_messages(self) -> list[dict[str, Any]]:
        """Every answer a user rated, with the stored response, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT m.id, m.question, m.category, m.response_json, m.created_at, "
                "f.rating, f.comment FROM feedback f JOIN messages m ON m.id = f.message_id "
                "ORDER BY m.created_at"
            ).fetchall()
        return [
            {**{k: row[k] for k in ("id", "question", "category", "created_at",
                                    "rating", "comment")},
             "response": json.loads(row["response_json"])}
            for row in rows
        ]

    def feedback_summary(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) n FROM feedback GROUP BY rating"
            ).fetchall()
        summary = {"up": 0, "down": 0}
        summary.update({r["rating"]: r["n"] for r in rows})
        return summary
