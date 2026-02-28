"""SQLite-backed LLM session persistence.

Stores the raw conversation turns (role + serialized parts) so the LLM
conversation context survives process restarts. Each turn is one row.

Only text, function_call, and function_response parts are persisted.
Binary data (images, audio, PDFs) is skipped to keep the DB small.
"""

import json
import logging
import sqlite3
import threading
import time

log = logging.getLogger("session-store")


class SessionStore:
    """Thread-safe SQLite store for LLM conversation turns."""

    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        log.info("Session store initialized at %s", db_path)

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    parts_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_chat
                ON session_turns (chat_id, id)
            """)
            conn.commit()
            conn.close()

    def save_turn(self, chat_id: int, role: str, parts_data: list[dict]):
        """Save a conversation turn (serialized parts) to the database."""
        now = time.time()
        parts_json = json.dumps(parts_data, ensure_ascii=False)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO session_turns (chat_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, parts_json, now),
            )
            conn.commit()
            conn.close()

    def load_turns(self, chat_id: int, limit: int = 20) -> list[dict]:
        """Load the most recent conversation turns for a chat.

        Returns list of {"role": str, "parts": list[dict]} dicts,
        ordered chronologically (oldest first).
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT role, parts_json FROM (
                       SELECT role, parts_json, id FROM session_turns
                       WHERE chat_id = ?
                       ORDER BY id DESC LIMIT ?
                   ) sub ORDER BY id ASC""",
                (chat_id, limit),
            ).fetchall()
            conn.close()

        turns = []
        for role, parts_json in rows:
            try:
                parts = json.loads(parts_json)
                turns.append({"role": role, "parts": parts})
            except json.JSONDecodeError:
                log.warning("Skipping corrupt session turn for chat %d", chat_id)
        return turns

    def clear(self, chat_id: int):
        """Clear all stored turns for a chat."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM session_turns WHERE chat_id = ?", (chat_id,))
            conn.commit()
            conn.close()
        log.info("Cleared session for chat %d", chat_id)

    def trim(self, chat_id: int, keep: int = 20):
        """Keep only the most recent N turns for a chat."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """DELETE FROM session_turns WHERE chat_id = ? AND id NOT IN (
                       SELECT id FROM session_turns WHERE chat_id = ?
                       ORDER BY id DESC LIMIT ?
                   )""",
                (chat_id, chat_id, keep),
            )
            conn.commit()
            conn.close()

    def prune(self, max_age_days: int = 30) -> int:
        """Delete all turns older than max_age_days across ALL chats.

        Returns the number of deleted rows.
        """
        cutoff = time.time() - max_age_days * 86400
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "DELETE FROM session_turns WHERE created_at < ?", (cutoff,)
            )
            deleted = cur.rowcount
            conn.commit()
            conn.close()
        if deleted:
            log.info("Pruned %d session turns older than %d days", deleted, max_age_days)
        return deleted

    def stats(self) -> dict:
        """Return summary statistics for diagnostics.

        Returns {"total_turns": int, "chat_count": int}.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT chat_id) FROM session_turns"
            ).fetchone()
            conn.close()
        return {"total_turns": row[0], "chat_count": row[1]}

    def replace_all(self, chat_id: int, turns_data: list[dict]):
        """Atomically replace all turns for a chat with the given data.

        Each item in turns_data should be {"role": str, "parts": list[dict]}.
        Used by context compaction to swap old turns with a summary.
        """
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM session_turns WHERE chat_id = ?", (chat_id,))
            for turn in turns_data:
                parts_json = json.dumps(turn["parts"], ensure_ascii=False)
                conn.execute(
                    "INSERT INTO session_turns (chat_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
                    (chat_id, turn["role"], parts_json, now),
                )
            conn.commit()
            conn.close()
        log.info("Replaced session for chat %d with %d compacted turns", chat_id, len(turns_data))


# ── Serialization helpers for google-genai types ──────────────────

def serialize_content(content) -> dict | None:
    """Serialize a google.genai.types.Content object to a JSON-safe dict.

    Skips turns that only contain binary data (images, audio).
    Returns None if nothing meaningful to persist.
    """
    parts_data = []
    for part in (content.parts or []):
        if hasattr(part, "text") and part.text is not None:
            parts_data.append({"type": "text", "text": part.text})
        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            parts_data.append({
                "type": "function_call",
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })
        elif hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            resp = fr.response if hasattr(fr, "response") else {}
            parts_data.append({
                "type": "function_response",
                "name": fr.name,
                "response": resp,
            })
        # Skip inline_data / blob parts (images, audio, PDFs)

    if not parts_data:
        return None

    return {"role": content.role, "parts": parts_data}


def deserialize_content(data: dict):
    """Deserialize a dict back to a google.genai.types.Content object."""
    from google.genai import types

    parts = []
    for p in data.get("parts", []):
        ptype = p.get("type", "")
        if ptype == "text":
            parts.append(types.Part(text=p["text"]))
        elif ptype == "function_call":
            parts.append(types.Part(
                function_call=types.FunctionCall(name=p["name"], args=p.get("args", {}))
            ))
        elif ptype == "function_response":
            parts.append(types.Part.from_function_response(
                name=p["name"], response=p.get("response", {})
            ))

    if not parts:
        return None

    return types.Content(role=data["role"], parts=parts)
