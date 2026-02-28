"""SQLite-backed persistent memory store.

Dual-layer memory system:
- Facts: Key-value pairs about users/topics that persist across conversations.
  E.g., "samim_language" = "prefers German summaries"
- Notes: Timestamped daily observations for event recall.
  E.g., "2026-02-10: Deployed knarr-skills to Hetzner VPS"

The LLM is instructed to proactively save important facts it learns about
users (preferences, expertise, context) and can recall them later. Memory
context is injected into the system prompt so the LLM always has access.
"""

import sqlite3
import time
import datetime
import threading
import logging
from typing import Optional

log = logging.getLogger("memory-store")

MAX_FACTS_IN_CONTEXT = 50
MAX_NOTES_IN_CONTEXT = 20
MAX_CONTEXT_CHARS = 10000


class MemoryStore:
    """Thread-safe SQLite persistent memory."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        log.info("Memory store initialized at %s", db_path)

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    saved_by TEXT DEFAULT 'system',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_chat_key
                ON facts (chat_id, key)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notes_chat_date
                ON notes (chat_id, date)
            """)
            conn.commit()
            conn.close()

    def save_fact(self, chat_id: int, key: str, value: str, saved_by: str = "system") -> int:
        """Save or update a fact (upserts by chat_id + key). Returns the fact ID."""
        now = time.time()
        key = key.strip().lower()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO facts (chat_id, key, value, saved_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, key) DO UPDATE SET
                    value = excluded.value,
                    saved_by = excluded.saved_by,
                    updated_at = excluded.updated_at
            """, (chat_id, key, value, saved_by, now, now))
            fact_id = conn.execute(
                "SELECT id FROM facts WHERE chat_id = ? AND key = ?", (chat_id, key)
            ).fetchone()[0]
            conn.commit()
            conn.close()
        log.info("Saved fact [%s] = %s (chat %d)", key, value[:60], chat_id)
        return fact_id

    def get_facts(self, chat_id: int, limit: int = 50) -> list[dict]:
        """Get all facts for a chat."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT id, key, value, saved_by, updated_at
                   FROM facts WHERE chat_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (chat_id, limit),
            ).fetchall()
            conn.close()
        return [
            {"id": r[0], "key": r[1], "value": r[2], "saved_by": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def search_facts(self, chat_id: int, query: str, limit: int = 20) -> list[dict]:
        """Search facts by keyword (matches key or value, case-insensitive)."""
        pattern = f"%{query.strip()}%"
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT id, key, value, saved_by, updated_at
                   FROM facts WHERE chat_id = ?
                   AND (key LIKE ? OR value LIKE ?)
                   ORDER BY updated_at DESC LIMIT ?""",
                (chat_id, pattern, pattern, limit),
            ).fetchall()
            conn.close()
        return [
            {"id": r[0], "key": r[1], "value": r[2], "saved_by": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def delete_fact(self, fact_id: int, chat_id: int) -> bool:
        """Delete a fact by ID (only if it belongs to the given chat)."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "DELETE FROM facts WHERE id = ? AND chat_id = ?", (fact_id, chat_id)
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
        return deleted

    def clear_all(self, chat_id: int) -> int:
        """Delete ALL facts and notes for a chat. Returns number of items deleted."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            r1 = conn.execute("DELETE FROM facts WHERE chat_id = ?", (chat_id,))
            r2 = conn.execute("DELETE FROM notes WHERE chat_id = ?", (chat_id,))
            total = r1.rowcount + r2.rowcount
            conn.commit()
            conn.close()
        log.info("Cleared all memory for chat %d (%d items)", chat_id, total)
        return total

    def save_note(self, chat_id: int, text: str) -> int:
        """Save a daily note. Returns the note ID."""
        now = time.time()
        date_str = datetime.date.today().isoformat()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "INSERT INTO notes (chat_id, text, date, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, text, date_str, now),
            )
            note_id = cursor.lastrowid
            conn.commit()
            conn.close()
        log.info("Saved note for %s (chat %d)", date_str, chat_id)
        return note_id

    def get_recent_notes(self, chat_id: int, days: int = 7) -> list[dict]:
        """Get notes from the last N days."""
        cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT id, text, date, created_at FROM notes
                   WHERE chat_id = ? AND date >= ?
                   ORDER BY created_at DESC""",
                (chat_id, cutoff),
            ).fetchall()
            conn.close()
        return [
            {"id": r[0], "text": r[1], "date": r[2], "created_at": r[3]}
            for r in rows
        ]

    # Key prefixes that should always appear near the top of context
    _PRIORITY_PREFIXES = ("mission_", "blocker_", "action_")

    def format_memory_context(self, chat_id: int) -> str:
        """Format memories for injection into the LLM system prompt.

        Returns an empty string if no memories exist.
        Priority facts (mission_*, blocker_*, action_*) float to the top
        so they're never truncated — the agent always sees its mission state.
        """
        facts = self.get_facts(chat_id, limit=MAX_FACTS_IN_CONTEXT)
        notes = self.get_recent_notes(chat_id, days=7)[:MAX_NOTES_IN_CONTEXT]

        if not facts and not notes:
            return ""

        # Sort: priority-prefixed facts first, then by recency (already sorted)
        priority = [f for f in facts if any(f["key"].startswith(p) for p in self._PRIORITY_PREFIXES)]
        rest = [f for f in facts if f not in priority]
        sorted_facts = priority + rest

        parts = []
        total_chars = 0

        if sorted_facts:
            parts.append("YOUR MEMORY (durable facts stored for this chat):")
            for f in sorted_facts:
                line = f"  {f['key']}: {f['value']}"
                total_chars += len(line)
                if total_chars > MAX_CONTEXT_CHARS:
                    parts.append(
                        "  ... (more stored — use recall_memories or search_memory to see all)"
                    )
                    break
                parts.append(line)

        if notes:
            parts.append("\nRECENT NOTES (ephemeral, last 7 days):")
            for n in notes:
                line = f"  [{n['date']}] {n['text']}"
                total_chars += len(line)
                if total_chars > MAX_CONTEXT_CHARS:
                    parts.append("  ... (more notes stored, use get_daily_notes)")
                    break
                parts.append(line)

        return "\n".join(parts)

    def format_facts_text(self, chat_id: int) -> str:
        """Format facts list as human-readable text."""
        facts = self.get_facts(chat_id)
        if not facts:
            return "No stored memories for this chat."
        lines = [f"Stored memories ({len(facts)} facts):"]
        for f in facts:
            lines.append(f"  [{f['id']}] {f['key']}: {f['value']}")
        return "\n".join(lines)
