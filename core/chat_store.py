"""SQLite-backed chat message store.

Stores all messages the gateway processes so the LLM can search and
summarize past conversations.  Each message is stored with its chat_id,
timestamp, username, and text.
"""

import sqlite3
import time
import threading
import logging
from typing import Optional

log = logging.getLogger("chat-store")

# Cap returned messages to prevent blowing the LLM context
MAX_RETURN_MESSAGES = 100
MAX_RETURN_CHARS = 8000


class ChatStore:
    """Thread-safe SQLite message store."""

    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        log.info("Chat store initialized at %s", db_path)

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    chat_title TEXT,
                    username TEXT,
                    display_name TEXT,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    message_id INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_ts
                ON messages (chat_id, timestamp)
            """)
            conn.commit()
            conn.close()

    def store_message(
        self,
        chat_id: int,
        username: str,
        text: str,
        chat_title: str = "",
        display_name: str = "",
        message_id: int = 0,
        timestamp: Optional[float] = None,
    ):
        """Store a message in the database."""
        ts = timestamp or time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO messages
                   (chat_id, chat_title, username, display_name, text, timestamp, message_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chat_id, chat_title, username, display_name, text, ts, message_id),
            )
            conn.commit()
            conn.close()

    def get_history(
        self,
        chat_id: int,
        limit: int = 50,
        since_minutes: Optional[int] = None,
        username: Optional[str] = None,
        search: Optional[str] = None,
    ) -> str:
        """Retrieve chat history as formatted text.

        Args:
            chat_id: The Telegram chat ID
            limit: Max number of messages to return (capped at MAX_RETURN_MESSAGES)
            since_minutes: Only return messages from the last N minutes
            username: Filter by username
            search: Search for text in messages

        Returns:
            Formatted string of messages, ready for LLM consumption.
        """
        limit = min(limit, MAX_RETURN_MESSAGES)

        conditions = ["chat_id = ?"]
        params: list = [chat_id]

        if since_minutes:
            cutoff = time.time() - (since_minutes * 60)
            conditions.append("timestamp >= ?")
            params.append(cutoff)

        if username:
            # Strip @ prefix if present
            username = username.lstrip("@")
            conditions.append("username = ?")
            params.append(username)

        if search:
            conditions.append("text LIKE ?")
            params.append(f"%{search}%")

        where = " AND ".join(conditions)
        params.append(limit)

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                f"""SELECT username, display_name, text, timestamp
                    FROM messages
                    WHERE {where}
                    ORDER BY timestamp DESC
                    LIMIT ?""",
                params,
            ).fetchall()
            conn.close()

        if not rows:
            return "No messages found matching the criteria."

        # Reverse so oldest is first (chronological order)
        rows.reverse()

        lines = []
        total_chars = 0
        for username, display_name, text, ts in rows:
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            name = display_name or username or "unknown"
            line = f"[{time_str}] @{username} ({name}): {text}"
            total_chars += len(line)
            if total_chars > MAX_RETURN_CHARS:
                lines.append("... (older messages truncated) ...")
                break
            lines.append(line)

        header = f"Chat history ({len(lines)} messages"
        if since_minutes:
            header += f", last {since_minutes} min"
        header += "):"

        return header + "\n" + "\n".join(lines)

    def get_active_chats(self, since_timestamp: float) -> list:
        """Return chats that have had messages since the given timestamp.

        Returns a list of dicts: [{chat_id, chat_title, msg_count}, ...]
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT chat_id, chat_title, COUNT(*) as msg_count
                   FROM messages
                   WHERE timestamp > ?
                   GROUP BY chat_id
                   ORDER BY msg_count DESC""",
                (since_timestamp,),
            ).fetchall()
            conn.close()

        return [
            {"chat_id": row[0], "chat_title": row[1] or "DM", "msg_count": row[2]}
            for row in rows
        ]

    def get_stats(self, chat_id: int) -> dict:
        """Get stats about stored messages for a chat."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            conn.close()

        total, first_ts, last_ts = row
        return {
            "total_messages": total,
            "first_message": time.strftime("%Y-%m-%d %H:%M", time.localtime(first_ts)) if first_ts else None,
            "last_message": time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ts)) if last_ts else None,
        }
