"""SQLite-backed scheduled task store.

Stores cron jobs that fire at specified intervals or times,
executing messages through the LLM router on behalf of a chat.

Schedule types:
- 'once': Fire once after N minutes (schedule_value = minutes from now)
- 'interval': Recurring every N seconds (schedule_value = seconds, min 60)
- 'daily': Every day at HH:MM (schedule_value = "HH:MM")
- 'cron': Full cron expression (schedule_value = "* * * * *")
"""

import sqlite3
import time
import datetime
import threading
import logging
from typing import Optional

log = logging.getLogger("cron-store")

# Optional croniter import — gracefully degrade if not installed
try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False
    log.warning("croniter not installed — 'cron' schedule type will not work. pip install croniter")


class CronStore:
    """Thread-safe SQLite cron job store."""

    def __init__(self, db_path: str = "cron.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        log.info("Cron store initialized at %s", db_path)

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    schedule_value TEXT NOT NULL,
                    next_run_at REAL NOT NULL,
                    last_run_at REAL,
                    last_status TEXT,
                    last_error TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cron_next_run
                ON cron_jobs (enabled, next_run_at)
            """)
            conn.commit()
            conn.close()

    def add_job(
        self,
        chat_id: int,
        name: str,
        message: str,
        schedule_type: str,
        schedule_value: str,
    ) -> int:
        """Add a new scheduled job.

        Args:
            chat_id: Telegram chat to deliver results to.
            name: Human-readable job name.
            message: The instruction to send to the LLM when the job fires.
            schedule_type: 'once', 'interval', 'daily', or 'cron'.
            schedule_value: For 'once': minutes from now.
                           For 'interval': seconds between runs.
                           For 'daily': HH:MM time string.
                           For 'cron': cron expression (e.g. "0 9 * * 1-5").

        Returns: The new job ID.
        Raises: ValueError if schedule_type is 'cron' but croniter is not installed,
                or if the cron expression is invalid.
        """
        if schedule_type == "cron":
            if not HAS_CRONITER:
                raise ValueError("croniter library not installed — cannot use 'cron' schedule type")
            # Validate the cron expression
            try:
                croniter(schedule_value)
            except (ValueError, KeyError) as e:
                raise ValueError(f"Invalid cron expression '{schedule_value}': {e}")
        now = time.time()
        next_run = self._compute_next_run(schedule_type, schedule_value, now)

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """INSERT INTO cron_jobs
                   (chat_id, name, message, schedule_type, schedule_value,
                    next_run_at, enabled, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                (chat_id, name, message, schedule_type, schedule_value, next_run, now),
            )
            job_id = cursor.lastrowid
            conn.commit()
            conn.close()

        log.info("Created cron job %d: '%s' (%s=%s) for chat %d, next run at %s",
                 job_id, name, schedule_type, schedule_value, chat_id,
                 time.strftime("%Y-%m-%d %H:%M", time.localtime(next_run)))
        return job_id

    def list_jobs(self, chat_id: int) -> list[dict]:
        """List all jobs for a chat."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT id, name, message, schedule_type, schedule_value,
                          next_run_at, last_run_at, last_status, enabled
                   FROM cron_jobs WHERE chat_id = ?
                   ORDER BY next_run_at""",
                (chat_id,),
            ).fetchall()
            conn.close()

        return [
            {
                "id": r[0], "name": r[1], "message": r[2],
                "schedule_type": r[3], "schedule_value": r[4],
                "next_run_at": r[5], "last_run_at": r[6],
                "last_status": r[7], "enabled": bool(r[8]),
            }
            for r in rows
        ]

    def remove_job(self, job_id: int, chat_id: int) -> bool:
        """Remove a job (only if it belongs to the given chat)."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "DELETE FROM cron_jobs WHERE id = ? AND chat_id = ?",
                (job_id, chat_id),
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
        return deleted

    def get_due_jobs(self) -> list[dict]:
        """Get all enabled jobs whose next_run_at has passed."""
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT id, chat_id, name, message, schedule_type, schedule_value
                   FROM cron_jobs
                   WHERE enabled = 1 AND next_run_at <= ?""",
                (now,),
            ).fetchall()
            conn.close()

        return [
            {
                "id": r[0], "chat_id": r[1], "name": r[2],
                "message": r[3], "schedule_type": r[4], "schedule_value": r[5],
            }
            for r in rows
        ]

    def mark_job_run(self, job_id: int):
        """Mark a job as successfully run and compute next run time."""
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT schedule_type, schedule_value FROM cron_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

            if not row:
                conn.close()
                return

            schedule_type, schedule_value = row

            if schedule_type == "once":
                # One-shot job — disable it
                conn.execute(
                    "UPDATE cron_jobs SET last_run_at = ?, last_status = 'ok', enabled = 0 WHERE id = ?",
                    (now, job_id),
                )
            else:
                next_run = self._compute_next_run(schedule_type, schedule_value, now)
                conn.execute(
                    "UPDATE cron_jobs SET last_run_at = ?, last_status = 'ok', next_run_at = ? WHERE id = ?",
                    (now, next_run, job_id),
                )
            conn.commit()
            conn.close()

    def mark_job_error(self, job_id: int, error: str):
        """Mark a job as failed but still schedule next run."""
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT schedule_type, schedule_value FROM cron_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

            if not row:
                conn.close()
                return

            schedule_type, schedule_value = row
            next_run = self._compute_next_run(schedule_type, schedule_value, now)

            conn.execute(
                """UPDATE cron_jobs SET last_run_at = ?, last_status = 'error',
                   last_error = ?, next_run_at = ? WHERE id = ?""",
                (now, error[:500], next_run, job_id),
            )
            conn.commit()
            conn.close()

    @staticmethod
    def _compute_next_run(schedule_type: str, schedule_value: str, after: float) -> float:
        """Compute the next run timestamp."""
        if schedule_type == "once":
            # schedule_value = minutes from now
            try:
                minutes = float(schedule_value)
                return after + minutes * 60
            except ValueError:
                return after + 60  # default 1 minute

        elif schedule_type == "interval":
            # schedule_value = seconds between runs
            try:
                seconds = float(schedule_value)
                return after + max(seconds, 60)  # minimum 1 minute
            except ValueError:
                return after + 3600  # default 1 hour

        elif schedule_type == "daily":
            # schedule_value = "HH:MM"
            try:
                hour, minute = map(int, schedule_value.split(":"))
                now_dt = datetime.datetime.fromtimestamp(after)
                target = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now_dt:
                    target += datetime.timedelta(days=1)
                return target.timestamp()
            except (ValueError, AttributeError):
                return after + 86400  # default 24 hours

        elif schedule_type == "cron":
            # schedule_value = cron expression (e.g., "0 9 * * 1-5")
            if HAS_CRONITER:
                try:
                    base_dt = datetime.datetime.fromtimestamp(after)
                    cron = croniter(schedule_value, base_dt)
                    next_dt = cron.get_next(datetime.datetime)
                    return next_dt.timestamp()
                except Exception:
                    log.warning("Invalid cron expression '%s', falling back to 1h", schedule_value)
                    return after + 3600
            else:
                log.warning("croniter not installed, falling back to 1h for cron job")
                return after + 3600

        else:
            return after + 3600  # fallback

    def format_jobs_text(self, chat_id: int) -> str:
        """Format job list as human-readable text."""
        jobs = self.list_jobs(chat_id)
        if not jobs:
            return "No scheduled tasks for this chat."

        lines = [f"Scheduled tasks ({len(jobs)}):"]
        for j in jobs:
            status = "enabled" if j["enabled"] else "disabled"
            schedule = f"{j['schedule_type']}={j['schedule_value']}"
            next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(j["next_run_at"]))
            msg_preview = j["message"][:60] + ("..." if len(j["message"]) > 60 else "")
            lines.append(
                f"  [{j['id']}] {j['name']} ({schedule}, {status})\n"
                f"       Next: {next_run}\n"
                f"       Task: {msg_preview}"
            )
        return "\n".join(lines)
