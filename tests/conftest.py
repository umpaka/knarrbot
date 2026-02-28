"""Shared fixtures for the knarrbot test suite."""

import os
import sys

import pytest

# Add core/ to sys.path so shared modules are importable
CORE_DIR = os.path.join(os.path.dirname(__file__), "..", "core")
sys.path.insert(0, CORE_DIR)

# Add adapters/telegram/ so adapter modules are importable
TELEGRAM_DIR = os.path.join(os.path.dirname(__file__), "..", "adapters", "telegram")
sys.path.insert(0, TELEGRAM_DIR)

# Also add tests dir so helpers.py is importable
TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from helpers import MockKnarrClient


# ── SQLite store fixtures ────────────────────────────────────────

@pytest.fixture
def memory_store(tmp_path):
    """Fresh MemoryStore backed by a temp SQLite file."""
    from memory_store import MemoryStore
    return MemoryStore(db_path=str(tmp_path / "memory.db"))


@pytest.fixture
def chat_store(tmp_path):
    """Fresh ChatStore backed by a temp SQLite file."""
    from chat_store import ChatStore
    return ChatStore(db_path=str(tmp_path / "chat.db"))


@pytest.fixture
def cron_store(tmp_path):
    """Fresh CronStore backed by a temp SQLite file."""
    from cron_store import CronStore
    return CronStore(db_path=str(tmp_path / "cron.db"))


@pytest.fixture
def session_store(tmp_path):
    """Fresh SessionStore backed by a temp SQLite file."""
    from session_store import SessionStore
    return SessionStore(db_path=str(tmp_path / "sessions.db"))


# ── Mock send function ──────────────────────────────────────────

class MessageCapture:
    """Captures messages sent via the agent's send_fn callback."""

    def __init__(self):
        self.messages: list[tuple[int, str, str]] = []  # (chat_id, text, parse_mode)

    async def __call__(self, chat_id: int, text: str, parse_mode: str = ""):
        self.messages.append((chat_id, text, parse_mode))

    @property
    def texts(self) -> list[str]:
        """Just the text of each sent message."""
        return [m[1] for m in self.messages]

    @property
    def last_text(self) -> str:
        """Text of the most recently sent message."""
        return self.messages[-1][1] if self.messages else ""

    def clear(self):
        self.messages.clear()


@pytest.fixture
def send_fn():
    """A MessageCapture instance usable as an async send callback."""
    return MessageCapture()


# ── Mock Knarr client ────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """A MockKnarrClient instance for tests that need a Knarr API client."""
    return MockKnarrClient()
