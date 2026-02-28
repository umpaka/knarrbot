"""Shared test helpers — mock classes and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Legacy MockTaskResult (still used by format_skill_result tests) ───

@dataclass
class MockTaskResult:
    status: str = "completed"
    output_data: dict = field(default_factory=lambda: {"result": "ok"})
    error: dict = field(default_factory=dict)


# ── MockKnarrClient ──────────────────────────────────────────────────

class MockKnarrClient:
    """Mock of KnarrClient for agent/router tests.

    Pre-populated with reasonable defaults. Override ``_status``, ``_skills``,
    ``_execute_result``, ``_peers``, ``_economy``, ``_messages`` to control
    return values in your tests.
    """

    def __init__(self) -> None:
        # -- Status -----------------------------------------------------------
        self._status: dict[str, Any] = {
            "node_id": "abcdef1234567890abcdef1234567890",
            "port": 9100,
            "peer_count": 0,
            "uptime_seconds": 3600,
            "skill_count": 0,
            "network_skill_count": 0,
            "version": "0.13.2",
            "task_slots": {"used": 0, "total": 10},
            "advertise_host": "",
        }

        # -- Skills -----------------------------------------------------------
        self._skills: dict[str, Any] = {
            "local": [],
            "network": [],
        }

        # -- Execute ----------------------------------------------------------
        self._execute_result: dict[str, Any] = {
            "status": "completed",
            "output_data": {"result": "ok"},
            "error": {},
            "wall_time_ms": 42,
        }

        # -- Peers ------------------------------------------------------------
        self._peers: list[dict[str, Any]] = []

        # -- Economy ----------------------------------------------------------
        self._economy: dict[str, Any] = {}

        # -- Messages ---------------------------------------------------------
        self._messages: dict[str, Any] = {"messages": [], "total_unread": 0}
        self._ack_count: int = 0

    # -- Discovery --------------------------------------------------------

    async def get_skills(self) -> dict[str, Any]:
        return self._skills

    async def query_skill(self, name: str) -> list[dict[str, Any]]:
        for s in self._skills.get("network", []):
            if s.get("name", "").lower() == name.lower():
                return s.get("providers", [])
        return []

    # -- Execution --------------------------------------------------------

    async def execute(self, skill: str, input_data: dict[str, Any],
                      provider: dict[str, Any] | None = None,
                      timeout: int = 30, local: bool = False) -> dict[str, Any]:
        return self._execute_result

    # -- Messages ---------------------------------------------------------

    async def poll_messages(self, since: str | None = None,
                            limit: int = 50) -> dict[str, Any]:
        return self._messages

    async def ack_messages(self, message_ids: list[str]) -> dict[str, Any]:
        self._ack_count += len(message_ids)
        return {"acknowledged": len(message_ids)}

    async def send_message(self, to_node: str, body: dict[str, Any],
                           ttl_hours: float = 72) -> dict[str, Any]:
        return {
            "status": "completed",
            "output_data": {"status": "delivered"},
        }

    # -- Assets -----------------------------------------------------------

    async def upload_asset(self, data: bytes, host: str = "",
                           sidecar_port: int = 0) -> str:
        return "a" * 64

    async def download_asset(self, asset_hash: str, host: str = "",
                             sidecar_port: int = 0) -> bytes:
        return b"mock-asset-content"

    # -- Status / metadata ------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        return self._status

    async def get_peers(self) -> list[dict[str, Any]]:
        return self._peers

    async def get_economy(self) -> dict[str, Any]:
        return self._economy

    # -- Lifecycle --------------------------------------------------------

    async def close(self) -> None:
        pass
