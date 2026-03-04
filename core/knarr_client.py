"""Knarr Cockpit API client — typed async HTTP wrapper.

Replaces direct DHTNode imports with clean HTTP calls to the Cockpit REST API.
Every method maps 1-to-1 to a Cockpit endpoint; the bot no longer embeds its
own node process.

Environment:
    KNARR_API_URL    — Base URL of the Cockpit API (e.g. http://localhost:9100)
    KNARR_API_TOKEN  — Bearer token for authentication
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("knarr-client")

__all__ = ["KnarrClient", "KnarrAPIError"]


class KnarrAPIError(Exception):
    """Raised when the Cockpit API returns a non-2xx status."""

    def __init__(self, status_code: int, message: str, endpoint: str = ""):
        self.status_code = status_code
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"[{status_code}] {endpoint}: {message}")


class KnarrClient:
    """Async HTTP client for the Knarr Cockpit REST API.

    Provides typed, high-level methods for every operation the Telegram bot
    needs: skill discovery, task execution, messaging, asset transfer, and
    node status queries.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        # Disable TLS verification for localhost self-signed certs
        verify = False if "localhost" in base_url or "127.0.0.1" in base_url else True
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=10.0),
            verify=verify,
        )

    # ── Internal helpers ──────────────────────────────────────────

    async def _get(self, path: str, params: dict[str, Any] | None = None,
                   timeout: float | None = None) -> Any:
        """GET request with JSON response parsing."""
        try:
            resp = await self._http.get(path, params=params, timeout=timeout)
        except httpx.TimeoutException:
            log.warning("GET %s timed out", path)
            raise KnarrAPIError(504, "Request timed out", path)
        except httpx.ConnectError as e:
            log.warning("GET %s connection failed: %s", path, e)
            raise KnarrAPIError(502, f"Connection failed: {e}", path)
        if resp.status_code >= 400:
            body = resp.text[:500]
            log.warning("GET %s → %d: %s", path, resp.status_code, body)
            raise KnarrAPIError(resp.status_code, body, path)
        return resp.json()

    async def _post(self, path: str, json_data: dict[str, Any] | None = None,
                    content: bytes | None = None,
                    params: dict[str, Any] | None = None,
                    timeout: float | None = None,
                    extra_headers: dict[str, str] | None = None) -> Any:
        """POST request with JSON or binary body."""
        kwargs: dict[str, Any] = {"timeout": timeout, "params": params}
        if extra_headers:
            kwargs["headers"] = extra_headers
        if content is not None:
            kwargs["content"] = content
        else:
            kwargs["json"] = json_data or {}
        try:
            resp = await self._http.post(path, **kwargs)
        except httpx.TimeoutException:
            log.warning("POST %s timed out", path)
            raise KnarrAPIError(504, "Request timed out", path)
        except httpx.ConnectError as e:
            log.warning("POST %s connection failed: %s", path, e)
            raise KnarrAPIError(502, f"Connection failed: {e}", path)
        if resp.status_code >= 400:
            body = resp.text[:500]
            log.warning("POST %s → %d: %s", path, resp.status_code, body)
            raise KnarrAPIError(resp.status_code, body, path)
        return resp.json()

    async def _get_raw(self, path: str, params: dict[str, Any] | None = None,
                       timeout: float | None = None) -> bytes:
        """GET request returning raw bytes (for asset downloads)."""
        try:
            resp = await self._http.get(path, params=params, timeout=timeout)
        except httpx.TimeoutException:
            log.warning("GET %s timed out (raw)", path)
            raise KnarrAPIError(504, "Request timed out", path)
        except httpx.ConnectError as e:
            log.warning("GET %s connection failed (raw): %s", path, e)
            raise KnarrAPIError(502, f"Connection failed: {e}", path)
        if resp.status_code >= 400:
            body = resp.text[:500]
            log.warning("GET %s → %d (raw): %s", path, resp.status_code, body)
            raise KnarrAPIError(resp.status_code, body, path)
        return resp.content

    # ── Discovery ─────────────────────────────────────────────────

    async def get_skills(self) -> dict[str, Any]:
        """Fetch local and network skills.

        Returns ``{"local": [...], "network": [...]}``.
        Each network entry includes ``providers`` with host/port/sidecar_port.
        """
        return await self._get("/api/skills")

    async def query_skill(self, name: str) -> list[dict[str, Any]]:
        """Find providers for a specific skill by name.

        Returns a list of provider dicts (may be empty).
        """
        skills = await self.get_skills()
        for s in skills.get("network", []):
            if s.get("name", "").lower() == name.lower():
                return s.get("providers", [])
        return []

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        skill: str,
        input_data: dict[str, Any],
        provider: dict[str, Any] | None = None,
        timeout: int = 30,
        local: bool = False,
    ) -> dict[str, Any]:
        """Execute a skill task via the Cockpit API.

        Args:
            skill: Skill name (e.g. "echo", "knarr-mail").
            input_data: Input parameters for the skill.
            provider: Optional provider dict with node_id, host, port.
            timeout: Timeout in seconds.
            local: Force local execution.

        Returns:
            Response dict with ``status``, ``output_data`` or ``error``,
            and ``wall_time_ms``.
        """
        body: dict[str, Any] = {
            "skill": skill,
            "input": input_data,
            "timeout": timeout,
        }
        if provider:
            body["provider"] = provider
        if local:
            body["local"] = True
        return await self._post(
            "/api/execute", json_data=body,
            timeout=float(timeout) + 10.0,  # HTTP timeout > skill timeout
        )

    async def execute_async(
        self,
        skill: str,
        input_data: dict[str, Any],
        provider: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Submit an async task via the Cockpit API (v0.13.0+).

        Returns immediately with ``{"status": "accepted", "job_id": "...",
        "position": N}``.  Use :meth:`get_job_status` / :meth:`get_job_result`
        to poll for completion.

        Falls back to synchronous :meth:`execute` if the API returns a
        non-202 response (older protocol version).
        """
        body: dict[str, Any] = {
            "skill": skill,
            "input": input_data,
            "timeout": timeout,
            "async": True,
        }
        if provider:
            body["provider"] = provider
        try:
            resp = await self._http.post(
                "/api/execute", json=body, timeout=30.0)
        except httpx.TimeoutException:
            log.warning("POST /api/execute (async) timed out for '%s'", skill)
            raise KnarrAPIError(504, "Request timed out", "/api/execute")
        except httpx.ConnectError as e:
            log.warning("POST /api/execute (async) connection failed: %s", e)
            raise KnarrAPIError(502, f"Connection failed: {e}", "/api/execute")

        if resp.status_code == 202:
            return resp.json()

        # Fallback: API didn't understand async — treat as sync response
        if resp.status_code < 400:
            log.info("Async execute not supported for '%s', got sync response", skill)
            return resp.json()

        body_text = resp.text[:500]
        log.warning("POST /api/execute (async) → %d: %s", resp.status_code, body_text)
        raise KnarrAPIError(resp.status_code, body_text, "/api/execute")

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Poll async job status (v0.13.0+).

        Returns ``{"job_id": "...", "status": "queued"|"running"|"completed"|"failed"|"expired",
        "position": N, "updated_at": "..."}``.
        """
        return await self._get(f"/api/jobs/{job_id}", timeout=10.0)

    async def get_job_result(self, job_id: str) -> dict[str, Any]:
        """Retrieve completed async job result (v0.13.0+).

        Returns ``{"job_id": "...", "status": "completed", "output_data": {...}}``.
        Raises :class:`KnarrAPIError` (404) if job not found, (410) if expired.
        """
        return await self._get(f"/api/jobs/{job_id}/result", timeout=30.0)

    # ── Messages (knarr-mail) ─────────────────────────────────────

    async def poll_messages(
        self,
        since: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Poll the knarr-mail inbox.

        Returns ``{"messages": [...], "next_token": "...", "total_unread": N}``.
        """
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if limit != 50:
            params["limit"] = limit
        return await self._get("/api/messages", params=params or None)

    async def ack_messages(self, message_ids: list[str]) -> dict[str, Any]:
        """Acknowledge (mark as read) one or more messages.

        Returns ``{"acknowledged": N}``.
        """
        return await self._post("/api/messages/ack", json_data={
            "message_ids": message_ids,
        })

    async def send_message(
        self,
        to_node: str,
        body: dict[str, Any],
        ttl_hours: float = 72,
    ) -> dict[str, Any]:
        """Send a knarr-mail message to another node.

        Uses the dedicated ``POST /api/messages/send`` endpoint.
        The network handles delivery to the recipient.
        """
        return await self._post("/api/messages/send", json_data={
            "to": to_node,
            "body": body,
            "ttl_hours": ttl_hours,
        })

    # ── Assets ────────────────────────────────────────────────────

    async def upload_asset(
        self,
        data: bytes,
        host: str = "",
        sidecar_port: int = 0,
    ) -> str:
        """Upload binary data to a sidecar.

        Args:
            data: Raw bytes to upload.
            host: Target host (empty = local sidecar).
            sidecar_port: Target sidecar port (0 = local).

        Returns:
            The asset hash (64-char hex string).
        """
        params: dict[str, Any] = {}
        if host:
            params["host"] = host
        if sidecar_port > 0:
            params["sidecar_port"] = str(sidecar_port)
        result = await self._post(
            "/api/upload",
            content=data,
            params=params or None,
            timeout=60.0,
            extra_headers={"Content-Type": "application/octet-stream"},
        )
        return result.get("hash", "")

    async def download_asset(
        self,
        asset_hash: str,
        host: str = "",
        sidecar_port: int = 0,
    ) -> bytes:
        """Download an asset by hash.

        Args:
            asset_hash: 64-char hex SHA256 hash.
            host: Remote host to proxy from (empty = local).
            sidecar_port: Remote sidecar port (0 = local).

        Returns:
            Raw file bytes.
        """
        params: dict[str, Any] = {}
        if host:
            params["host"] = host
        if sidecar_port > 0:
            params["sidecar_port"] = str(sidecar_port)
        return await self._get_raw(
            f"/api/assets/{asset_hash}",
            params=params or None,
            timeout=60.0,
        )

    # ── Status / metadata ─────────────────────────────────────────

    async def get_status(self) -> dict[str, Any]:
        """Node status summary.

        Returns dict with ``node_id``, ``port``, ``peer_count``,
        ``uptime_seconds``, ``task_slots``, ``advertise_host``, etc.
        """
        return await self._get("/api/status")

    async def get_peers(self) -> list[dict[str, Any]]:
        """List known peers.

        Each entry has ``node_id``, ``host``, ``port``, ``last_seen``, ``load``.
        """
        result = await self._get("/api/peers")
        if isinstance(result, list):
            return result
        # Some versions wrap in a dict
        return result.get("peers", result) if isinstance(result, dict) else []

    async def get_economy(self) -> dict[str, Any]:
        """Economy summary (per-peer positions + totals)."""
        return await self._get("/api/economy")

    async def get_reputation(self) -> list[dict[str, Any]]:
        """Per-provider reputation data from the protocol's task history.

        Returns a list of dicts with ``provider_node_id``, ``success_rate``
        (0.0-1.0 or null), ``avg_wall_time_ms``, ``total_tasks_30d``,
        ``balance``, and ``last_interaction``.
        """
        result = await self._get("/api/reputation")
        if isinstance(result, list):
            return result
        return result.get("reputations", result) if isinstance(result, dict) else []

    async def get_skill_schema(self, name: str) -> dict[str, Any]:
        """Full skill metadata including per-provider load values.

        Returns dict with ``name``, ``description``, ``input_schema``,
        ``providers`` (with ``load``), ``price``, etc.
        Raises :class:`KnarrAPIError` (404) if the skill is not found.
        """
        return await self._get(f"/api/skills/{name}/schema")

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "KnarrClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
