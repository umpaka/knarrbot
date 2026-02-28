"""Tests for knarr_client.py — HTTP client for the Knarr Cockpit API.

Uses httpx mock transport so no real HTTP calls are made.
"""

import json
import pytest
import httpx

from knarr_client import KnarrClient, KnarrAPIError


# ── Helpers ───────────────────────────────────────────────────────

def _mock_transport(handler):
    """Create an httpx.MockTransport from an async handler function."""
    return httpx.MockTransport(handler)


def _json_response(data, status_code=200):
    """Build an httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        headers={"content-type": "application/json"},
    )


def _bytes_response(data: bytes, status_code=200, content_type="application/octet-stream"):
    return httpx.Response(
        status_code=status_code,
        content=data,
        headers={"content-type": content_type},
    )


def _error_response(status_code, message):
    return httpx.Response(
        status_code=status_code,
        text=message,
        headers={"content-type": "text/plain"},
    )


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def make_client():
    """Factory fixture: create a KnarrClient backed by a mock transport."""
    clients = []

    def _factory(handler):
        transport = _mock_transport(handler)
        client = KnarrClient.__new__(KnarrClient)
        client._base_url = "http://test:9100"
        client._token = "test-token"
        client._http = httpx.AsyncClient(
            transport=transport,
            base_url="http://test:9100",
            headers={
                "Authorization": "Bearer test-token",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        clients.append(client)
        return client

    yield _factory

    # Cleanup — close the underlying httpx client synchronously
    for c in clients:
        try:
            c._http.close()
        except Exception:
            pass


# ── Auth header tests ─────────────────────────────────────────────

class TestAuthHeader:
    @pytest.mark.asyncio
    async def test_bearer_token_sent(self, make_client):
        """Every request includes the Authorization: Bearer header."""
        captured_headers = {}

        def handler(request: httpx.Request):
            captured_headers.update(dict(request.headers))
            return _json_response({"ok": True})

        client = make_client(handler)
        await client.get_status()
        assert captured_headers.get("authorization") == "Bearer test-token"


# ── GET /api/status ───────────────────────────────────────────────

class TestGetStatus:
    @pytest.mark.asyncio
    async def test_returns_status_dict(self, make_client):
        status_data = {
            "node_id": "abcdef1234567890",
            "port": 9100,
            "peer_count": 5,
            "uptime_seconds": 3600,
        }

        def handler(request: httpx.Request):
            assert request.url.path == "/api/status"
            return _json_response(status_data)

        client = make_client(handler)
        result = await client.get_status()
        assert result["node_id"] == "abcdef1234567890"
        assert result["peer_count"] == 5

    @pytest.mark.asyncio
    async def test_error_raises(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(401, "Unauthorized")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.get_status()
        assert exc_info.value.status_code == 401


# ── GET /api/skills ───────────────────────────────────────────────

class TestGetSkills:
    @pytest.mark.asyncio
    async def test_returns_skills(self, make_client):
        skills_data = {
            "local": [{"name": "knarr-mail", "handler": "handler.py:handle"}],
            "network": [
                {
                    "name": "echo",
                    "description": "Echo back",
                    "providers": [{"node_id": "abc", "host": "1.2.3.4", "port": 9200}],
                }
            ],
        }

        def handler(request: httpx.Request):
            assert request.url.path == "/api/skills"
            return _json_response(skills_data)

        client = make_client(handler)
        result = await client.get_skills()
        assert len(result["local"]) == 1
        assert result["network"][0]["name"] == "echo"


class TestQuerySkill:
    @pytest.mark.asyncio
    async def test_finds_skill_providers(self, make_client):
        skills_data = {
            "local": [],
            "network": [
                {
                    "name": "echo",
                    "providers": [
                        {"node_id": "abc", "host": "1.2.3.4", "port": 9200},
                    ],
                },
                {
                    "name": "browse-web",
                    "providers": [
                        {"node_id": "def", "host": "5.6.7.8", "port": 9300},
                    ],
                },
            ],
        }

        def handler(request: httpx.Request):
            return _json_response(skills_data)

        client = make_client(handler)
        providers = await client.query_skill("echo")
        assert len(providers) == 1
        assert providers[0]["node_id"] == "abc"

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_skill(self, make_client):
        skills_data = {"local": [], "network": []}

        def handler(request: httpx.Request):
            return _json_response(skills_data)

        client = make_client(handler)
        providers = await client.query_skill("nonexistent")
        assert providers == []


# ── POST /api/execute ─────────────────────────────────────────────

class TestExecute:
    @pytest.mark.asyncio
    async def test_basic_execute(self, make_client):
        captured_body = {}

        def handler(request: httpx.Request):
            assert request.url.path == "/api/execute"
            captured_body.update(json.loads(request.content))
            return _json_response({
                "status": "completed",
                "output_data": {"result": "hello"},
                "wall_time_ms": 42,
            })

        client = make_client(handler)
        result = await client.execute("echo", {"text": "hi"})
        assert result["status"] == "completed"
        assert result["output_data"]["result"] == "hello"
        assert captured_body["skill"] == "echo"
        assert captured_body["input"] == {"text": "hi"}

    @pytest.mark.asyncio
    async def test_execute_with_provider(self, make_client):
        captured_body = {}

        def handler(request: httpx.Request):
            captured_body.update(json.loads(request.content))
            return _json_response({"status": "completed", "output_data": {}})

        client = make_client(handler)
        await client.execute(
            "echo", {"text": "hi"},
            provider={"node_id": "abc", "host": "1.2.3.4", "port": 9200},
            timeout=60,
        )
        assert captured_body["provider"]["node_id"] == "abc"
        assert captured_body["timeout"] == 60

    @pytest.mark.asyncio
    async def test_execute_failure(self, make_client):
        def handler(request: httpx.Request):
            return _json_response({
                "status": "failed",
                "error": {"code": "HANDLER_ERROR", "message": "boom"},
            })

        client = make_client(handler)
        result = await client.execute("bad-skill", {})
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_http_error(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(500, "Internal Server Error")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.execute("echo", {})
        assert exc_info.value.status_code == 500


# ── Messages ──────────────────────────────────────────────────────

class TestPollMessages:
    @pytest.mark.asyncio
    async def test_poll(self, make_client):
        def handler(request: httpx.Request):
            assert request.url.path == "/api/messages"
            return _json_response({
                "messages": [{"message_id": "m1", "body": {"content": "hi"}}],
                "total_unread": 1,
            })

        client = make_client(handler)
        result = await client.poll_messages()
        assert len(result["messages"]) == 1


class TestAckMessages:
    @pytest.mark.asyncio
    async def test_ack(self, make_client):
        captured_body = {}

        def handler(request: httpx.Request):
            assert request.url.path == "/api/messages/ack"
            captured_body.update(json.loads(request.content))
            return _json_response({"acknowledged": 2})

        client = make_client(handler)
        result = await client.ack_messages(["m1", "m2"])
        assert result["acknowledged"] == 2
        assert captured_body["message_ids"] == ["m1", "m2"]


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send(self, make_client):
        captured_body = {}

        def handler(request: httpx.Request):
            assert request.url.path == "/api/execute"
            captured_body.update(json.loads(request.content))
            return _json_response({
                "status": "completed",
                "output_data": {"status": "delivered"},
            })

        client = make_client(handler)
        result = await client.send_message("target-node-id", {"type": "text", "content": "hello"})
        assert result["status"] == "completed"
        assert captured_body["skill"] == "knarr-mail"
        assert captured_body["input"]["action"] == "send"
        assert captured_body["provider"]["node_id"] == "target-node-id"


# ── Assets ────────────────────────────────────────────────────────

class TestUploadAsset:
    @pytest.mark.asyncio
    async def test_upload_local(self, make_client):
        def handler(request: httpx.Request):
            assert request.url.path == "/api/upload"
            assert request.content == b"file-data-here"
            return _json_response({"hash": "a" * 64})

        client = make_client(handler)
        h = await client.upload_asset(b"file-data-here")
        assert h == "a" * 64

    @pytest.mark.asyncio
    async def test_upload_remote(self, make_client):
        captured_params = {}

        def handler(request: httpx.Request):
            captured_params.update(dict(request.url.params))
            return _json_response({"hash": "b" * 64})

        client = make_client(handler)
        await client.upload_asset(b"data", host="1.2.3.4", sidecar_port=8001)
        assert captured_params["host"] == "1.2.3.4"
        assert captured_params["sidecar_port"] == "8001"


class TestDownloadAsset:
    @pytest.mark.asyncio
    async def test_download_local(self, make_client):
        def handler(request: httpx.Request):
            assert "/api/assets/" in str(request.url)
            return _bytes_response(b"pdf-content")

        client = make_client(handler)
        data = await client.download_asset("c" * 64)
        assert data == b"pdf-content"

    @pytest.mark.asyncio
    async def test_download_remote(self, make_client):
        captured_params = {}

        def handler(request: httpx.Request):
            captured_params.update(dict(request.url.params))
            return _bytes_response(b"content")

        client = make_client(handler)
        await client.download_asset("d" * 64, host="5.6.7.8", sidecar_port=8002)
        assert captured_params["host"] == "5.6.7.8"
        assert captured_params["sidecar_port"] == "8002"

    @pytest.mark.asyncio
    async def test_download_404(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(404, "Not found")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.download_asset("e" * 64)
        assert exc_info.value.status_code == 404


# ── GET /api/peers ────────────────────────────────────────────────

class TestGetPeers:
    @pytest.mark.asyncio
    async def test_returns_list(self, make_client):
        peers_data = [
            {"node_id": "abc", "host": "1.2.3.4", "port": 9200, "last_seen": 1700000000.0},
        ]

        def handler(request: httpx.Request):
            assert request.url.path == "/api/peers"
            return _json_response(peers_data)

        client = make_client(handler)
        result = await client.get_peers()
        assert len(result) == 1
        assert result[0]["node_id"] == "abc"


# ── GET /api/economy ──────────────────────────────────────────────

class TestGetEconomy:
    @pytest.mark.asyncio
    async def test_returns_economy(self, make_client):
        econ_data = {"total_earned": 100.0, "total_spent": 50.0}

        def handler(request: httpx.Request):
            assert request.url.path == "/api/economy"
            return _json_response(econ_data)

        client = make_client(handler)
        result = await client.get_economy()
        assert result["total_earned"] == 100.0


# ── Error handling ────────────────────────────────────────────────

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_401_unauthorized(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(401, "Unauthorized")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.get_status()
        assert exc_info.value.status_code == 401
        assert "Unauthorized" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_404_not_found(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(404, "Not Found")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.get_economy()
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_500_server_error(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(500, "Internal Server Error")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.get_skills()
        assert exc_info.value.status_code == 500


# ── Lifecycle ─────────────────────────────────────────────────────

# ── GET /api/reputation ───────────────────────────────────────────

class TestGetReputation:
    @pytest.mark.asyncio
    async def test_returns_list(self, make_client):
        rep_data = [
            {
                "provider_node_id": "node1",
                "success_rate": 0.95,
                "avg_wall_time_ms": 150,
                "total_tasks_30d": 20,
                "balance": 3.5,
                "last_interaction": "2026-02-10T12:00:00Z",
            },
            {
                "provider_node_id": "node2",
                "success_rate": 0.80,
                "avg_wall_time_ms": 400,
                "total_tasks_30d": 5,
                "balance": -1.0,
                "last_interaction": "2026-02-09T08:00:00Z",
            },
        ]

        def handler(request: httpx.Request):
            assert request.url.path == "/api/reputation"
            return _json_response(rep_data)

        client = make_client(handler)
        result = await client.get_reputation()
        assert len(result) == 2
        assert result[0]["provider_node_id"] == "node1"
        assert result[0]["success_rate"] == 0.95
        assert result[1]["balance"] == -1.0

    @pytest.mark.asyncio
    async def test_handles_wrapped_response(self, make_client):
        """Some API versions may wrap the list in a dict."""
        def handler(request: httpx.Request):
            return _json_response({"reputations": [{"provider_node_id": "n1"}]})

        client = make_client(handler)
        result = await client.get_reputation()
        assert len(result) == 1
        assert result[0]["provider_node_id"] == "n1"

    @pytest.mark.asyncio
    async def test_empty_reputation(self, make_client):
        def handler(request: httpx.Request):
            return _json_response([])

        client = make_client(handler)
        result = await client.get_reputation()
        assert result == []


# ── GET /api/skills/{name}/schema ─────────────────────────────────

class TestGetSkillSchema:
    @pytest.mark.asyncio
    async def test_returns_schema(self, make_client):
        schema_data = {
            "name": "echo",
            "description": "Echo back",
            "input_schema": {"text": "string"},
            "providers": [
                {"node_id": "abc", "host": "1.2.3.4", "port": 9200, "load": 2},
            ],
            "price": 0.5,
        }

        def handler(request: httpx.Request):
            assert request.url.path == "/api/skills/echo/schema"
            return _json_response(schema_data)

        client = make_client(handler)
        result = await client.get_skill_schema("echo")
        assert result["name"] == "echo"
        assert result["providers"][0]["load"] == 2
        assert result["price"] == 0.5

    @pytest.mark.asyncio
    async def test_404_for_unknown_skill(self, make_client):
        def handler(request: httpx.Request):
            return _error_response(404, "Skill not found")

        client = make_client(handler)
        with pytest.raises(KnarrAPIError) as exc_info:
            await client.get_skill_schema("nonexistent")
        assert exc_info.value.status_code == 404


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close(self, make_client):
        def handler(request: httpx.Request):
            return _json_response({})

        client = make_client(handler)
        await client.close()
        # Should not raise on double-close
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, make_client):
        def handler(request: httpx.Request):
            return _json_response({"node_id": "x"})

        client = make_client(handler)
        async with client as c:
            result = await c.get_status()
        assert result["node_id"] == "x"
