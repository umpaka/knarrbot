"""Tests for agent_core.py — command routing and pure functions."""

import json
import os
import time

import pytest

from helpers import MockTaskResult, MockKnarrClient


# ── Pure function tests ──────────────────────────────────────────

class TestParseCommand:
    def test_simple_command(self):
        from agent_core import parse_command
        cmd, args = parse_command("/help")
        assert cmd == "/help"
        assert args == ""

    def test_command_with_args(self):
        from agent_core import parse_command
        cmd, args = parse_command('/run echo {"text": "hi"}')
        assert cmd == "/run"
        assert args == 'echo {"text": "hi"}'

    def test_command_with_botname(self):
        from agent_core import parse_command
        cmd, args = parse_command("/help@knarrbot")
        assert cmd == "/help"
        assert args == ""

    def test_not_a_command(self):
        from agent_core import parse_command
        cmd, args = parse_command("Hello there")
        assert cmd == ""
        assert args == "Hello there"

    def test_empty_string(self):
        from agent_core import parse_command
        cmd, args = parse_command("")
        assert cmd == ""

    def test_none(self):
        from agent_core import parse_command
        cmd, args = parse_command(None)
        assert cmd == ""


class TestFormatUptime:
    def test_seconds(self):
        from agent_core import format_uptime
        assert format_uptime(45) == "45s"

    def test_minutes(self):
        from agent_core import format_uptime
        assert format_uptime(125) == "2m 5s"

    def test_hours(self):
        from agent_core import format_uptime
        assert format_uptime(3725) == "1h 2m"

    def test_zero(self):
        from agent_core import format_uptime
        assert format_uptime(0) == "0s"


class TestFormatSkillResult:
    def test_completed(self):
        from agent_core import format_skill_result
        result = MockTaskResult(
            status="completed",
            output_data={"summary": "All good"},
            error={},
        )
        text = format_skill_result(result)
        assert "completed" in text
        assert "All good" in text

    def test_completed_truncates_long_values(self):
        from agent_core import format_skill_result
        result = MockTaskResult(
            status="completed",
            output_data={"data": "x" * 2000},
            error={},
        )
        text = format_skill_result(result)
        assert "..." in text or len(text) < 2000

    def test_failed(self):
        from agent_core import format_skill_result
        result = MockTaskResult(
            status="failed",
            output_data={},
            error={"code": "TIMEOUT", "message": "Timed out after 30s"},
        )
        text = format_skill_result(result)
        assert "TIMEOUT" in text

    def test_failed_no_error_detail(self):
        from agent_core import format_skill_result
        result = MockTaskResult(status="failed", output_data={}, error={})
        text = format_skill_result(result)
        assert "UNKNOWN" in text

    def test_completed_dict(self):
        """format_skill_result also accepts plain dicts (from KnarrClient)."""
        from agent_core import format_skill_result
        result = {
            "status": "completed",
            "output_data": {"summary": "dict result"},
            "error": {},
        }
        text = format_skill_result(result)
        assert "completed" in text
        assert "dict result" in text

    def test_failed_dict(self):
        from agent_core import format_skill_result
        result = {
            "status": "failed",
            "output_data": {},
            "error": {"code": "NET_ERR", "message": "timeout"},
        }
        text = format_skill_result(result)
        assert "NET_ERR" in text


# ── Agent command dispatch tests ─────────────────────────────────

@pytest.fixture
def agent(mock_client, send_fn, memory_store, cron_store):
    """An AgentCore wired to mocks."""
    from agent_core import AgentCore
    return AgentCore(
        client=mock_client,
        llm_router=None,  # no LLM in these tests
        chat_store=None,
        cron_store=cron_store,
        memory_store=memory_store,
        send_fn=send_fn,
        bot_info={"start_time": time.time() - 3600},
    )


def make_msg(text, chat_id=1, **kwargs):
    from bus import InboundMessage
    return InboundMessage(channel="test", chat_id=chat_id, text=text, **kwargs)


class TestCommandDispatch:
    @pytest.mark.asyncio
    async def test_help(self, agent, send_fn):
        await agent.process_message(make_msg("/help"))
        assert len(send_fn.messages) == 1
        assert "Commands" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_start(self, agent, send_fn):
        await agent.process_message(make_msg("/start"))
        assert "Commands" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_reset_without_llm(self, agent, send_fn):
        await agent.process_message(make_msg("/reset"))
        assert "Fresh start" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_cron_empty(self, agent, send_fn):
        await agent.process_message(make_msg("/cron"))
        assert "No scheduled tasks" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_memory_empty(self, agent, send_fn):
        await agent.process_message(make_msg("/memory"))
        assert "No stored memories" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_memory_with_facts(self, agent, send_fn, memory_store):
        memory_store.save_fact(1, "test_key", "test_value")
        await agent.process_message(make_msg("/memory"))
        assert "test_key" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_status(self, agent, send_fn):
        await agent.process_message(make_msg("/status"))
        text = send_fn.last_text
        assert "Knarr Gateway Status" in text
        assert "Uptime" in text

    @pytest.mark.asyncio
    async def test_skills_empty(self, agent, send_fn, mock_client):
        mock_client._skills = {"local": [], "network": []}
        await agent.process_message(make_msg("/skills"))
        assert "No skills found" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_skills_with_results(self, agent, send_fn, mock_client):
        mock_client._skills = {
            "local": [],
            "network": [
                {"name": "echo", "description": "Echo back input", "providers": []},
                {"name": "browse-web", "description": "Browse the web", "providers": []},
            ],
        }
        await agent.process_message(make_msg("/skills"))
        assert "echo" in send_fn.last_text
        assert "browse-web" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_unknown_command(self, agent, send_fn):
        await agent.process_message(make_msg("/foobar"))
        assert "Unknown command" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_run_no_args_no_context(self, agent, send_fn):
        await agent.process_message(make_msg("/run"))
        assert "Usage" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_run_skill_not_found(self, agent, send_fn, mock_client):
        mock_client._skills = {"local": [], "network": []}
        await agent.process_message(make_msg("/run nonexistent"))
        assert "No providers found" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_run_skill_success(self, agent, send_fn, mock_client):
        mock_client._skills = {
            "local": [],
            "network": [
                {
                    "name": "echo",
                    "description": "Echo back",
                    "providers": [
                        {"node_id": "abc", "host": "127.0.0.1", "port": 9200},
                    ],
                }
            ],
        }
        mock_client._execute_result = {
            "status": "completed",
            "output_data": {"text": "echoed"},
            "error": {},
            "wall_time_ms": 10,
        }
        await agent.process_message(make_msg('/run echo {"text": "hi"}'))
        texts = send_fn.texts
        assert any("Running" in t for t in texts)
        assert any("echoed" in t for t in texts)

    @pytest.mark.asyncio
    async def test_natural_language_no_llm(self, agent, send_fn):
        await agent.process_message(make_msg("What is the weather?"))
        assert "I only understand /commands" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_natural_language_no_response_in_group(self, agent, send_fn):
        await agent.process_message(make_msg("random chat", is_group=True))
        # No LLM and is_group → should NOT send anything
        assert len(send_fn.messages) == 0

    @pytest.mark.asyncio
    async def test_reset_mentions_memories(self, agent, send_fn):
        await agent.process_message(make_msg("/reset"))
        assert "memories kept" in send_fn.last_text.lower() or "reset all" in send_fn.last_text.lower()

    @pytest.mark.asyncio
    async def test_reset_all_clears_memories(self, agent, send_fn, memory_store):
        memory_store.save_fact(1, "pref", "French accent")
        await agent.process_message(make_msg("/reset all"))
        assert "1 memories wiped" in send_fn.last_text
        assert memory_store.get_facts(1) == []

    @pytest.mark.asyncio
    async def test_memory_clear(self, agent, send_fn, memory_store):
        memory_store.save_fact(1, "key1", "val1")
        memory_store.save_fact(1, "key2", "val2")
        await agent.process_message(make_msg("/memory clear"))
        assert "2 memories" in send_fn.last_text
        assert memory_store.get_facts(1) == []

    @pytest.mark.asyncio
    async def test_doctor(self, agent, send_fn, mock_client):
        mock_client._skills = {"local": [], "network": []}
        await agent.process_message(make_msg("/doctor"))
        text = send_fn.last_text
        assert "Doctor Report" in text
        assert "Services" in text
        assert "API Keys" in text

    @pytest.mark.asyncio
    async def test_doctor_shows_services(self, agent, send_fn, mock_client):
        mock_client._skills = {
            "local": [],
            "network": [
                {
                    "name": "web-search",
                    "description": "Search the web",
                    "providers": [{"node_id": "x", "host": "127.0.0.1", "port": 9300}],
                },
            ],
        }
        await agent.process_message(make_msg("/doctor"))
        text = send_fn.last_text
        assert "web-search" in text

    @pytest.mark.asyncio
    async def test_help_includes_doctor(self, agent, send_fn):
        await agent.process_message(make_msg("/help"))
        assert "/doctor" in send_fn.last_text


# ── Pairing tests ────────────────────────────────────────────────

class TestPairing:
    """Test the /pair, /unpair, and pairing code redemption flow."""

    @pytest.fixture(autouse=True)
    def setup_access_control(self, tmp_path, monkeypatch):
        """Set ALLOWED_USERS so access control is active, and use temp paired_users file."""
        import agent_core
        monkeypatch.setenv("ALLOWED_USERS", "100")
        monkeypatch.setattr(agent_core, "PAIRED_USERS_FILE", str(tmp_path / "paired.json"))
        agent_core.reload_access_lists()
        # Clear any leftover pairing codes
        agent_core._pairing_codes.clear()
        yield
        # Restore open access after test
        monkeypatch.delenv("ALLOWED_USERS", raising=False)
        agent_core.reload_access_lists()

    @pytest.fixture
    def restricted_agent(self, mock_client, send_fn, memory_store, cron_store):
        from agent_core import AgentCore
        return AgentCore(
            client=mock_client,
            llm_router=None,
            chat_store=None,
            cron_store=cron_store,
            memory_store=memory_store,
            send_fn=send_fn,
            bot_info={"start_time": time.time()},
        )

    @pytest.mark.asyncio
    async def test_pair_requires_admin(self, restricted_agent, send_fn):
        # User 999 is NOT in ALLOWED_USERS (100)
        # They can't even reach /pair — access_check blocks them
        await restricted_agent.process_message(make_msg("/pair", user_id=999))
        # Access denied = silent ignore, no messages sent
        assert len(send_fn.messages) == 0

    @pytest.mark.asyncio
    async def test_pair_generates_code(self, restricted_agent, send_fn):
        await restricted_agent.process_message(make_msg("/pair", user_id=100))
        text = send_fn.last_text
        assert "Pairing code" in text
        assert "5 minutes" in text

    @pytest.mark.asyncio
    async def test_full_pairing_flow(self, restricted_agent, send_fn):
        import agent_core

        # Admin generates code
        await restricted_agent.process_message(make_msg("/pair", user_id=100))
        # Extract the code from the response
        text = send_fn.last_text
        code = None
        for word in text.split("`"):
            if word.isdigit() and len(word) == 6:
                code = word
                break
        assert code is not None, f"Could not find 6-digit code in: {text}"

        send_fn.clear()

        # Unknown user sends the code as a DM
        await restricted_agent.process_message(make_msg(code, user_id=200))
        assert "paired successfully" in send_fn.last_text

        # The user should now be in the paired set
        assert 200 in agent_core._paired_users

        send_fn.clear()

        # The paired user can now use the bot
        await restricted_agent.process_message(make_msg("/help", user_id=200))
        assert "Commands" in send_fn.last_text

    @pytest.mark.asyncio
    async def test_unpair_lists_users(self, restricted_agent, send_fn):
        import agent_core
        agent_core._paired_users = {200, 300}

        await restricted_agent.process_message(make_msg("/unpair", user_id=100))
        text = send_fn.last_text
        assert "200" in text
        assert "300" in text

    @pytest.mark.asyncio
    async def test_unpair_removes_user(self, restricted_agent, send_fn, tmp_path):
        import agent_core
        agent_core._paired_users = {200}

        await restricted_agent.process_message(make_msg("/unpair 200", user_id=100))
        assert "unpaired" in send_fn.last_text.lower()
        assert 200 not in agent_core._paired_users

    @pytest.mark.asyncio
    async def test_invalid_code_ignored(self, restricted_agent, send_fn):
        # Unknown user sends a random 6-digit number that's not a valid code
        await restricted_agent.process_message(make_msg("123456", user_id=999))
        # Should be silently ignored (access denied, invalid code)
        assert len(send_fn.messages) == 0

    @pytest.mark.asyncio
    async def test_expired_code_rejected(self, restricted_agent, send_fn):
        import agent_core

        # Manually insert an expired code
        agent_core._pairing_codes["654321"] = {
            "admin_id": 100,
            "expires": time.time() - 10,  # expired 10s ago
        }

        await restricted_agent.process_message(make_msg("654321", user_id=999))
        # Expired code → not redeemed → access denied → silent
        assert len(send_fn.messages) == 0
