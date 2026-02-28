# Core Modules

Shared agent framework for Knarr channel adapters. Contains the channel-agnostic logic that powers any bot (Telegram, Discord, Slack, etc.) connected to the Knarr P2P network.

## Modules

| Module | Purpose |
|--------|---------|
| `bus.py` | Message types (`InboundMessage`, `SendFn`) shared by all channels |
| `agent_core.py` | Command routing, LLM dispatch, skill execution, access control, subagent spawning |
| `knarr_client.py` | Typed async HTTP client for the Knarr Cockpit REST API |
| `llm_router.py` | Multi-provider LLM agent (Gemini primary, LiteLLM fallback) with progressive skill loading |
| `chat_store.py` | Chat history storage (SQLite) |
| `memory_store.py` | Persistent memory -- facts + daily notes (SQLite) |
| `session_store.py` | LLM conversation persistence across restarts (SQLite) |
| `cron_store.py` | Scheduled task storage with cron expressions (SQLite) |

## Personality & Instructions

- `PERSONALITY.md` -- base bot persona (loaded by `llm_router.py` at startup)
- `INSTRUCTIONS.md` -- LLM behavior guidelines and tool usage rules

Channel adapters can override these by passing a custom `base_dir` to `build_system_prompt()`.
