# knarrbot

Autonomous agent for the [Knarr](https://github.com/knarrnet/knarr) P2P network. Contains channel-agnostic core logic and channel adapters (currently Telegram, more planned).

## Structure

```
knarrbot/
├── README.md
├── knarr.toml              # Knarr node config (port 9100, skill definitions)
├── .env / .env.example     # Secrets and configuration
├── heartbeat.md            # Standing instructions for periodic heartbeat checks
├── core/                   # Shared agent framework (all channels)
│   ├── bus.py              # Message types (InboundMessage, SendFn)
│   ├── agent_core.py       # Command routing, LLM dispatch, skill execution
│   ├── knarr_client.py     # HTTP client for Cockpit REST API
│   ├── llm_router.py       # Multi-provider LLM agent (Gemini + LiteLLM fallback)
│   ├── chat_store.py       # Chat history (SQLite)
│   ├── memory_store.py     # Persistent memory (SQLite)
│   ├── session_store.py    # LLM conversation persistence (SQLite)
│   ├── cron_store.py       # Scheduled tasks (SQLite)
│   ├── PERSONALITY.md      # Base bot persona
│   └── INSTRUCTIONS.md     # LLM behavior guidelines
├── adapters/
│   └── telegram/           # Telegram channel adapter
│       ├── telegram_gateway.py   # Main entry point (polling, media, typing, mail/email pollers)
│       ├── telegram_format.py    # Markdown-to-Telegram-HTML converter
│       ├── send_telegram.py      # Knarr skill: send message
│       └── fetch_telegram.py     # Knarr skill: fetch messages
└── tests/                  # Full test suite
```

## How it works

```
Channel Adapter ──► InboundMessage ──► AgentCore ──► LLMRouter ──► KnarrClient ──► Cockpit API
                                          │                              │
                                          ├── Commands (/help, /run)     ├── Skill discovery
                                          ├── LLM routing                ├── Task execution
                                          ├── Cron jobs                  ├── knarr-mail
                                          └── Heartbeat                  └── Asset transfer
```

The agent communicates with the Knarr network exclusively through `KnarrClient` -- no direct `DHTNode` or internal knarr imports. The bot points to an external Knarr node's Cockpit API via `KNARR_API_URL` and `KNARR_API_TOKEN`.

## Setup

1. Copy the env template and fill in your values:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies (from the repo root):
   ```bash
   pip install -r ../requirements.txt
   ```

3. Run the Telegram bot:
   ```bash
   python adapters/telegram/telegram_gateway.py
   ```

4. Optionally, run the skill node to expose `send-telegram` and `fetch-telegram` on the network:
   ```bash
   knarr serve
   ```

## Knowledge Vault (highly recommended)

knarrbot is designed to use the [umpaka/vault](https://github.com/umpaka/vault) skill as its long-term memory. Without it the bot operates stateless — no goals, no reasoning continuity, no contact book, no economic ledger.

Install it alongside knarrbot on the same knarr node:

```bash
git clone https://github.com/umpaka/vault /opt/knarr-skills/vault
cp /opt/knarr-skills/vault/.env.example /opt/knarr-skills/vault/.env
# Set KNARR_NODE_ID and VAULT_ROOT in .env
```

Once installed, seed the vault structure the bot expects:

```bash
mkdir -p /opt/knarr-vault/default/goals /opt/knarr-vault/default/scratch

# Heartbeat protocol (agent can self-modify this)
cp heartbeat.md /opt/knarr-vault/default/goals/heartbeat.md

# Starter goals (agent updates these every cycle)
cat > /opt/knarr-vault/default/goals/active.md << 'EOF'
# My Active Goals
- [ ] Map the network: list_peers, note interesting nodes in notes/network-map.md
- [ ] Make first contact: introduce myself to one peer via knarr-mail
- [ ] Understand my economy: check /economy, log balance to economy/ledger.md
EOF

# Bootstrap reasoning continuity
echo "Fresh start — no prior context." > /opt/knarr-vault/default/scratch/current-thinking.md
```

Set `VAULT_ROOT` in your `.env` to match the directory above (default: `/opt/knarr-vault`).

The vault also needs to be registered as a skill in your `knarr.toml`. See the [vault README](https://github.com/umpaka/vault) for details.

**Why it matters:** knarrbot's heartbeat loop reads `goals/heartbeat.md` each cycle and injects `scratch/current-thinking.md` as prior context. Without these files, every heartbeat starts from zero and returns immediately. With them, the agent has a persistent agenda, continuity of reasoning across restarts, and an economic self-model.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `KNARR_API_URL` | Yes | Cockpit API base URL (e.g. `http://localhost:9100`) |
| `KNARR_API_TOKEN` | No | Cockpit API bearer token |
| `GEMINI_API_KEY` | No | Google Gemini API key (enables LLM routing) |
| `FALLBACK_MODEL` | No | LiteLLM model string for fallback (e.g. `openrouter/anthropic/claude-sonnet-4`) |
| `FALLBACK_API_KEY` | No | API key for the fallback provider |
| `ALLOWED_USERS` | No | Comma-separated Telegram user IDs (empty = open access) |
| `ALLOWED_GROUPS` | No | Comma-separated Telegram group chat IDs |
| `HEARTBEAT_CHAT_ID` | No | Chat ID for proactive heartbeat messages |
| `HEARTBEAT_INTERVAL` | No | Seconds between heartbeat checks (default: 1800) |
| `MAIL_POLL_INTERVAL` | No | Seconds between knarr-mail inbox checks (default: 10) |
| `POSTMASTER_DB` | No | Path to postmaster SQLite DB for email polling |
| `EMAIL_POLL_INTERVAL` | No | Seconds between postmaster email checks (default: 15) |
| `VAULT_ROOT` | No | Root directory for vault data (default: `/opt/knarr-vault`) |
| `FAST_LLM_MODEL` | No | LiteLLM model string for heartbeats (cheaper/faster than primary) |

## Building a new channel adapter

To add a new channel (e.g. Discord), create `adapters/discord/` and write an adapter that:

1. **Adds core and adapter dir to `sys.path`:**
   ```python
   import os, sys
   _HERE = os.path.dirname(os.path.abspath(__file__))
   _KNARRBOT_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
   sys.path.insert(0, os.path.join(_KNARRBOT_DIR, "core"))
   sys.path.insert(0, _HERE)  # so adapter-local imports work
   ```

2. **Creates a `KnarrClient`:**
   ```python
   from knarr_client import KnarrClient
   client = KnarrClient(os.environ["KNARR_API_URL"], os.environ["KNARR_API_TOKEN"])
   ```

3. **Creates an `AgentCore` with a send callback:**
   ```python
   from agent_core import AgentCore
   agent = AgentCore(
       client=client,
       llm_router=llm_router,  # or None for command-only
       chat_store=chat_store,
       cron_store=cron_store,
       memory_store=memory_store,
       send_fn=your_send_fn,       # async def send(chat_id, text, parse_mode="")
       send_file_fn=your_file_fn,  # async def send_file(chat_id, bytes, filename, caption="")
       bot_info={"start_time": time.time()},
   )
   ```

4. **Converts incoming messages to `InboundMessage` and routes through the agent:**
   ```python
   from bus import InboundMessage
   msg = InboundMessage(channel="discord", chat_id=channel_id, text=text, ...)
   await agent.process_message(msg)
   ```

That's it. All command handling, LLM routing, skill execution, memory, and cron work automatically.

## Tests

```bash
cd knarrbot && python -m pytest tests/ -v
```
