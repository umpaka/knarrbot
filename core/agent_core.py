"""Channel-agnostic agent core.

Processes InboundMessage objects from any channel (Telegram, Discord, etc.),
handles commands, routes to the LLM, executes skills, and sends responses
via the send callback. This module contains zero channel-specific code.

The agent communicates with the Knarr network exclusively through the
KnarrClient HTTP wrapper — no direct DHTNode imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

from bus import InboundMessage, SendFn
from knarr_client import KnarrClient

log = logging.getLogger("agent")


# ── Access control ────────────────────────────────────────────────

PAIRED_USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paired_users.json",
)


def load_access_list(env_var: str) -> set[int]:
    """Load a set of allowed IDs from a comma-separated env var."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return set()
    ids = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            try:
                ids.add(int(item))
            except ValueError:
                log.warning("Invalid ID in %s: %s", env_var, item)
    return ids


def load_paired_users() -> set[int]:
    """Load paired user IDs from the persistent JSON file."""
    if not os.path.exists(PAIRED_USERS_FILE):
        return set()
    try:
        with open(PAIRED_USERS_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("users", []))
    except Exception:
        log.warning("Failed to load paired_users.json, starting fresh")
        return set()


def save_paired_users(users: set[int]):
    """Persist paired user IDs to JSON file."""
    try:
        with open(PAIRED_USERS_FILE, "w") as f:
            json.dump({"users": sorted(users)}, f)
    except Exception:
        log.exception("Failed to save paired_users.json")


# Loaded once at import time; reload by calling reload_access_lists()
_allowed_users: set[int] = set()
_allowed_groups: set[int] = set()
_paired_users: set[int] = set()

# Active pairing codes: {code_str: {"user_id": admin_id, "expires": timestamp}}
_pairing_codes: dict[str, dict] = {}


def reload_access_lists():
    """Reload access lists from environment variables + paired users file."""
    global _allowed_users, _allowed_groups, _paired_users
    _allowed_users = load_access_list("ALLOWED_USERS")
    _allowed_groups = load_access_list("ALLOWED_GROUPS")
    _paired_users = load_paired_users()
    if _allowed_users or _allowed_groups:
        log.info("Access control: %d allowed users, %d allowed groups, %d paired users",
                 len(_allowed_users), len(_allowed_groups), len(_paired_users))
    else:
        log.info("Access control: open (no ALLOWED_USERS/ALLOWED_GROUPS set)")


# Initialize on import
reload_access_lists()


def is_admin(user_id: int) -> bool:
    """Check if a user is an admin (in the ALLOWED_USERS env var list)."""
    return bool(_allowed_users and user_id in _allowed_users)


def access_check(msg: InboundMessage) -> bool:
    """Check if a message sender is allowed to use the bot.

    Rules:
    - If neither ALLOWED_USERS nor ALLOWED_GROUPS is set → open access (allow all)
    - If ALLOWED_USERS is set → user's numeric ID must be in the list OR in paired users
    - If ALLOWED_GROUPS is set → group's chat_id must be in the list
    - In groups: the group must be allowed AND (if ALLOWED_USERS is set) the user too

    This function is designed to be replaceable — e.g., swap in a crypto payment
    check when Knarr adds that feature.
    """
    if not _allowed_users and not _allowed_groups:
        return True  # Open access

    all_allowed = _allowed_users | _paired_users

    if msg.is_group:
        if _allowed_groups and msg.chat_id not in _allowed_groups:
            return False
        # In groups, if ALLOWED_USERS is set, also check the sender
        if all_allowed and msg.user_id and msg.user_id not in all_allowed:
            return False
        return True
    else:
        # DMs: check user against env + paired
        if all_allowed and msg.user_id not in all_allowed:
            return False
        return True


def parse_command(text: str) -> tuple[str, str]:
    """Parse a /command from message text. Returns (command, args)."""
    if not text or not text.startswith("/"):
        return "", text or ""
    parts = text.split(None, 1)
    cmd = parts[0].lower().split("@")[0]  # strip @botname suffix
    args = parts[1] if len(parts) > 1 else ""
    return cmd, args


def _parse_generated_personality(raw: str) -> tuple[str, str]:
    """Parse PERSONALITY: / INSTRUCTIONS: sections from LLM output."""
    personality = ""
    instructions = ""
    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PERSONALITY:"):
            current = "p"
            rest = stripped[len("PERSONALITY:"):].strip()
            if rest:
                personality += rest + "\n"
        elif stripped.upper().startswith("INSTRUCTIONS:"):
            current = "i"
            rest = stripped[len("INSTRUCTIONS:"):].strip()
            if rest:
                instructions += rest + "\n"
        elif current == "p":
            personality += line + "\n"
        elif current == "i":
            instructions += line + "\n"
    return personality.strip(), instructions.strip()


def format_skill_result(result: Any) -> str:
    """Format a skill execution result into a Markdown message.

    Accepts either a dict (from KnarrClient.execute) or a legacy TaskResult
    object with .status / .output_data / .error attributes.
    """
    # Normalise to dict
    if isinstance(result, dict):
        status = result.get("status", "unknown")
        output_data = result.get("output_data", {})
        error = result.get("error", {})
    else:
        status = getattr(result, "status", "unknown")
        output_data = getattr(result, "output_data", {}) or {}
        error = getattr(result, "error", {}) or {}

    if status == "completed":
        lines = ["*Status:* completed"]
        for k, v in output_data.items():
            value = str(v)
            if len(value) > 800:
                value = value[:800] + "... (truncated)"
            lines.append(f"*{k}:*\n{value}")
        return "\n\n".join(lines)
    else:
        code = error.get("code", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
        msg = error.get("message", "No detail") if isinstance(error, dict) else str(error)
        return f"*Status:* {status}\n*Error:* {code} — {msg}"


def format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60}s"
    else:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h}h {m}m"


class AgentCore:
    """Channel-agnostic agent that processes messages and responds."""

    def __init__(
        self,
        client: KnarrClient,
        llm_router: Any = None,
        chat_store: Any = None,
        cron_store: Any = None,
        memory_store: Any = None,
        send_fn: SendFn = None,
        bot_info: dict | None = None,
        send_file_fn: Any = None,
        status_send_fn: Any = None,
    ):
        """
        Args:
            client: KnarrClient HTTP wrapper for skill discovery/execution.
            llm_router: LLM router for natural language processing.
            chat_store: Chat history store (message logging).
            cron_store: Scheduled task store.
            memory_store: Persistent memory store (facts + notes).
            send_fn: Callback to send messages: async def(chat_id, text, parse_mode="")
            bot_info: Dict with bot metadata (start_time, bot_username, etc.)
            send_file_fn: Callback to send files: async def(chat_id, file_bytes, filename, caption="")
            status_send_fn: Callback for edit-in-place status updates: async def(chat_id, text)
        """
        self.client = client
        self.llm_router = llm_router
        self.chat_store = chat_store
        self.cron_store = cron_store
        self.memory_store = memory_store
        self.send = send_fn
        self.send_file = send_file_fn
        self.send_status = status_send_fn
        self.bot_info = bot_info or {}

        # Per-chat context: tracks last skill used for /run re-run
        self._chat_context: dict[int, dict] = {}

        # Running background tasks: {task_id: {name, chat_id, status, started_at, asyncio_task}}
        self._running_tasks: dict[int, dict] = {}
        self._next_task_id = 1

        # Active per-user tasks: (chat_id, user_id) -> asyncio.Task
        # Used for cancel-and-replace: if the same user sends a new message
        # while one is being processed, the old task is cancelled.
        self._active_user_tasks: dict[tuple[int, int], asyncio.Task] = {}

        # In-progress /configure wizard sessions: chat_id -> state dict
        # state keys: step, custom_personality
        self._configure_sessions: dict[int, dict] = {}

        # Wire up callbacks so the LLM can spawn tasks and send files
        if self.llm_router:
            self.llm_router._spawn_callback = self._spawn_task
            self.llm_router._send_file_fn = self.send_file

    def cancel_user_task(self, chat_id: int, user_id: int) -> bool:
        """Cancel the active task for a specific user in a chat.

        Returns True if a task was cancelled, False if none was running.
        """
        key = (chat_id, user_id)
        task = self._active_user_tasks.get(key)
        if task and not task.done():
            task.cancel()
            del self._active_user_tasks[key]
            log.info("Cancelled active task for user %d in chat %d", user_id, chat_id)
            return True
        return False

    def register_user_task(self, chat_id: int, user_id: int, task: asyncio.Task):
        """Register the active task for a user. Cancels any previous task first."""
        key = (chat_id, user_id)
        old = self._active_user_tasks.get(key)
        if old and not old.done():
            old.cancel()
            log.info("Cancel-and-replace: cancelled previous task for user %d in chat %d", user_id, chat_id)
        self._active_user_tasks[key] = task

    def unregister_user_task(self, chat_id: int, user_id: int):
        """Remove the user's task from tracking (called when task completes)."""
        self._active_user_tasks.pop((chat_id, user_id), None)

    async def process_message(self, msg: InboundMessage):
        """Process an inbound message from any channel."""
        # Before access control, check for pairing code redemption from unknown DMs.
        # This allows an unauthorized user to redeem a code and gain access.
        if not access_check(msg):
            if await self._try_redeem_pairing_code(msg):
                return  # Code redeemed — access granted
            log.info("Access denied for user %d in chat %d", msg.user_id, msg.chat_id)
            return  # Silently ignore unauthorized messages

        cmd, args = parse_command(msg.text)

        # If a /configure wizard session is active for this chat, intercept input
        if msg.chat_id in self._configure_sessions and cmd != "/configure":
            await self._configure_step(msg)
            return

        if cmd == "/help" or cmd == "/start":
            await self._cmd_help(msg)
        elif cmd == "/configure":
            await self._cmd_configure(msg)
        elif cmd == "/cancel":
            await self._cmd_cancel(msg)
        elif cmd == "/reset":
            await self._cmd_reset(msg, args)
        elif cmd == "/cron":
            await self._cmd_cron(msg)
        elif cmd == "/memory":
            await self._cmd_memory(msg, args)
        elif cmd == "/tasks":
            await self._cmd_tasks(msg)
        elif cmd == "/status":
            await self._cmd_status(msg)
        elif cmd == "/doctor":
            await self._cmd_doctor(msg)
        elif cmd == "/pair":
            await self._cmd_pair(msg)
        elif cmd == "/unpair":
            await self._cmd_unpair(msg, args)
        elif cmd == "/skills":
            await self._cmd_skills(msg)
        elif cmd == "/run":
            await self._cmd_run(msg, args)
        elif cmd:
            await self.send(
                msg.chat_id,
                f"Unknown command: `{cmd}`\nType /help for available commands.",
                "Markdown",
            )
        else:
            await self._route_to_llm(msg)

    # ── Commands ─────────────────────────────────────────────────────

    async def _cmd_cancel(self, msg: InboundMessage):
        """Cancel the calling user's active task or an open /configure session."""
        # Also clear any open configure wizard
        if msg.chat_id in self._configure_sessions:
            del self._configure_sessions[msg.chat_id]
            await self.send(msg.chat_id, "Configuration cancelled.")
            return
        cancelled = self.cancel_user_task(msg.chat_id, msg.user_id)
        if cancelled:
            await self.send(msg.chat_id, "Cancelled your active request.")
        else:
            await self.send(msg.chat_id, "No active request to cancel.")

    async def _cmd_help(self, msg: InboundMessage):
        help_text = (
            "*Knarr Gateway*\n\n"
            "*Commands:*\n"
            "  /configure — Customize your agent's personality and role\n"
            "  /skills — List skills on the network\n"
            "  /run `<skill>` `<input>` — Execute a skill\n"
            "  /run — Re-run last skill\n"
            "  /cancel — Cancel your active request\n"
            "  /cron — List scheduled tasks\n"
            "  /tasks — List background tasks\n"
            "  /memory — Show stored memories\n"
            "  /memory clear — Wipe all memories\n"
            "  /reset — Clear conversation history\n"
            "  /reset all — Clear history + wipe memories\n"
            "  /status — Node info and uptime\n"
            "  /doctor — Diagnose services, API keys, and health\n"
            "  /pair — Generate a pairing code (admin only)\n"
            "  /unpair `<user_id>` — Revoke a paired user (admin only)\n"
            "  /help — Show this message\n\n"
            "*Supported inputs:*\n"
            "  Text, images, PDFs, text files, voice messages\n\n"
            "*Examples:*\n"
            '  `/run echo {"text": "hello"}`\n'
            "  /skills\n"
        )
        if self.llm_router:
            help_text += (
                "\nYou can also type questions in natural language, send voice messages, "
                "images, or files. Ask me to schedule recurring tasks or remember things!"
            )
        await self.send(msg.chat_id, help_text, "Markdown")

    async def _cmd_reset(self, msg: InboundMessage, args: str = ""):
        if args.strip().lower() == "all":
            # Clear everything: conversation history + persistent memory
            if self.llm_router:
                self.llm_router.clear_history(msg.chat_id)
            cleared = 0
            if self.memory_store:
                cleared = self.memory_store.clear_all(msg.chat_id)
            await self.send(
                msg.chat_id,
                f"Full reset: conversation history cleared + {cleared} memories wiped. Clean slate!",
            )
        else:
            if self.llm_router:
                self.llm_router.clear_history(msg.chat_id)
            await self.send(
                msg.chat_id,
                "Conversation history cleared. Fresh start!\n"
                "(Persistent memories kept. Use /reset all to also wipe memories.)",
            )

    async def _cmd_cron(self, msg: InboundMessage):
        if self.cron_store:
            text = self.cron_store.format_jobs_text(msg.chat_id)
            await self.send(msg.chat_id, text)
        else:
            await self.send(msg.chat_id, "Scheduled tasks not available.")

    async def _cmd_memory(self, msg: InboundMessage, args: str = ""):
        if not self.memory_store:
            await self.send(msg.chat_id, "Memory not available.")
            return

        subcmd = args.strip().lower()
        if subcmd == "clear":
            cleared = self.memory_store.clear_all(msg.chat_id)
            await self.send(msg.chat_id, f"Cleared {cleared} memories. All forgotten!")
        else:
            text = self.memory_store.format_facts_text(msg.chat_id)
            await self.send(msg.chat_id, text)

    async def _cmd_status(self, msg: InboundMessage):
        uptime = format_uptime(time.time() - self.bot_info.get("start_time", 0))
        llm_status = "connected" if self.llm_router else "not configured"

        # Fetch node status via Cockpit API
        try:
            node_status = await self.client.get_status()
        except Exception as e:
            log.warning("Failed to fetch node status: %s", e)
            node_status = {}

        node_id_short = node_status.get("node_id", "?")[:16]
        node_port = node_status.get("port", "?")
        peer_count = node_status.get("peer_count", 0)
        own_skills = node_status.get("skill_count", 0)

        cron_count = 0
        if self.cron_store:
            cron_count = len(self.cron_store.list_jobs(msg.chat_id))

        memory_count = 0
        if self.memory_store:
            memory_count = len(self.memory_store.get_facts(msg.chat_id))

        active_tasks = sum(1 for t in self._running_tasks.values()
                          if t["status"] == "running" and t["chat_id"] == msg.chat_id)

        status_text = (
            f"*Knarr Gateway Status*\n\n"
            f"*Node ID:* `{node_id_short}...`\n"
            f"*Port:* {node_port}\n"
            f"*Peers:* {peer_count}\n"
            f"*Local skills:* {own_skills}\n"
            f"*Uptime:* {uptime}\n"
            f"*LLM agent:* {llm_status}\n"
            f"*Scheduled tasks:* {cron_count}\n"
            f"*Background tasks:* {active_tasks}\n"
            f"*Stored memories:* {memory_count}"
        )

        # ── Task queue status ─────────────────────────────────────────
        try:
            task_slots = node_status.get("task_slots", {})
            if task_slots:
                used = task_slots.get("used", 0)
                total = task_slots.get("total", 0)
                load = min(10, int((used / total) * 10)) if total > 0 else 0
                load_bar = "\U0001f7e2" if load <= 3 else ("\U0001f7e1" if load <= 7 else "\U0001f534")
                status_text += (
                    f"\n\n*Task Queue:*\n"
                    f"  Workers: {used}/{total}\n"
                    f"  Load: {load}/10 {load_bar}"
                )
        except Exception:
            pass

        # ── Economy summary ───────────────────────────────────────────
        try:
            econ = await self.client.get_economy()
            if econ:
                summary = econ.get("summary", {})
                net = summary.get("net_position", 0)
                green = summary.get("peers_green", 0)
                amber = summary.get("peers_amber", 0)
                red = summary.get("peers_red", 0)
                wallet = econ.get("wallet", "")
                token_bal = econ.get("token_balance", 0)
                status_text += (
                    f"\n\n*Economy:*\n"
                    f"  Net position: {net:+.1f} credits\n"
                    f"  Peers: \U0001f7e2{green} \U0001f7e1{amber} \U0001f534{red}\n"
                )
                if wallet:
                    status_text += f"  Wallet: `{wallet[:12]}...`\n"
                if token_bal:
                    status_text += f"  $KNARR balance: {token_bal}\n"
        except Exception:
            pass

        await self.send(msg.chat_id, status_text, "Markdown")

    async def _cmd_doctor(self, msg: InboundMessage):
        """Run diagnostics on all services, API keys, and data stores."""
        lines = ["*Knarr Doctor Report*\n"]

        # -- Service probes via skill catalog --
        lines.append("*Services:*")
        probe_skills = ["web-search", "generate-report", "browse-web"]
        try:
            t0 = time.time()
            all_skills = await self.client.get_skills()
            elapsed_ms = int((time.time() - t0) * 1000)
            network_skills = {s["name"]: s for s in all_skills.get("network", [])}
            for skill_name in probe_skills:
                if skill_name in network_skills:
                    s = network_skills[skill_name]
                    providers = s.get("providers", [])
                    if providers:
                        p = providers[0]
                        lines.append(
                            f"  `{skill_name}` — OK ({p.get('host', '?')}:{p.get('port', '?')}, {elapsed_ms}ms)"
                        )
                    else:
                        lines.append(f"  `{skill_name}` — FOUND (no active providers)")
                else:
                    lines.append(f"  `{skill_name}` — NOT FOUND")
        except Exception as e:
            for skill_name in probe_skills:
                lines.append(f"  `{skill_name}` — ERROR: {e}")

        # -- API keys --
        lines.append("\n*API Keys:*")
        key_checks = {
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
            "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        }
        for name, val in key_checks.items():
            status = "set" if val and not val.startswith("your-") else "MISSING"
            lines.append(f"  `{name}` — {status}")

        # -- Skill catalog --
        if self.llm_router:
            catalog_size = len(getattr(self.llm_router, '_skill_catalog', {}))
            last_refresh = getattr(self.llm_router, '_catalog_updated', 0)
            age = int(time.time() - last_refresh) if last_refresh else "never"
            lines.append(f"\n*Catalog:* {catalog_size} skills, refreshed {age}s ago")

        # -- Data stores --
        if self.llm_router and hasattr(self.llm_router, 'session_store') and self.llm_router.session_store:
            ss = self.llm_router.session_store
            if hasattr(ss, 'stats'):
                st = ss.stats()
                lines.append(
                    f"*Sessions:* {st['total_turns']} turns across {st['chat_count']} chats"
                )

        if self.memory_store:
            try:
                facts = self.memory_store.get_facts(msg.chat_id)
                notes = self.memory_store.get_recent_notes(msg.chat_id, days=30)
                lines.append(f"*Memory:* {len(facts)} facts, {len(notes)} notes")
            except Exception:
                lines.append("*Memory:* error reading")

        if self.cron_store:
            try:
                jobs = self.cron_store.list_jobs(msg.chat_id)
                lines.append(f"*Cron:* {len(jobs)} scheduled tasks")
            except Exception:
                lines.append("*Cron:* error reading")

        # -- Node info from Cockpit API --
        try:
            status = await self.client.get_status()
            version = status.get("version", "unknown")
            peer_count = status.get("peer_count", "?")
            lines.append(f"\n*Knarr:* v{version}")
            lines.append(f"*Peers:* {peer_count}")
        except Exception:
            lines.append("\n*Knarr:* version unknown")

        await self.send(msg.chat_id, "\n".join(lines), "Markdown")

    async def _cmd_pair(self, msg: InboundMessage):
        """Generate a pairing code so an unknown user can gain access."""
        global _pairing_codes

        # Only admins (ALLOWED_USERS) can generate pairing codes
        if not is_admin(msg.user_id):
            await self.send(msg.chat_id, "Only admins can generate pairing codes.")
            return

        # In open mode, pairing is unnecessary
        if not _allowed_users and not _allowed_groups:
            await self.send(msg.chat_id, "Access control is open — pairing not needed.")
            return

        # Clean up expired codes
        now = time.time()
        _pairing_codes = {k: v for k, v in _pairing_codes.items() if v["expires"] > now}

        # Generate a 6-digit code
        code = str(random.randint(100000, 999999))
        _pairing_codes[code] = {
            "admin_id": msg.user_id,
            "expires": now + 300,  # 5-minute TTL
        }

        await self.send(
            msg.chat_id,
            f"Pairing code: `{code}`\n\n"
            "Send this code to the person you want to grant access. "
            "They should DM it to the bot within 5 minutes.",
            "Markdown",
        )

    async def _cmd_unpair(self, msg: InboundMessage, args: str):
        """Revoke access for a paired user (admin only)."""
        global _paired_users

        if not is_admin(msg.user_id):
            await self.send(msg.chat_id, "Only admins can unpair users.")
            return

        target = args.strip()
        if not target:
            # Show current paired users
            if _paired_users:
                user_list = ", ".join(str(u) for u in sorted(_paired_users))
                await self.send(msg.chat_id, f"Paired users: {user_list}\n\nUsage: /unpair <user_id>")
            else:
                await self.send(msg.chat_id, "No paired users.")
            return

        try:
            target_id = int(target)
        except ValueError:
            await self.send(msg.chat_id, "Invalid user ID. Usage: /unpair <user_id>")
            return

        if target_id not in _paired_users:
            await self.send(msg.chat_id, f"User {target_id} is not paired.")
            return

        _paired_users.discard(target_id)
        save_paired_users(_paired_users)
        log.info("Admin %d unpaired user %d", msg.user_id, target_id)
        await self.send(msg.chat_id, f"User {target_id} has been unpaired and can no longer access the bot.")

    # ── /configure wizard ─────────────────────────────────────────────
    #
    # Two paths:
    #   1. Custom  — user pastes their own PERSONALITY.md text directly.
    #   2. Generate — user describes what they want in plain language;
    #                 the bot's own LLM generates PERSONALITY.md +
    #                 INSTRUCTIONS.md from that description.
    #
    # No templates live here. Content comes from the user or the LLM.
    # If a School skill exists on the KNARR network, users discover it
    # via /skills and call it like any other skill — no special handling.

    async def _cmd_configure(self, msg: InboundMessage):
        """Start the /configure wizard — admin only."""
        if not is_admin(msg.user_id):
            await self.send(msg.chat_id, "Only the bot owner can use /configure.")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        p_path = os.path.join(base_dir, "PERSONALITY.md")
        current = ""
        if os.path.exists(p_path):
            with open(p_path, encoding="utf-8") as f:
                current = f.read(200).strip()
        if not current:
            current = "(default — no custom personality set)"

        has_llm = self.llm_router is not None
        options = "  1. *Custom* — paste your own personality text\n"
        if has_llm:
            options += "  2. *Generate* — describe what you want, I'll write it\n"
        options += "  0. *Cancel*"

        self._configure_sessions[msg.chat_id] = {"step": "main_menu", "has_llm": has_llm}

        await self.send(
            msg.chat_id,
            f"*Configure your agent*\n\n"
            f"*Current personality (preview):*\n_{current[:200]}_\n\n"
            f"*How do you want to set the new personality?*\n{options}\n\n"
            f"Reply with a number.",
            "Markdown",
        )

    async def _configure_step(self, msg: InboundMessage):
        """Handle replies inside an active /configure wizard session."""
        state = self._configure_sessions.get(msg.chat_id)
        if not state:
            return

        text = (msg.text or "").strip()

        if text in ("/cancel", "0"):
            del self._configure_sessions[msg.chat_id]
            await self.send(msg.chat_id, "Configuration cancelled. Nothing changed.")
            return

        step = state.get("step")

        if step == "main_menu":
            try:
                choice = int(text)
            except ValueError:
                await self.send(msg.chat_id, "Please reply with a number from the menu.")
                return

            if choice == 1:
                state["step"] = "custom_personality"
                await self.send(
                    msg.chat_id,
                    "*Custom personality*\n\n"
                    "Send your PERSONALITY.md text. This defines who your agent is — "
                    "its character, tone, knowledge domain, and capabilities.\n\n"
                    "_(Send *0* at any time to cancel.)_",
                    "Markdown",
                )
            elif choice == 2 and state.get("has_llm"):
                state["step"] = "generate_describe"
                await self.send(
                    msg.chat_id,
                    "*Generate personality*\n\n"
                    "Describe the role you want your agent to have. A sentence or two is enough.\n\n"
                    "_Example: \"A sharp Swiss legal assistant focused on startup contracts and GDPR compliance\"_\n\n"
                    "_(Send *0* to cancel.)_",
                    "Markdown",
                )
            else:
                await self.send(msg.chat_id, "Invalid choice. Reply with a number from the menu.")

        elif step == "custom_personality":
            if len(text) < 20:
                await self.send(
                    msg.chat_id,
                    "That seems too short (min 20 characters). Try again or send *0* to cancel.",
                    "Markdown",
                )
                return
            state["custom_personality"] = text
            state["step"] = "custom_instructions"
            await self.send(
                msg.chat_id,
                "*Instructions (optional)*\n\n"
                "Now send your INSTRUCTIONS.md text — behavioural rules, output format, "
                "things to avoid, how to open conversations.\n\n"
                "_(Send a dash *-* to skip and leave instructions empty.)_",
                "Markdown",
            )

        elif step == "custom_instructions":
            instructions = "" if text in ("-", "—", "skip") else text
            personality = state.get("custom_personality", "")
            await self._write_personality_files(
                msg.chat_id, personality, instructions, "Custom"
            )
            del self._configure_sessions[msg.chat_id]

        elif step == "generate_describe":
            if len(text) < 10:
                await self.send(
                    msg.chat_id,
                    "Description too short. Give me a bit more to work with, or send *0* to cancel.",
                    "Markdown",
                )
                return
            await self.send(msg.chat_id, "Generating your personality files... one moment.")
            try:
                personality, instructions = await self._generate_personality_from_description(text)
            except Exception as e:
                log.error("Personality generation failed: %s", e)
                await self.send(
                    msg.chat_id,
                    f"Generation failed: `{e}`\n\nTry again or use option 1 (Custom) to paste text manually.",
                    "Markdown",
                )
                del self._configure_sessions[msg.chat_id]
                return

            state["step"] = "confirm_generated"
            state["generated_personality"] = personality
            state["generated_instructions"] = instructions

            preview = personality[:400] + ("..." if len(personality) > 400 else "")
            await self.send(
                msg.chat_id,
                f"*Generated personality preview:*\n\n_{preview}_\n\n"
                f"Reply *yes* to apply, *0* to cancel, or *edit* to paste your own text instead.",
                "Markdown",
            )

        elif step == "confirm_generated":
            if text.lower() in ("yes", "y", "ja", "oui", "si"):
                await self._write_personality_files(
                    msg.chat_id,
                    state["generated_personality"],
                    state["generated_instructions"],
                    "Generated",
                )
                del self._configure_sessions[msg.chat_id]
            elif text.lower() == "edit":
                state["step"] = "custom_personality"
                await self.send(
                    msg.chat_id,
                    "OK — send your own personality text to use instead.",
                )
            else:
                await self.send(
                    msg.chat_id,
                    "Reply *yes* to apply, *edit* to write your own, or *0* to cancel.",
                    "Markdown",
                )

    async def _generate_personality_from_description(self, description: str) -> tuple[str, str]:
        """Use the bot's LLM to generate PERSONALITY.md + INSTRUCTIONS.md from a plain-language description."""
        prompt = (
            "You are helping configure a KNARR network agent. "
            "Based on the role description below, generate two markdown files.\n\n"
            f"Role description: {description}\n\n"
            "Respond with EXACTLY this format — no extra text before or after:\n\n"
            "PERSONALITY:\n"
            "<personality text — 100-250 words describing who the agent is, its character, "
            "domain expertise, and tone>\n\n"
            "INSTRUCTIONS:\n"
            "<instructions text — 50-150 words: behavioural rules, output format preferences, "
            "how to open conversations, what to avoid>"
        )

        llm = self.llm_router
        # Prefer the primary Gemini client for a clean one-shot call
        if llm.client:
            import asyncio
            from google.genai import types as genai_types
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: llm.client.models.generate_content(
                    model=llm.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=800,
                    ),
                ),
            )
            raw = response.text or ""
        else:
            # Fallback to LiteLLM
            import litellm
            kwargs = {
                "model": llm.fallback_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if llm.fallback_api_key:
                kwargs["api_key"] = llm.fallback_api_key
            if llm.fallback_api_base:
                kwargs["api_base"] = llm.fallback_api_base
            response = await litellm.acompletion(**kwargs)
            raw = response.choices[0].message.content or ""

        return _parse_generated_personality(raw)

    async def _write_personality_files(
        self,
        chat_id: int,
        personality: str,
        instructions: str,
        label: str,
    ):
        """Write PERSONALITY.md and INSTRUCTIONS.md, then notify the chat."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        p_path = os.path.join(base_dir, "PERSONALITY.md")
        i_path = os.path.join(base_dir, "INSTRUCTIONS.md")

        try:
            with open(p_path, "w", encoding="utf-8") as f:
                f.write(personality)
            with open(i_path, "w", encoding="utf-8") as f:
                f.write(instructions)
            log.info("Personality updated to '%s' via /configure in chat %d", label, chat_id)
            await self.send(
                chat_id,
                f"*Role updated: {label}*\n\n"
                "PERSONALITY.md and INSTRUCTIONS.md have been written. "
                "The new personality will take effect on the next message — no restart needed.",
                "Markdown",
            )
        except OSError as e:
            log.error("Failed to write personality files: %s", e)
            await self.send(
                chat_id,
                f"Failed to write personality files: `{e}`\n\n"
                "Check that the bot has write access to its core directory.",
                "Markdown",
            )

    async def _try_redeem_pairing_code(self, msg: InboundMessage) -> bool:
        """Check if an incoming DM text is a valid pairing code. Returns True if redeemed."""
        global _paired_users, _pairing_codes

        # Only works in DMs, only when access control is active
        if msg.is_group or (not _allowed_users and not _allowed_groups):
            return False

        text = (msg.text or "").strip()
        # Must be exactly 6 digits
        if not text.isdigit() or len(text) != 6:
            return False

        # Clean up expired codes
        now = time.time()
        _pairing_codes = {k: v for k, v in _pairing_codes.items() if v["expires"] > now}

        if text not in _pairing_codes:
            return False

        # Valid code — pair this user
        _pairing_codes.pop(text)
        _paired_users.add(msg.user_id)
        save_paired_users(_paired_users)
        log.info("User %d (%s) paired via code", msg.user_id, msg.from_user)
        await self.send(
            msg.chat_id,
            "You've been paired successfully! You now have access to the bot. Type /help to get started.",
        )
        return True

    async def _cmd_skills(self, msg: InboundMessage):
        try:
            all_skills = await self.client.get_skills()
            network = all_skills.get("network", [])
            if not network:
                await self.send(msg.chat_id, "No skills found on the network.")
                return

            lines = ["*Skills on the network:*\n"]
            for s in sorted(network, key=lambda x: x.get("name", "")):
                name = s.get("name", "?")
                desc = s.get("description", "")
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                lines.append(f"  `{name}` — {desc}")

            await self.send(msg.chat_id, "\n".join(lines), "Markdown")
        except Exception as e:
            log.exception("Error querying skills")
            await self.send(msg.chat_id, f"Error querying network: {e}")

    async def _cmd_run(self, msg: InboundMessage, args: str):
        chat_id = msg.chat_id
        if not args:
            ctx = self._chat_context.get(chat_id)
            if ctx:
                skill_name = ctx["last_skill"]
                input_data = ctx["last_input"]
                await self.send(chat_id, f"Re-running `{skill_name}`...", "Markdown")
                await self._execute_skill(chat_id, skill_name, input_data)
            else:
                await self.send(
                    chat_id,
                    "Usage: `/run <skill_name> <input>`\n\n"
                    "Tip: run `/run` with no args to re-run your last skill.",
                    "Markdown",
                )
            return

        parts = args.split(None, 1)
        skill_name = parts[0]
        raw_input = parts[1] if len(parts) > 1 else ""

        try:
            input_data = json.loads(raw_input)
            if not isinstance(input_data, dict):
                input_data = {"text": raw_input}
        except (json.JSONDecodeError, ValueError):
            input_data = {"text": raw_input} if raw_input else {}

        await self._execute_skill(chat_id, skill_name, input_data)

    # ── Skill execution ──────────────────────────────────────────────

    async def _execute_skill(self, chat_id: int, skill_name: str, input_data: dict):
        """Discover and execute a Knarr skill, sending results to the chat."""
        try:
            providers = await self.client.query_skill(skill_name)
            if not providers:
                await self.send(chat_id, f"No providers found for skill `{skill_name}`.", "Markdown")
                return None

            provider = providers[0]

            await self.send(
                chat_id,
                f"Running `{skill_name}` on `{provider.get('host', '?')}:{provider.get('port', '?')}`...",
                "Markdown",
            )

            result = await self.client.execute(
                skill_name,
                input_data,
                provider={
                    "node_id": provider.get("node_id", ""),
                    "host": provider.get("host", ""),
                    "port": provider.get("port", 0),
                },
                timeout=300,
            )

            reply = format_skill_result(result)
            await self.send(chat_id, reply, "Markdown")

            self._chat_context[chat_id] = {"last_skill": skill_name, "last_input": input_data}
            return result

        except Exception as e:
            log.exception("Error executing skill %s", skill_name)
            await self.send(chat_id, f"Error: {e}")
            return None

    # ── LLM routing ──────────────────────────────────────────────────

    async def _route_to_llm(self, msg: InboundMessage):
        """Route a non-command message through the LLM."""
        if self.llm_router:
            try:
                # Status callback: edit-in-place if the adapter supports it,
                # otherwise falls back to regular send
                async def status_fn(text: str):
                    if self.send_status:
                        await self.send_status(msg.chat_id, text)
                    else:
                        await self.send(msg.chat_id, text)

                reply = await self.llm_router.route_message(
                    self.client, msg.chat_id, msg.text,
                    media_bytes=msg.media_bytes,
                    media_mime=msg.media_mime,
                    status_fn=status_fn,
                )
                if reply and reply.strip().upper() != "NO_REPLY":
                    try:
                        await self.send(msg.chat_id, reply)
                    except Exception:
                        log.exception("Failed to deliver LLM reply to chat %d", msg.chat_id)
            except Exception as e:
                log.exception("LLM routing error")
                try:
                    await self.send(msg.chat_id, f"Error: {e}")
                except Exception:
                    log.warning("Failed to deliver error message to chat %d", msg.chat_id)
        else:
            if not msg.is_group:
                await self.send(
                    msg.chat_id,
                    "I only understand /commands right now.\nType /help to see what's available.",
                )

    # ── Background tasks ────────────────────────────────────────────

    async def _cmd_tasks(self, msg: InboundMessage):
        """List all background tasks for this chat."""
        chat_tasks = [
            t for t in self._running_tasks.values()
            if t["chat_id"] == msg.chat_id
        ]
        if not chat_tasks:
            await self.send(msg.chat_id, "No background tasks for this chat.")
            return

        lines = [f"Background tasks ({len(chat_tasks)}):"]
        for t in chat_tasks:
            elapsed = format_uptime(time.time() - t["started_at"])
            lines.append(
                f"  [{t['id']}] {t['name']} — {t['status']} ({elapsed})"
            )
        await self.send(msg.chat_id, "\n".join(lines))

    async def _spawn_task(self, chat_id: int, name: str, instructions: str) -> int:
        """Spawn an isolated background task with its own LLM conversation.

        The subagent gets access to Knarr skills but NOT to memory/cron tools
        (to keep it isolated). When done, it delivers the result to the chat.

        Returns the task ID.
        """
        task_id = self._next_task_id
        self._next_task_id += 1

        task_info = {
            "id": task_id,
            "name": name,
            "chat_id": chat_id,
            "status": "running",
            "started_at": time.time(),
            "asyncio_task": None,
        }
        self._running_tasks[task_id] = task_info

        async def _run_subagent():
            try:
                log.info("Subagent task %d '%s' starting", task_id, name)

                async def task_status_fn(text: str):
                    if self.send_status:
                        await self.send_status(chat_id, f"[Task #{task_id}: {name}] {text}")
                    else:
                        await self.send(chat_id, f"[Task #{task_id}: {name}] {text}")

                # Use a simplified prompt for the subagent — no memory/cron tools
                subagent_prompt = (
                    "You are a focused task executor. Complete the following task "
                    "using available tools. Be thorough and report your findings clearly.\n"
                    "Use the send_status_update tool to keep the user informed of your "
                    "progress on major milestones (e.g. 'Found 3 relevant sources, now "
                    "analyzing...'), but don't spam updates for every small step.\n\n"
                    f"Task: {instructions}"
                )

                # Route through the LLM with the task instructions + status updates
                if self.llm_router:
                    result = await self.llm_router.route_message(
                        self.client, chat_id, subagent_prompt,
                        status_fn=task_status_fn,
                    )
                else:
                    result = "No LLM available to execute background task."

                task_info["status"] = "completed"
                notice = f"[Task #{task_id}: {name}]\n\n{result}"
                await self.send(chat_id, notice)
                log.info("Subagent task %d completed", task_id)

            except Exception as e:
                task_info["status"] = f"failed: {e}"
                log.exception("Subagent task %d failed", task_id)
                try:
                    await self.send(chat_id, f"[Task #{task_id}: {name}] Failed: {e}")
                except Exception:
                    log.warning("Failed to deliver subagent error for task %d", task_id)

        # Launch as an asyncio background task
        asyncio_task = asyncio.create_task(_run_subagent())
        task_info["asyncio_task"] = asyncio_task

        log.info("Spawned background task %d '%s' for chat %d", task_id, name, chat_id)
        return task_id

    # ── Cron job execution ───────────────────────────────────────────

    async def execute_cron_job(self, job: dict):
        """Execute a scheduled cron job by routing its message through the LLM."""
        chat_id = job["chat_id"]
        job_name = job["name"]
        message = job["message"]
        job_id = job["id"]

        # Advance next_run_at IMMEDIATELY so the tick loop doesn't re-fire
        # while this (potentially long-running) job is still executing.
        self.cron_store.mark_job_run(job_id)

        log.info("Executing cron job %d '%s': %s", job_id, job_name, message[:80])

        try:
            if self.llm_router:
                # Wrap the cron message with formatting guidance so the LLM
                # produces a polished, well-structured response.
                prompt = (
                    f"[Scheduled task: {job_name}]\n\n"
                    f"{message}\n\n"
                    "This is a background task. If it involves multiple slow steps (30s+), "
                    "send ONE status update with a brief summary of progress so far. "
                    "Do not narrate every step — just deliver the final result.\n\n"
                    "Format your final response clearly with a headline, short summary, "
                    "and key bullet points. Use markdown formatting (bold, bullets). "
                    "Keep it concise — this is an automated digest delivered to the chat."
                )
                reply = await self.llm_router.route_message(self.client, chat_id, prompt)
                if reply and reply.strip().upper() != "NO_REPLY":
                    notice = f"[Scheduled: {job_name}]\n\n{reply}"
                    try:
                        await self.send(chat_id, notice)
                    except Exception:
                        log.exception("Failed to deliver cron job %d result", job_id)
                log.info("Cron job %d completed", job_id)
            else:
                log.warning("Cron job %d skipped — no LLM router", job_id)
        except Exception as e:
            log.exception("Error executing cron job %d", job_id)
            self.cron_store.mark_job_error(job_id, str(e))

    # ── Heartbeat execution ──────────────────────────────────────────

    async def execute_heartbeat(self, chat_id: int, instructions: str) -> bool:
        """Execute a heartbeat check. Returns True if nothing to report."""
        log.info("Heartbeat firing for chat %d", chat_id)
        try:
            if self.llm_router:
                prompt = (
                    f"[HEARTBEAT] The following are your standing instructions. "
                    f"Execute them now. If there is nothing actionable, respond with "
                    f"just the word HEARTBEAT_OK and nothing else.\n\n{instructions}"
                )
                reply = await self.llm_router.route_message(self.client, chat_id, prompt)
                if reply and reply.strip().upper() not in ("HEARTBEAT_OK", "NO_REPLY"):
                    notice = f"[Heartbeat]\n\n{reply}"
                    try:
                        await self.send(chat_id, notice)
                    except Exception:
                        log.exception("Failed to deliver heartbeat to chat %d", chat_id)
                    log.info("Heartbeat produced output")
                    return False
                else:
                    log.info("Heartbeat: nothing to report")
                    return True
        except Exception as e:
            log.exception("Error executing heartbeat")
        return False
