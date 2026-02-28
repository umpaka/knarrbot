"""Fetch recent messages from a Telegram chat via the Bot API."""

import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API = "https://api.telegram.org/bot{token}"

# In-memory offset tracking per chat.
# Persists across calls while the node is running, resets on restart.
_offsets: dict[str, int] = {}


def _get_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token or token == "your-token-here":
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Create a bot via @BotFather and add the token to .env"
        )
    return token


async def handle(input_data: dict) -> dict:
    """Fetch recent messages from a Telegram chat.

    Args:
        input_data: dict with keys:
            - chat_id (optional): Filter messages to this chat ID.
              If omitted, returns messages from all chats.
            - limit (optional): Max messages to return (default "20", max "100")
            - mark_read (optional): "true" to advance the offset so these
              messages aren't returned again (default "true")

    Returns:
        dict with keys:
            - messages_json: JSON array of message objects, each with:
                chat_id, message_id, from_user, date, text
            - count: Number of messages returned
    """
    chat_id_filter = input_data.get("chat_id", "").strip()
    limit = min(int(input_data.get("limit", "20") or "20"), 100)
    mark_read = input_data.get("mark_read", "true").strip().lower() != "false"

    token = _get_token()
    url = f"{TELEGRAM_API.format(token=token)}/getUpdates"

    # Use stored offset to avoid re-fetching old messages
    offset_key = chat_id_filter or "__all__"
    params: dict = {"timeout": 0, "limit": limit}
    if offset_key in _offsets:
        params["offset"] = _offsets[offset_key]

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=params)

    data = resp.json()

    if not data.get("ok"):
        description = data.get("description", "Unknown Telegram API error")
        return {"error": description}

    updates = data.get("result", [])
    messages = []
    max_update_id = _offsets.get(offset_key, 0)

    for update in updates:
        update_id = update.get("update_id", 0)
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            # Track offset even for non-message updates
            max_update_id = max(max_update_id, update_id)
            continue

        msg_chat_id = str(msg.get("chat", {}).get("id", ""))

        # Filter by chat_id if specified
        if chat_id_filter and msg_chat_id != chat_id_filter:
            max_update_id = max(max_update_id, update_id)
            continue

        from_user = msg.get("from", {})
        username = from_user.get("username", "")
        first_name = from_user.get("first_name", "")
        display_name = username or first_name or str(from_user.get("id", "unknown"))

        messages.append({
            "chat_id": msg_chat_id,
            "message_id": str(msg.get("message_id", "")),
            "from_user": display_name,
            "date": str(msg.get("date", "")),
            "text": msg.get("text", ""),
        })

        max_update_id = max(max_update_id, update_id)

    # Advance offset so next call gets only new messages
    if mark_read and max_update_id > 0:
        _offsets[offset_key] = max_update_id + 1

    return {
        "messages_json": json.dumps(messages),
        "count": str(len(messages)),
    }
