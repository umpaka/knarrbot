"""Send a message to a Telegram chat via the Bot API."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def _get_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token or token == "your-token-here":
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Create a bot via @BotFather and add the token to .env"
        )
    return token


async def handle(input_data: dict) -> dict:
    """Send a message to a Telegram chat.

    Args:
        input_data: dict with keys:
            - chat_id (required): Telegram chat ID
            - text (required): Message text to send
            - parse_mode (optional): "Markdown" or "HTML" for formatted text

    Returns:
        dict with keys:
            - message_id: ID of the sent message
            - status: "sent" on success
    """
    chat_id = input_data.get("chat_id", "").strip()
    text = input_data.get("text", "").strip()

    if not chat_id:
        return {"error": "chat_id is required"}
    if not text:
        return {"error": "text is required"}

    token = _get_token()
    url = f"{TELEGRAM_API.format(token=token)}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    parse_mode = input_data.get("parse_mode", "").strip()
    if parse_mode in ("Markdown", "HTML"):
        payload["parse_mode"] = parse_mode

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload)

    data = resp.json()

    if not data.get("ok"):
        description = data.get("description", "Unknown Telegram API error")
        return {"error": description}

    message_id = str(data["result"]["message_id"])
    return {"message_id": message_id, "status": "sent"}
