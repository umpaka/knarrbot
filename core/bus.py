"""Message bus types for channel-agnostic communication.

Defines the message types that flow between channel adapters (Telegram, Discord, etc.)
and the agent core. Adding a new channel means writing a new adapter that produces
InboundMessage objects and handles OutboundMessage via the send callback.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable


@dataclass
class InboundMessage:
    """A message coming from any channel into the agent."""

    channel: str  # "telegram", "discord", "cron", "heartbeat"
    chat_id: int
    text: str
    from_user: str = ""
    display_name: str = ""
    user_id: int = 0  # Numeric user ID for access control
    chat_title: str = "DM"
    message_id: int = 0
    is_group: bool = False
    media_bytes: Optional[bytes] = None
    media_mime: str = ""


# Type alias for the send callback:
#   async def send(chat_id: int, text: str, parse_mode: str = "") -> None
SendFn = Callable[..., Awaitable[None]]
