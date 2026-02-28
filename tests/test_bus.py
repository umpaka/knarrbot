"""Tests for bus.py — message types."""

from bus import InboundMessage


class TestInboundMessage:
    def test_required_fields(self):
        msg = InboundMessage(channel="telegram", chat_id=123, text="hello")
        assert msg.channel == "telegram"
        assert msg.chat_id == 123
        assert msg.text == "hello"

    def test_defaults(self):
        msg = InboundMessage(channel="test", chat_id=1, text="")
        assert msg.from_user == ""
        assert msg.display_name == ""
        assert msg.chat_title == "DM"
        assert msg.message_id == 0
        assert msg.is_group is False
        assert msg.media_bytes is None
        assert msg.media_mime == ""

    def test_all_fields(self):
        msg = InboundMessage(
            channel="discord",
            chat_id=42,
            text="hi",
            from_user="alice",
            display_name="Alice W",
            chat_title="General",
            message_id=999,
            is_group=True,
            media_bytes=b"\x89PNG",
            media_mime="image/png",
        )
        assert msg.channel == "discord"
        assert msg.from_user == "alice"
        assert msg.display_name == "Alice W"
        assert msg.chat_title == "General"
        assert msg.message_id == 999
        assert msg.is_group is True
        assert msg.media_bytes == b"\x89PNG"
        assert msg.media_mime == "image/png"

    def test_equality(self):
        a = InboundMessage(channel="t", chat_id=1, text="x")
        b = InboundMessage(channel="t", chat_id=1, text="x")
        assert a == b

    def test_inequality(self):
        a = InboundMessage(channel="t", chat_id=1, text="x")
        b = InboundMessage(channel="t", chat_id=1, text="y")
        assert a != b
