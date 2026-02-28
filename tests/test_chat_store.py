"""Tests for chat_store.py — chat message history."""

import time


class TestStoreAndRetrieve:
    def test_store_and_get(self, chat_store):
        chat_store.store_message(
            chat_id=1, username="samim", text="Hello!",
            chat_title="Test", display_name="Samim",
        )
        history = chat_store.get_history(1)
        assert "Hello!" in history
        assert "@samim" in history

    def test_multiple_messages_chronological(self, chat_store):
        now = time.time()
        chat_store.store_message(1, "alice", "First", timestamp=now - 10)
        chat_store.store_message(1, "bob", "Second", timestamp=now - 5)
        chat_store.store_message(1, "alice", "Third", timestamp=now)
        history = chat_store.get_history(1)
        # Should be in chronological order (oldest first)
        first_pos = history.index("First")
        second_pos = history.index("Second")
        third_pos = history.index("Third")
        assert first_pos < second_pos < third_pos

    def test_limit(self, chat_store):
        now = time.time()
        for i in range(20):
            chat_store.store_message(1, "user", f"msg{i}", timestamp=now + i)
        history = chat_store.get_history(1, limit=5)
        assert "5 messages" in history

    def test_empty_chat(self, chat_store):
        history = chat_store.get_history(999)
        assert "No messages found" in history


class TestFilters:
    def test_filter_by_username(self, chat_store):
        now = time.time()
        chat_store.store_message(1, "alice", "Alice msg", timestamp=now)
        chat_store.store_message(1, "bob", "Bob msg", timestamp=now + 1)
        history = chat_store.get_history(1, username="alice")
        assert "Alice msg" in history
        assert "Bob msg" not in history

    def test_filter_by_username_strips_at(self, chat_store):
        chat_store.store_message(1, "alice", "Hi")
        history = chat_store.get_history(1, username="@alice")
        assert "Hi" in history

    def test_filter_by_search(self, chat_store):
        now = time.time()
        chat_store.store_message(1, "u", "The weather is nice", timestamp=now)
        chat_store.store_message(1, "u", "Python is great", timestamp=now + 1)
        history = chat_store.get_history(1, search="Python")
        assert "Python" in history
        assert "weather" not in history

    def test_filter_by_since_minutes(self, chat_store):
        now = time.time()
        chat_store.store_message(1, "u", "Old message", timestamp=now - 7200)  # 2 hours ago
        chat_store.store_message(1, "u", "Recent message", timestamp=now)
        history = chat_store.get_history(1, since_minutes=60)
        assert "Recent message" in history
        assert "Old message" not in history


class TestChatIsolation:
    def test_messages_isolated(self, chat_store):
        chat_store.store_message(1, "u", "Chat 1 msg")
        chat_store.store_message(2, "u", "Chat 2 msg")
        h1 = chat_store.get_history(1)
        h2 = chat_store.get_history(2)
        assert "Chat 1 msg" in h1
        assert "Chat 2 msg" not in h1
        assert "Chat 2 msg" in h2
        assert "Chat 1 msg" not in h2


class TestStats:
    def test_stats_with_messages(self, chat_store):
        now = time.time()
        chat_store.store_message(1, "u", "A", timestamp=now - 100)
        chat_store.store_message(1, "u", "B", timestamp=now)
        stats = chat_store.get_stats(1)
        assert stats["total_messages"] == 2
        assert stats["first_message"] is not None
        assert stats["last_message"] is not None

    def test_stats_empty_chat(self, chat_store):
        stats = chat_store.get_stats(999)
        assert stats["total_messages"] == 0
