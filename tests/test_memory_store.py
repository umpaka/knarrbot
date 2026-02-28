"""Tests for memory_store.py — persistent fact/note storage."""


class TestSaveFact:
    def test_save_and_retrieve(self, memory_store):
        fid = memory_store.save_fact(chat_id=1, key="name", value="Samim")
        assert fid > 0
        facts = memory_store.get_facts(chat_id=1)
        assert len(facts) == 1
        assert facts[0]["key"] == "name"
        assert facts[0]["value"] == "Samim"

    def test_key_normalized_lowercase(self, memory_store):
        memory_store.save_fact(1, "  User_Language  ", "German")
        facts = memory_store.get_facts(1)
        assert facts[0]["key"] == "user_language"

    def test_upsert_updates_value(self, memory_store):
        id1 = memory_store.save_fact(1, "lang", "English")
        id2 = memory_store.save_fact(1, "lang", "German")
        # Same key → same row, updated value
        assert id1 == id2
        facts = memory_store.get_facts(1)
        assert len(facts) == 1
        assert facts[0]["value"] == "German"

    def test_saved_by_field(self, memory_store):
        memory_store.save_fact(1, "pref", "dark mode", saved_by="llm")
        facts = memory_store.get_facts(1)
        assert facts[0]["saved_by"] == "llm"

    def test_multiple_facts(self, memory_store):
        memory_store.save_fact(1, "name", "Samim")
        memory_store.save_fact(1, "lang", "German")
        memory_store.save_fact(1, "role", "developer")
        facts = memory_store.get_facts(1)
        assert len(facts) == 3

    def test_limit(self, memory_store):
        for i in range(10):
            memory_store.save_fact(1, f"key{i}", f"val{i}")
        facts = memory_store.get_facts(1, limit=3)
        assert len(facts) == 3


class TestDeleteFact:
    def test_delete_existing(self, memory_store):
        fid = memory_store.save_fact(1, "temp", "data")
        assert memory_store.delete_fact(fid, chat_id=1) is True
        assert memory_store.get_facts(1) == []

    def test_delete_nonexistent(self, memory_store):
        assert memory_store.delete_fact(999, chat_id=1) is False

    def test_delete_wrong_chat(self, memory_store):
        fid = memory_store.save_fact(1, "secret", "val")
        # Chat 2 cannot delete chat 1's fact
        assert memory_store.delete_fact(fid, chat_id=2) is False
        # Still exists for chat 1
        assert len(memory_store.get_facts(1)) == 1


class TestChatIsolation:
    def test_facts_isolated_by_chat(self, memory_store):
        memory_store.save_fact(1, "name", "Alice")
        memory_store.save_fact(2, "name", "Bob")
        facts_1 = memory_store.get_facts(1)
        facts_2 = memory_store.get_facts(2)
        assert len(facts_1) == 1
        assert facts_1[0]["value"] == "Alice"
        assert len(facts_2) == 1
        assert facts_2[0]["value"] == "Bob"

    def test_notes_isolated_by_chat(self, memory_store):
        memory_store.save_note(1, "Chat 1 note")
        memory_store.save_note(2, "Chat 2 note")
        notes_1 = memory_store.get_recent_notes(1)
        notes_2 = memory_store.get_recent_notes(2)
        assert len(notes_1) == 1
        assert len(notes_2) == 1
        assert notes_1[0]["text"] == "Chat 1 note"


class TestNotes:
    def test_save_and_retrieve(self, memory_store):
        nid = memory_store.save_note(1, "Deployed to VPS")
        assert nid > 0
        notes = memory_store.get_recent_notes(1)
        assert len(notes) == 1
        assert notes[0]["text"] == "Deployed to VPS"

    def test_multiple_notes_same_day(self, memory_store):
        memory_store.save_note(1, "First event")
        memory_store.save_note(1, "Second event")
        notes = memory_store.get_recent_notes(1)
        assert len(notes) == 2

    def test_date_filter(self, memory_store):
        # Notes from today should appear with days=1
        memory_store.save_note(1, "Today's note")
        notes = memory_store.get_recent_notes(1, days=1)
        assert len(notes) == 1
        # days=0 means only today (cutoff = today's date)
        notes_0 = memory_store.get_recent_notes(1, days=0)
        assert len(notes_0) == 1


class TestFormatMemoryContext:
    def test_empty_returns_empty_string(self, memory_store):
        assert memory_store.format_memory_context(1) == ""

    def test_with_facts(self, memory_store):
        memory_store.save_fact(1, "name", "Samim")
        ctx = memory_store.format_memory_context(1)
        assert "MEMORY" in ctx.upper()  # heading may vary
        assert "name: Samim" in ctx

    def test_with_notes(self, memory_store):
        memory_store.save_note(1, "Deployed today")
        ctx = memory_store.format_memory_context(1)
        assert "NOTE" in ctx.upper()  # heading may vary
        assert "Deployed today" in ctx

    def test_with_both(self, memory_store):
        memory_store.save_fact(1, "name", "Alice")
        memory_store.save_note(1, "Meeting with Bob")
        ctx = memory_store.format_memory_context(1)
        assert "MEMORY" in ctx.upper()
        assert "NOTE" in ctx.upper()


class TestClearAll:
    def test_clear_all(self, memory_store):
        memory_store.save_fact(1, "name", "Samim")
        memory_store.save_fact(1, "lang", "German")
        memory_store.save_note(1, "Deployed stuff")
        cleared = memory_store.clear_all(1)
        assert cleared == 3
        assert memory_store.get_facts(1) == []
        assert memory_store.get_recent_notes(1) == []

    def test_clear_all_empty(self, memory_store):
        cleared = memory_store.clear_all(999)
        assert cleared == 0

    def test_clear_all_only_affects_target_chat(self, memory_store):
        memory_store.save_fact(1, "a", "val")
        memory_store.save_fact(2, "b", "val")
        memory_store.clear_all(1)
        assert memory_store.get_facts(1) == []
        assert len(memory_store.get_facts(2)) == 1


class TestFormatFactsText:
    def test_empty(self, memory_store):
        text = memory_store.format_facts_text(1)
        assert "No stored memories" in text

    def test_with_facts(self, memory_store):
        memory_store.save_fact(1, "color", "blue")
        text = memory_store.format_facts_text(1)
        assert "1 facts" in text
        assert "color: blue" in text
