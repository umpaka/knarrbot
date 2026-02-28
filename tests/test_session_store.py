"""Tests for session_store.py — session persistence, pruning, stats, replace_all."""

import time


class TestSaveAndLoad:
    def test_save_and_load(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "hello"}])
        ss.save_turn(1, "model", [{"type": "text", "text": "hi there"}])

        turns = ss.load_turns(1)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "model"

    def test_load_respects_limit(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        for i in range(10):
            ss.save_turn(1, "user", [{"type": "text", "text": f"msg {i}"}])
        turns = ss.load_turns(1, limit=3)
        assert len(turns) == 3
        # Should be the 3 most recent, in chronological order
        assert "msg 7" in turns[0]["parts"][0]["text"]

    def test_chat_isolation(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "chat1"}])
        ss.save_turn(2, "user", [{"type": "text", "text": "chat2"}])
        assert len(ss.load_turns(1)) == 1
        assert len(ss.load_turns(2)) == 1

    def test_clear(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "hello"}])
        ss.clear(1)
        assert ss.load_turns(1) == []


class TestPrune:
    def test_prune_deletes_old_turns(self, tmp_path):
        from session_store import SessionStore
        import sqlite3

        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        # Insert a turn manually with an old timestamp (60 days ago)
        old_ts = time.time() - 60 * 86400
        conn = sqlite3.connect(str(tmp_path / "s.db"))
        conn.execute(
            "INSERT INTO session_turns (chat_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
            (1, "user", '[{"type":"text","text":"old"}]', old_ts),
        )
        conn.commit()
        conn.close()

        # Add a recent turn normally
        ss.save_turn(1, "user", [{"type": "text", "text": "recent"}])

        deleted = ss.prune(max_age_days=30)
        assert deleted == 1

        turns = ss.load_turns(1)
        assert len(turns) == 1
        assert "recent" in turns[0]["parts"][0]["text"]

    def test_prune_nothing_to_delete(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "fresh"}])
        deleted = ss.prune(max_age_days=30)
        assert deleted == 0

    def test_prune_across_chats(self, tmp_path):
        from session_store import SessionStore
        import sqlite3

        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        old_ts = time.time() - 60 * 86400
        conn = sqlite3.connect(str(tmp_path / "s.db"))
        conn.execute(
            "INSERT INTO session_turns (chat_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
            (1, "user", '[{"type":"text","text":"old1"}]', old_ts),
        )
        conn.execute(
            "INSERT INTO session_turns (chat_id, role, parts_json, created_at) VALUES (?, ?, ?, ?)",
            (2, "user", '[{"type":"text","text":"old2"}]', old_ts),
        )
        conn.commit()
        conn.close()

        deleted = ss.prune(max_age_days=30)
        assert deleted == 2


class TestStats:
    def test_empty_stats(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        st = ss.stats()
        assert st == {"total_turns": 0, "chat_count": 0}

    def test_stats_counts(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "a"}])
        ss.save_turn(1, "model", [{"type": "text", "text": "b"}])
        ss.save_turn(2, "user", [{"type": "text", "text": "c"}])
        st = ss.stats()
        assert st["total_turns"] == 3
        assert st["chat_count"] == 2


class TestReplaceAll:
    def test_replace_all(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))

        # Save some initial turns
        ss.save_turn(1, "user", [{"type": "text", "text": "old1"}])
        ss.save_turn(1, "model", [{"type": "text", "text": "old2"}])
        ss.save_turn(1, "user", [{"type": "text", "text": "old3"}])

        # Replace with compacted data
        new_turns = [
            {"role": "user", "parts": [{"type": "text", "text": "[summary]"}]},
            {"role": "model", "parts": [{"type": "text", "text": "ack"}]},
        ]
        ss.replace_all(1, new_turns)

        turns = ss.load_turns(1)
        assert len(turns) == 2
        assert "[summary]" in turns[0]["parts"][0]["text"]

    def test_replace_all_doesnt_affect_other_chats(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))

        ss.save_turn(1, "user", [{"type": "text", "text": "chat1"}])
        ss.save_turn(2, "user", [{"type": "text", "text": "chat2"}])

        ss.replace_all(1, [
            {"role": "user", "parts": [{"type": "text", "text": "replaced"}]},
        ])

        # Chat 2 should be untouched
        turns2 = ss.load_turns(2)
        assert len(turns2) == 1
        assert "chat2" in turns2[0]["parts"][0]["text"]

    def test_replace_all_empty(self, tmp_path):
        from session_store import SessionStore
        ss = SessionStore(db_path=str(tmp_path / "s.db"))
        ss.save_turn(1, "user", [{"type": "text", "text": "something"}])
        ss.replace_all(1, [])
        assert ss.load_turns(1) == []
