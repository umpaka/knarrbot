"""Tests for cron_store.py — scheduled task storage."""

import time


class TestAddAndList:
    def test_add_job(self, cron_store):
        jid = cron_store.add_job(
            chat_id=1, name="test", message="do stuff",
            schedule_type="once", schedule_value="30",
        )
        assert jid > 0

    def test_list_jobs(self, cron_store):
        cron_store.add_job(1, "job1", "msg1", "once", "10")
        cron_store.add_job(1, "job2", "msg2", "interval", "3600")
        jobs = cron_store.list_jobs(1)
        assert len(jobs) == 2
        names = {j["name"] for j in jobs}
        assert names == {"job1", "job2"}

    def test_list_empty(self, cron_store):
        assert cron_store.list_jobs(999) == []

    def test_job_fields(self, cron_store):
        cron_store.add_job(1, "daily_check", "check news", "daily", "09:00")
        jobs = cron_store.list_jobs(1)
        j = jobs[0]
        assert j["name"] == "daily_check"
        assert j["message"] == "check news"
        assert j["schedule_type"] == "daily"
        assert j["schedule_value"] == "09:00"
        assert j["enabled"] is True
        assert j["next_run_at"] > 0


class TestRemoveJob:
    def test_remove_existing(self, cron_store):
        jid = cron_store.add_job(1, "temp", "msg", "once", "5")
        assert cron_store.remove_job(jid, chat_id=1) is True
        assert cron_store.list_jobs(1) == []

    def test_remove_nonexistent(self, cron_store):
        assert cron_store.remove_job(999, chat_id=1) is False

    def test_remove_wrong_chat(self, cron_store):
        jid = cron_store.add_job(1, "mine", "msg", "once", "5")
        # Chat 2 cannot delete chat 1's job
        assert cron_store.remove_job(jid, chat_id=2) is False
        assert len(cron_store.list_jobs(1)) == 1


class TestChatIsolation:
    def test_jobs_isolated(self, cron_store):
        cron_store.add_job(1, "a", "m", "once", "10")
        cron_store.add_job(2, "b", "m", "once", "10")
        assert len(cron_store.list_jobs(1)) == 1
        assert len(cron_store.list_jobs(2)) == 1
        assert cron_store.list_jobs(1)[0]["name"] == "a"


class TestGetDueJobs:
    def test_due_job_returned(self, cron_store):
        # Create a job that fires 0 minutes from now (immediately due)
        cron_store.add_job(1, "now", "do it", "once", "0")
        # Sleep a tiny bit so next_run_at is in the past
        time.sleep(0.05)
        due = cron_store.get_due_jobs()
        assert len(due) == 1
        assert due[0]["name"] == "now"

    def test_future_job_not_returned(self, cron_store):
        cron_store.add_job(1, "later", "do it", "once", "9999")
        due = cron_store.get_due_jobs()
        assert len(due) == 0


class TestMarkJobRun:
    def test_once_job_disabled_after_run(self, cron_store):
        jid = cron_store.add_job(1, "oneshot", "msg", "once", "0")
        time.sleep(0.05)
        cron_store.mark_job_run(jid)
        jobs = cron_store.list_jobs(1)
        assert len(jobs) == 1
        assert jobs[0]["enabled"] is False
        assert jobs[0]["last_status"] == "ok"

    def test_interval_job_rescheduled_after_run(self, cron_store):
        jid = cron_store.add_job(1, "recurring", "msg", "interval", "3600")
        jobs_before = cron_store.list_jobs(1)
        old_next_run = jobs_before[0]["next_run_at"]

        cron_store.mark_job_run(jid)

        jobs_after = cron_store.list_jobs(1)
        assert jobs_after[0]["enabled"] is True
        assert jobs_after[0]["next_run_at"] > old_next_run

    def test_mark_error(self, cron_store):
        jid = cron_store.add_job(1, "failing", "msg", "interval", "3600")
        cron_store.mark_job_error(jid, "Connection timeout")
        jobs = cron_store.list_jobs(1)
        assert jobs[0]["last_status"] == "error"


class TestComputeNextRun:
    def test_once(self, cron_store):
        from cron_store import CronStore
        now = time.time()
        nxt = CronStore._compute_next_run("once", "30", now)
        assert abs(nxt - (now + 1800)) < 1  # 30 minutes

    def test_interval(self, cron_store):
        from cron_store import CronStore
        now = time.time()
        nxt = CronStore._compute_next_run("interval", "7200", now)
        assert abs(nxt - (now + 7200)) < 1

    def test_interval_minimum_60s(self, cron_store):
        from cron_store import CronStore
        now = time.time()
        nxt = CronStore._compute_next_run("interval", "10", now)
        # Should enforce 60s minimum
        assert abs(nxt - (now + 60)) < 1

    def test_daily(self, cron_store):
        from cron_store import CronStore
        import datetime
        now = time.time()
        nxt = CronStore._compute_next_run("daily", "09:00", now)
        # Should be a valid timestamp in the future
        assert nxt > now or abs(nxt - now) < 86400

    def test_invalid_type_fallback(self, cron_store):
        from cron_store import CronStore
        now = time.time()
        nxt = CronStore._compute_next_run("unknown", "x", now)
        assert abs(nxt - (now + 3600)) < 1  # defaults to 1 hour


class TestFormatJobsText:
    def test_empty(self, cron_store):
        text = cron_store.format_jobs_text(1)
        assert "No scheduled tasks" in text

    def test_with_jobs(self, cron_store):
        cron_store.add_job(1, "morning_news", "check news", "daily", "09:00")
        text = cron_store.format_jobs_text(1)
        assert "1)" in text or "morning_news" in text
        assert "daily=09:00" in text
