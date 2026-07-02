"""Unit tests for the 30s deferred-compound-action buffer in bot.py (spec 4/7):
classify_deferral, and the queue/cancel/fire lifecycle of PendingDeferred.

No Telegram/LLM calls — _execute_and_reply is mocked so only the timer/queue
plumbing is under test.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot


class FakeMessage:
    async def reply_text(self, *args, **kwargs):
        return None


class FakeUpdate:
    def __init__(self):
        self.message = FakeMessage()


def _reset_bot_state():
    bot.pending_deferred = None


def test_classify_deferral_true_for_compound_functions():
    assert bot.classify_deferral([{"function": "switch_active_task", "params": {}}])
    assert bot.classify_deferral([{"function": "resume_previous_task", "params": {}}])


def test_classify_deferral_false_for_simple_functions():
    for fn in (
        "complete_task",
        "extend_task_instance",
        "extend_task_total",
        "extend_current_gcal_block",
        "create_event",
        "create_task",
        "reschedule_task",
        "reschedule_missed_work",
        "move_due_date",
        "update_task",
        "log_work",
        "get_schedule_for_window",
        "get_break_allowance",
    ):
        assert not bot.classify_deferral([{"function": fn, "params": {}}]), fn


def test_classify_deferral_true_if_any_call_is_compound():
    calls = [
        {"function": "extend_task_instance", "params": {}},
        {"function": "switch_active_task", "params": {}},
    ]
    assert bot.classify_deferral(calls)


def test_classify_deferral_empty_list_is_false():
    assert not bot.classify_deferral([])


async def test_cancel_before_fire_prevents_execution():
    _reset_bot_state()
    calls = [{"function": "switch_active_task", "params": {"new_task_query": "orgo"}}]
    with patch.object(bot, "_execute_and_reply", new_callable=AsyncMock) as mock_exec:
        bot._queue_deferred(FakeUpdate(), calls, "doing orgo instead", "On it!", None)
        assert bot.pending_deferred is not None

        cancelled = bot._cancel_pending_deferred()
        assert cancelled is not None
        assert bot.pending_deferred is None

        # Give the background task a chance to run; it must not fire after cancel.
        await asyncio.sleep(0.05)
        mock_exec.assert_not_called()


async def test_queue_fires_after_window():
    _reset_bot_state()
    calls = [{"function": "resume_previous_task", "params": {"work_duration_minutes": 30}}]
    with (
        patch.object(bot, "DEFER_WINDOW_SEC", 0.05),
        patch.object(bot, "_execute_and_reply", new_callable=AsyncMock) as mock_exec,
    ):
        update = FakeUpdate()
        bot._queue_deferred(update, calls, "still on my last task", "Keeping you there!", "ctx")
        assert bot.pending_deferred is not None

        await asyncio.sleep(0.2)

        mock_exec.assert_called_once()
        args, kwargs = mock_exec.call_args
        assert args[0] is update
        assert args[1] == calls
        assert args[2] == "still on my last task"
        assert kwargs.get("amendment_context") == "ctx"
        assert bot.pending_deferred is None


async def test_requeue_cancels_previous_pending():
    _reset_bot_state()
    with (
        patch.object(bot, "DEFER_WINDOW_SEC", 5),
        patch.object(bot, "_execute_and_reply", new_callable=AsyncMock) as mock_exec,
    ):
        first_calls = [{"function": "switch_active_task", "params": {"new_task_query": "orgo"}}]
        second_calls = [{"function": "resume_previous_task", "params": {}}]

        bot._queue_deferred(FakeUpdate(), first_calls, "doing orgo", "reply1", None)
        first_pending = bot.pending_deferred
        first_task = first_pending.task

        bot._queue_deferred(FakeUpdate(), second_calls, "actually still on last task", "reply2", None)
        second_pending = bot.pending_deferred

        assert second_pending is not first_pending
        assert second_pending.calls == second_calls

        await asyncio.sleep(0)  # let cancellation propagate
        assert first_task.cancelled() or first_task.done()
        mock_exec.assert_not_called()

        bot._cancel_pending_deferred()


async def test_fire_deferred_is_noop_if_pending_was_replaced():
    """Guards a race: a stale timer firing after a newer action replaced it must no-op."""
    _reset_bot_state()
    stale = bot.PendingDeferred(
        calls=[{"function": "switch_active_task", "params": {}}],
        user_message="stale",
        reply="stale reply",
        update=FakeUpdate(),
    )
    fresh = bot.PendingDeferred(
        calls=[{"function": "resume_previous_task", "params": {}}],
        user_message="fresh",
        reply="fresh reply",
        update=FakeUpdate(),
    )
    bot.pending_deferred = fresh

    with patch.object(bot, "_execute_and_reply", new_callable=AsyncMock) as mock_exec:
        await bot._fire_deferred(stale)
        mock_exec.assert_not_called()
        # The fresh pending action must be untouched by the stale fire.
        assert bot.pending_deferred is fresh

    bot.pending_deferred = None


def main():
    test_classify_deferral_true_for_compound_functions()
    test_classify_deferral_false_for_simple_functions()
    test_classify_deferral_true_if_any_call_is_compound()
    test_classify_deferral_empty_list_is_false()
    asyncio.run(test_cancel_before_fire_prevents_execution())
    asyncio.run(test_queue_fires_after_window())
    asyncio.run(test_requeue_cancels_previous_pending())
    asyncio.run(test_fire_deferred_is_noop_if_pending_was_replaced())
    print("All defer buffer tests passed.")


if __name__ == "__main__":
    main()
