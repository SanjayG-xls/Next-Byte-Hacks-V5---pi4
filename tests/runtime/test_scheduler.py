import pytest

from src.runtime.scheduler import OptimizationScheduler


def test_scheduler_creation():
    scheduler = OptimizationScheduler()

    assert scheduler is not None
    assert scheduler.is_enabled() is True
    assert scheduler.get_interval() == 1.0


def test_first_evaluation_runs_immediately():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0
    )

    assert scheduler.should_evaluate(
        current_time=0.0
    ) is True


def test_scheduler_waits_for_interval():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0
    )

    scheduler.mark_evaluated(
        current_time=0.0
    )

    assert scheduler.should_evaluate(
        current_time=0.5
    ) is False

    assert scheduler.should_evaluate(
        current_time=1.0
    ) is True


def test_mark_evaluated_updates_statistics():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0
    )

    scheduler.mark_evaluated(
        current_time=0.0
    )

    scheduler.mark_evaluated(
        current_time=1.0
    )

    stats = scheduler.get_statistics()

    assert stats["total_cycles"] == 2
    assert stats["completed_intervals"] == 1
    assert stats["average_interval"] == pytest.approx(1.0)


def test_interval_clamping():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0,
        min_interval_seconds=0.25,
        max_interval_seconds=5.0,
    )

    scheduler.set_interval(0.01)

    assert scheduler.get_interval() == 0.25

    scheduler.set_interval(10.0)

    assert scheduler.get_interval() == 5.0


def test_adaptive_interval_high_load():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0
    )

    original = scheduler.get_interval()

    new_interval = scheduler.adapt_interval(
        cpu=95.0,
        temperature=85.0,
        fps=8.0,
        latency_ms=250.0,
    )

    assert new_interval > original
    assert new_interval <= scheduler.max_interval


def test_adaptive_interval_low_load():
    scheduler = OptimizationScheduler(
        interval_seconds=1.0
    )

    original = scheduler.get_interval()

    new_interval = scheduler.adapt_interval(
        cpu=20.0,
        temperature=45.0,
        fps=40.0,
        latency_ms=20.0,
    )

    assert new_interval < original
    assert new_interval >= scheduler.min_interval


def test_disable_scheduler():
    scheduler = OptimizationScheduler()

    scheduler.disable()

    assert scheduler.is_enabled() is False

    assert scheduler.should_evaluate(
        current_time=0.0
    ) is False


def test_enable_scheduler():
    scheduler = OptimizationScheduler()

    scheduler.disable()
    scheduler.enable()

    assert scheduler.is_enabled() is True

    assert scheduler.should_evaluate(
        current_time=0.0
    ) is True


def test_scheduler_reset():
    scheduler = OptimizationScheduler()

    scheduler.mark_evaluated(
        current_time=0.0
    )

    scheduler.mark_evaluated(
        current_time=1.0
    )

    scheduler.reset()

    stats = scheduler.get_statistics()

    assert stats["total_cycles"] == 0
    assert stats["skipped_cycles"] == 0
    assert stats["average_interval"] == 0.0
    assert stats["completed_intervals"] == 0

    assert scheduler.should_evaluate(
        current_time=2.0
    ) is True