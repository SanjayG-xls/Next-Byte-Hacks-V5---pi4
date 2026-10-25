"""
BytePi4 Optimization Scheduler

Controls when the optimization controller should be evaluated.

Responsibilities:
    - Avoid evaluating the optimizer every frame.
    - Schedule periodic optimization cycles.
    - Adapt evaluation frequency based on system load.
    - Track scheduler statistics.

The scheduler does NOT:
    - collect hardware metrics
    - make optimization decisions
    - change resolution
    - change frame skipping
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerStats:
    """Runtime statistics maintained by the scheduler."""

    total_cycles: int = 0
    skipped_cycles: int = 0
    last_cycle_time: float = 0.0
    average_interval: float = 0.0
    completed_intervals: int = 0


class OptimizationScheduler:
    """
    Controls how frequently optimization evaluations occur.
    """

    def __init__(
        self,
        interval_seconds: float = 1.0,
        min_interval_seconds: float = 0.25,
        max_interval_seconds: float = 5.0,
    ):

        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be greater than zero"
            )

        if min_interval_seconds <= 0:
            raise ValueError(
                "min_interval_seconds must be greater than zero"
            )

        if max_interval_seconds < min_interval_seconds:
            raise ValueError(
                "max_interval_seconds must be greater than "
                "or equal to min_interval_seconds"
            )

        self.min_interval = float(
            min_interval_seconds
        )

        self.max_interval = float(
            max_interval_seconds
        )

        self.interval = self._clamp_interval(
            interval_seconds
        )

        self.last_evaluation_time: Optional[float] = None

        self.stats = SchedulerStats()

        self.enabled = True

    # ==========================================================
    # SCHEDULING
    # ==========================================================

    def should_evaluate(
        self,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Determine whether an optimization evaluation
        should happen.
        """

        if not self.enabled:

            self.stats.skipped_cycles += 1

            return False

        now = (
            time.monotonic()
            if current_time is None
            else float(current_time)
        )

        # First evaluation happens immediately.
        if self.last_evaluation_time is None:

            return True

        elapsed = (
            now - self.last_evaluation_time
        )

        if elapsed >= self.interval:

            return True

        self.stats.skipped_cycles += 1

        return False

    # ==========================================================
    # MARK EVALUATION
    # ==========================================================

    def mark_evaluated(
        self,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Mark an optimization evaluation as completed.
        """

        now = (
            time.monotonic()
            if current_time is None
            else float(current_time)
        )

        if self.last_evaluation_time is not None:

            actual_interval = (
                now - self.last_evaluation_time
            )

            if actual_interval >= 0:

                self._update_average_interval(
                    actual_interval
                )

        self.last_evaluation_time = now

        self.stats.total_cycles += 1

        self.stats.last_cycle_time = now

    # ==========================================================
    # INTERVAL CONTROL
    # ==========================================================

    def set_interval(
        self,
        interval_seconds: float,
    ) -> float:
        """
        Change the optimization evaluation interval.
        """

        self.interval = self._clamp_interval(
            interval_seconds
        )

        return self.interval

    def get_interval(self) -> float:
        """Return current evaluation interval."""

        return self.interval

    # ==========================================================
    # ADAPTIVE INTERVAL
    # ==========================================================

    def adapt_interval(
        self,
        cpu: Optional[float] = None,
        temperature: Optional[float] = None,
        fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> float:
        """
        Adapt scheduler frequency according to system load.

        High load:
            optimizer runs less frequently.

        Low load:
            optimizer runs more frequently.
        """

        load_score = 0.0
        samples = 0

        # ------------------------------------------------------
        # CPU
        # ------------------------------------------------------

        if cpu is not None:

            cpu = float(cpu)

            if cpu >= 90.0:
                load_score += 1.0

            elif cpu >= 75.0:
                load_score += 0.7

            elif cpu >= 50.0:
                load_score += 0.4

            else:
                load_score += 0.1

            samples += 1

        # ------------------------------------------------------
        # TEMPERATURE
        # ------------------------------------------------------

        if temperature is not None:

            temperature = float(
                temperature
            )

            if temperature >= 80.0:
                load_score += 1.0

            elif temperature >= 70.0:
                load_score += 0.7

            elif temperature >= 60.0:
                load_score += 0.4

            else:
                load_score += 0.1

            samples += 1

        # ------------------------------------------------------
        # FPS
        # ------------------------------------------------------

        if fps is not None:

            fps = float(fps)

            if fps < 10.0:
                load_score += 1.0

            elif fps < 20.0:
                load_score += 0.7

            elif fps < 30.0:
                load_score += 0.4

            else:
                load_score += 0.1

            samples += 1

        # ------------------------------------------------------
        # LATENCY
        # ------------------------------------------------------

        if latency_ms is not None:

            latency_ms = float(
                latency_ms
            )

            if latency_ms >= 200.0:
                load_score += 1.0

            elif latency_ms >= 100.0:
                load_score += 0.7

            elif latency_ms >= 50.0:
                load_score += 0.4

            else:
                load_score += 0.1

            samples += 1

        # No metrics available.
        if samples == 0:
            return self.interval

        average_load = (
            load_score / samples
        )

        # High load â†’ evaluate less frequently.
        if average_load >= 0.75:

            new_interval = (
                self.interval * 1.5
            )

        # Medium load.
        elif average_load >= 0.45:

            new_interval = (
                self.interval * 1.1
            )

        # Low load â†’ react faster.
        else:

            new_interval = (
                self.interval * 0.8
            )

        self.interval = self._clamp_interval(
            new_interval
        )

        return self.interval

    # ==========================================================
    # ENABLE / DISABLE
    # ==========================================================

    def enable(self) -> None:
        """Enable scheduler."""

        self.enabled = True

    def disable(self) -> None:
        """Disable scheduler."""

        self.enabled = False

    def is_enabled(self) -> bool:
        """Return scheduler state."""

        return self.enabled

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self) -> None:
        """Reset scheduler state and statistics."""

        self.last_evaluation_time = None

        self.stats = SchedulerStats()

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def get_statistics(self) -> dict:
        """
        Return scheduler statistics.

        These metrics can later be exported to
        Prometheus/Grafana.
        """

        return {
            "enabled": self.enabled,

            "interval_seconds":
                self.interval,

            "total_cycles":
                self.stats.total_cycles,

            "skipped_cycles":
                self.stats.skipped_cycles,

            "last_cycle_time":
                self.stats.last_cycle_time,

            "average_interval":
                self.stats.average_interval,

            "completed_intervals":
                self.stats.completed_intervals,
        }

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def _clamp_interval(
        self,
        value: float,
    ) -> float:
        """
        Keep interval between configured limits.
        """

        return max(
            self.min_interval,
            min(
                float(value),
                self.max_interval,
            ),
        )

    def _update_average_interval(
        self,
        interval: float,
    ) -> None:
        """
        Update running average of the actual interval
        between optimization evaluations.
        """

        self.stats.completed_intervals += 1

        count = (
            self.stats.completed_intervals
        )

        self.stats.average_interval += (
            interval
            - self.stats.average_interval
        ) / count