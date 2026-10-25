"""
BytePi4 Runtime Metrics

Collects and exposes optimization metrics from the
runtime, profiler, and optimization engine.

This module is lightweight and works on both:
- Development laptop
- Raspberry Pi 4
"""

import time
from collections import deque


class Metrics:

    def __init__(self, history_size=300):

        self.history_size = max(
            10,
            int(history_size)
        )

        self.start_time = time.monotonic()

        self.samples = deque(
            maxlen=self.history_size
        )

        self.total_frames = 0
        self.processed_frames = 0

        self.last_metrics = {}

    # =========================================================
    # RECORD
    # =========================================================

    def record(
        self,
        fps=None,
        cpu=None,
        memory=None,
        temperature=None,
        latency=None,
        frame_skip=0,
        resolution=None,
        system_state="NORMAL",
        model=None,
    ):
        """
        Record one runtime metrics sample.
        """

        timestamp = time.monotonic()

        sample = {
            "timestamp": timestamp,
            "time": timestamp - self.start_time,

            "fps": self._number(fps),
            "cpu": self._number(cpu),
            "memory": self._number(memory),
            "temperature": self._number(
                temperature
            ),
            "latency": self._number(
                latency
            ),

            "frame_skip": int(
                frame_skip or 0
            ),

            "resolution": resolution,

            "system_state": system_state,

            "model": model,
        }

        self.samples.append(sample)

        self.last_metrics = sample

        return sample

    # =========================================================
    # FRAME COUNTERS
    # =========================================================

    def update_frames(
        self,
        total_frames,
        processed_frames,
    ):

        self.total_frames = max(
            0,
            int(total_frames)
        )

        self.processed_frames = max(
            0,
            int(processed_frames)
        )

    # =========================================================
    # PROCESSING RATIO
    # =========================================================

    def get_processing_ratio(self):

        if self.total_frames == 0:
            return 0.0

        return (
            self.processed_frames
            / self.total_frames
        )

    # =========================================================
    # CURRENT METRICS
    # =========================================================

    def get_current(self):

        return dict(
            self.last_metrics
        )

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(self):

        return list(
            self.samples
        )

    # =========================================================
    # AVERAGES
    # =========================================================

    def get_average(
        self,
        metric,
    ):

        values = []

        for sample in self.samples:

            value = sample.get(
                metric
            )

            if value is not None:
                values.append(value)

        if not values:
            return None

        return sum(values) / len(values)

    # =========================================================
    # MINIMUM
    # =========================================================

    def get_minimum(
        self,
        metric,
    ):

        values = []

        for sample in self.samples:

            value = sample.get(
                metric
            )

            if value is not None:
                values.append(value)

        if not values:
            return None

        return min(values)

    # =========================================================
    # MAXIMUM
    # =========================================================

    def get_maximum(
        self,
        metric,
    ):

        values = []

        for sample in self.samples:

            value = sample.get(
                metric
            )

            if value is not None:
                values.append(value)

        if not values:
            return None

        return max(values)

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_summary(self):

        return {
            "samples": len(
                self.samples
            ),

            "fps": {
                "current": self.get_current_value(
                    "fps"
                ),
                "average": self.get_average(
                    "fps"
                ),
                "minimum": self.get_minimum(
                    "fps"
                ),
                "maximum": self.get_maximum(
                    "fps"
                ),
            },

            "cpu": {
                "current": self.get_current_value(
                    "cpu"
                ),
                "average": self.get_average(
                    "cpu"
                ),
                "maximum": self.get_maximum(
                    "cpu"
                ),
            },

            "memory": {
                "current": self.get_current_value(
                    "memory"
                ),
                "average": self.get_average(
                    "memory"
                ),
                "maximum": self.get_maximum(
                    "memory"
                ),
            },

            "temperature": {
                "current": self.get_current_value(
                    "temperature"
                ),
                "average": self.get_average(
                    "temperature"
                ),
                "maximum": self.get_maximum(
                    "temperature"
                ),
            },

            "latency": {
                "current": self.get_current_value(
                    "latency"
                ),
                "average": self.get_average(
                    "latency"
                ),
                "maximum": self.get_maximum(
                    "latency"
                ),
            },

            "total_frames":
                self.total_frames,

            "processed_frames":
                self.processed_frames,

            "processing_ratio":
                self.get_processing_ratio(),

            "uptime":
                time.monotonic()
                - self.start_time,
        }

    # =========================================================
    # CURRENT VALUE
    # =========================================================

    def get_current_value(
        self,
        metric,
    ):

        return self.last_metrics.get(
            metric
        )

    # =========================================================
    # EXPORT
    # =========================================================

    def export_dict(self):

        return {
            "current":
                self.get_current(),

            "summary":
                self.get_summary(),

            "history":
                self.get_history(),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.samples.clear()

        self.total_frames = 0

        self.processed_frames = 0

        self.last_metrics = {}

        self.start_time = time.monotonic()

    # =========================================================
    # HELPER
    # =========================================================

    @staticmethod
    def _number(value):

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None