from __future__ import annotations

import time
from typing import Any, Dict, Optional

import psutil


class Profiler:

    def __init__(self):
        self.start_time = time.monotonic()

        self._fps = 0.0
        self._cpu = 0.0
        self._memory = 0.0
        self._temperature: Optional[float] = None
        self._inference_time_ms = 0.0

        self._frame_count = 0
        self._last_frame_time: Optional[float] = None

        self._fps_samples = []
        self._latency_samples = []

        psutil.cpu_percent(interval=None)

    def update(self) -> Dict[str, Any]:
        self._cpu = self.get_cpu()
        self._memory = self.get_memory()
        self._temperature = self.get_temperature()

        return self.get_metrics()

    def update_fps(self, frames: int = 1) -> float:
        frames = max(0, int(frames))
        self._frame_count += frames

        now = time.monotonic()

        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time

            if elapsed > 0:
                instant_fps = frames / elapsed

                if self._fps <= 0:
                    self._fps = instant_fps
                else:
                    self._fps = (
                        self._fps * 0.7
                        + instant_fps * 0.3
                    )

                self._fps_samples.append(
                    self._fps
                )

                if len(self._fps_samples) > 300:
                    self._fps_samples.pop(0)

        self._last_frame_time = now

        return self._fps

    def update_inference_time(
        self,
        inference_time_ms: float,
    ) -> float:
        try:
            value = float(
                inference_time_ms
            )
        except (
            TypeError,
            ValueError,
        ):
            return self._inference_time_ms

        if value < 0:
            value = 0.0

        self._inference_time_ms = value

        self._latency_samples.append(
            value
        )

        if len(self._latency_samples) > 300:
            self._latency_samples.pop(0)

        return self._inference_time_ms

    def set_inference_time(
        self,
        inference_time_ms: float,
    ) -> float:
        return self.update_inference_time(
            inference_time_ms
        )

    def get_cpu(self) -> float:
        try:
            self._cpu = float(
                psutil.cpu_percent(
                    interval=None
                )
            )
        except Exception:
            self._cpu = 0.0

        return self._cpu

    def get_memory(self) -> float:
        try:
            self._memory = float(
                psutil.virtual_memory().percent
            )
        except Exception:
            self._memory = 0.0

        return self._memory

    def get_temperature(self) -> Optional[float]:
        try:
            temperatures = (
                psutil.sensors_temperatures()
            )

            preferred = [
                "cpu_thermal",
                "cpu-thermal",
                "coretemp",
                "soc_thermal",
            ]

            for name in preferred:
                entries = temperatures.get(
                    name,
                    []
                )

                for entry in entries:
                    if entry.current is not None:
                        self._temperature = float(
                            entry.current
                        )
                        return self._temperature

            for entries in temperatures.values():
                for entry in entries:
                    if entry.current is not None:
                        self._temperature = float(
                            entry.current
                        )
                        return self._temperature

        except Exception:
            pass

        try:
            thermal_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
            ]

            for path in thermal_paths:
                try:
                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                    ) as file:
                        value = float(
                            file.read().strip()
                        )

                    if value > 1000:
                        value /= 1000.0

                    self._temperature = value

                    return self._temperature

                except Exception:
                    continue

        except Exception:
            pass

        return self._temperature

    def get_fps(self) -> float:
        return self._fps

    def get_inference_time(self) -> float:
        return self._inference_time_ms

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "cpu": self.get_cpu(),
            "memory": self.get_memory(),
            "temperature": self.get_temperature(),
            "fps": self.get_fps(),
            "inference_ms": self.get_inference_time(),
            "uptime_seconds": (
                time.monotonic()
                - self.start_time
            ),
            "frame_count": self._frame_count,
        }

    def get_average_fps(self) -> float:
        if not self._fps_samples:
            return 0.0

        return sum(
            self._fps_samples
        ) / len(
            self._fps_samples
        )

    def get_average_inference_time(self) -> float:
        if not self._latency_samples:
            return 0.0

        return sum(
            self._latency_samples
        ) / len(
            self._latency_samples
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "fps": self.get_fps(),
            "average_fps": self.get_average_fps(),
            "cpu": self.get_cpu(),
            "memory": self.get_memory(),
            "temperature": self.get_temperature(),
            "inference_ms": self.get_inference_time(),
            "average_inference_ms": (
                self.get_average_inference_time()
            ),
            "frame_count": self._frame_count,
            "uptime_seconds": (
                time.monotonic()
                - self.start_time
            ),
        }

    def reset(self) -> None:
        self.start_time = time.monotonic()

        self._fps = 0.0
        self._cpu = 0.0
        self._memory = 0.0
        self._temperature = None
        self._inference_time_ms = 0.0

        self._frame_count = 0
        self._last_frame_time = None

        self._fps_samples.clear()
        self._latency_samples.clear()

        psutil.cpu_percent(interval=None)