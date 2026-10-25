from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import psutil
import yaml


logger = logging.getLogger("BytePi4ResourceManager")


class SystemState(str, Enum):
    LOW_LOAD = "LOW_LOAD"
    NORMAL = "NORMAL"
    HIGH_LOAD = "HIGH_LOAD"
    CRITICAL = "CRITICAL"


class ThermalState(str, Enum):
    NORMAL = "NORMAL"
    WARM = "WARM"
    HOT = "HOT"
    CRITICAL = "CRITICAL"


@dataclass
class ResourceEvaluation:
    state: SystemState
    cpu: float
    memory: float
    temperature: Optional[float]
    fps: float
    latency_ms: float
    thermal_state: ThermalState
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "cpu": self.cpu,
            "memory": self.memory,
            "temperature": self.temperature,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "thermal_state": self.thermal_state.value,
            "reason": self.reason,
        }


DEFAULT_CONFIG = {
    "thresholds": {
        "cpu": {
            "low": 30.0,
            "high": 75.0,
            "critical": 90.0,
        },
        "memory": {
            "low": 40.0,
            "high": 80.0,
            "critical": 90.0,
        },
        "temperature": {
            "low": 50.0,
            "high": 70.0,
            "critical": 80.0,
        },
        "latency_ms": {
            "high": 100.0,
            "critical": 200.0,
        },
        "fps": {
            "low_target": 15.0,
        },
    },
    "stability": {
        "consecutive_samples_required": 2,
        "history_window": 5,
    },
}


class ResourceManager:

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = "configs/optimization.yaml",
    ):
        self.config = self._load_config(
            config,
            config_path,
        )

        thresholds = self.config.get(
            "thresholds",
            {},
        )

        self.cpu_thresholds = thresholds.get(
            "cpu",
            DEFAULT_CONFIG["thresholds"]["cpu"],
        )

        self.memory_thresholds = thresholds.get(
            "memory",
            DEFAULT_CONFIG["thresholds"]["memory"],
        )

        self.temperature_thresholds = thresholds.get(
            "temperature",
            DEFAULT_CONFIG["thresholds"]["temperature"],
        )

        self.latency_thresholds = thresholds.get(
            "latency_ms",
            DEFAULT_CONFIG["thresholds"]["latency_ms"],
        )

        self.fps_thresholds = thresholds.get(
            "fps",
            DEFAULT_CONFIG["thresholds"]["fps"],
        )

        stability = self.config.get(
            "stability",
            {},
        )

        self.consecutive_samples_required = max(
            1,
            int(
                stability.get(
                    "consecutive_samples_required",
                    2,
                )
            ),
        )

        self.history_window = max(
            1,
            int(
                stability.get(
                    "history_window",
                    5,
                )
            ),
        )

        self.current_state = SystemState.NORMAL

        self._candidate_state: Optional[SystemState] = None
        self._candidate_count = 0

        self._history = []

        self._latest_evaluation: Optional[
            ResourceEvaluation
        ] = None

    def _load_config(
        self,
        config: Optional[Dict[str, Any]],
        config_path: Optional[str],
    ) -> Dict[str, Any]:

        if config is not None:
            loaded = config

        elif config_path and os.path.exists(config_path):

            try:
                with open(
                    config_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    content = yaml.safe_load(file)

                if isinstance(content, dict):
                    loaded = content.get(
                        "resource_manager",
                        {},
                    )
                else:
                    loaded = {}

            except Exception as exc:
                logger.warning(
                    "Failed to load resource manager config: %s",
                    exc,
                )
                loaded = {}

        else:
            loaded = {}

        merged = {
            "thresholds": {
                "cpu": dict(
                    DEFAULT_CONFIG["thresholds"]["cpu"]
                ),
                "memory": dict(
                    DEFAULT_CONFIG["thresholds"]["memory"]
                ),
                "temperature": dict(
                    DEFAULT_CONFIG["thresholds"]["temperature"]
                ),
                "latency_ms": dict(
                    DEFAULT_CONFIG["thresholds"]["latency_ms"]
                ),
                "fps": dict(
                    DEFAULT_CONFIG["thresholds"]["fps"]
                ),
            },
            "stability": dict(
                DEFAULT_CONFIG["stability"]
            ),
        }

        loaded_thresholds = loaded.get(
            "thresholds",
            {},
        )

        for key in merged["thresholds"]:

            if isinstance(
                loaded_thresholds.get(key),
                dict,
            ):
                merged["thresholds"][key].update(
                    loaded_thresholds[key]
                )

        if isinstance(
            loaded.get("stability"),
            dict,
        ):
            merged["stability"].update(
                loaded["stability"]
            )

        return merged

    def evaluate(
        self,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ResourceEvaluation:

        metrics = metrics or {}

        cpu = self._get_cpu(metrics)
        memory = self._get_memory(metrics)
        temperature = self._get_temperature(metrics)

        fps = self._number(
            metrics.get("fps"),
            0.0,
        )

        latency_ms = self._number(
            metrics.get(
                "inference_ms",
                metrics.get("latency_ms"),
            ),
            0.0,
        )

        raw_state = self._determine_state(
            cpu=cpu,
            memory=memory,
            temperature=temperature,
            fps=fps,
            latency_ms=latency_ms,
        )

        state = self._apply_stability(
            raw_state
        )

        thermal_state = self._get_thermal_state(
            temperature
        )

        reason = self._build_reason(
            state=state,
            cpu=cpu,
            memory=memory,
            temperature=temperature,
            fps=fps,
            latency_ms=latency_ms,
            thermal_state=thermal_state,
        )

        evaluation = ResourceEvaluation(
            state=state,
            cpu=cpu,
            memory=memory,
            temperature=temperature,
            fps=fps,
            latency_ms=latency_ms,
            thermal_state=thermal_state,
            reason=reason,
        )

        self._latest_evaluation = evaluation

        self._history.append(
            evaluation
        )

        if len(self._history) > self.history_window:
            self._history.pop(0)

        return evaluation

    def _determine_state(
        self,
        cpu: float,
        memory: float,
        temperature: Optional[float],
        fps: float,
        latency_ms: float,
    ) -> SystemState:

        critical = False
        high = False
        low = True

        if cpu >= float(
            self.cpu_thresholds.get(
                "critical",
                90.0,
            )
        ):
            critical = True

        if memory >= float(
            self.memory_thresholds.get(
                "critical",
                90.0,
            )
        ):
            critical = True

        if (
            temperature is not None
            and temperature
            >= float(
                self.temperature_thresholds.get(
                    "critical",
                    80.0,
                )
            )
        ):
            critical = True

        if latency_ms >= float(
            self.latency_thresholds.get(
                "critical",
                200.0,
            )
        ):
            critical = True

        if critical:
            return SystemState.CRITICAL

        if cpu >= float(
            self.cpu_thresholds.get(
                "high",
                75.0,
            )
        ):
            high = True

        if memory >= float(
            self.memory_thresholds.get(
                "high",
                80.0,
            )
        ):
            high = True

        if (
            temperature is not None
            and temperature
            >= float(
                self.temperature_thresholds.get(
                    "high",
                    70.0,
                )
            )
        ):
            high = True

        if latency_ms >= float(
            self.latency_thresholds.get(
                "high",
                100.0,
            )
        ):
            high = True

        if fps > 0 and fps < float(
            self.fps_thresholds.get(
                "low_target",
                15.0,
            )
        ):
            high = True

        if high:
            return SystemState.HIGH_LOAD

        if cpu > float(
            self.cpu_thresholds.get(
                "low",
                30.0,
            )
        ):
            low = False

        if memory > float(
            self.memory_thresholds.get(
                "low",
                40.0,
            )
        ):
            low = False

        if (
            temperature is not None
            and temperature
            > float(
                self.temperature_thresholds.get(
                    "low",
                    50.0,
                )
            )
        ):
            low = False

        if low:
            return SystemState.LOW_LOAD

        return SystemState.NORMAL

    def _apply_stability(
        self,
        candidate: SystemState,
    ) -> SystemState:

        if candidate == self.current_state:

            self._candidate_state = None
            self._candidate_count = 0

            return self.current_state

        if candidate != self._candidate_state:

            self._candidate_state = candidate
            self._candidate_count = 1

            return self.current_state

        self._candidate_count += 1

        if (
            self._candidate_count
            >= self.consecutive_samples_required
        ):
            self.current_state = candidate
            self._candidate_state = None
            self._candidate_count = 0

        return self.current_state

    def _get_thermal_state(
        self,
        temperature: Optional[float],
    ) -> ThermalState:

        if temperature is None:
            return ThermalState.NORMAL

        if temperature >= float(
            self.temperature_thresholds.get(
                "critical",
                80.0,
            )
        ):
            return ThermalState.CRITICAL

        if temperature >= float(
            self.temperature_thresholds.get(
                "high",
                70.0,
            )
        ):
            return ThermalState.HOT

        if temperature >= float(
            self.temperature_thresholds.get(
                "low",
                50.0,
            )
        ):
            return ThermalState.WARM

        return ThermalState.NORMAL

    def _build_reason(
        self,
        state: SystemState,
        cpu: float,
        memory: float,
        temperature: Optional[float],
        fps: float,
        latency_ms: float,
        thermal_state: ThermalState,
    ) -> str:

        if state == SystemState.CRITICAL:
            return (
                f"Critical resource pressure: "
                f"CPU={cpu:.1f}%, "
                f"memory={memory:.1f}%, "
                f"temperature={temperature if temperature is not None else 0:.1f}C, "
                f"FPS={fps:.1f}, "
                f"latency={latency_ms:.1f}ms"
            )

        if state == SystemState.HIGH_LOAD:
            return (
                f"High workload pressure: "
                f"CPU={cpu:.1f}%, "
                f"memory={memory:.1f}%, "
                f"temperature={temperature if temperature is not None else 0:.1f}C, "
                f"FPS={fps:.1f}, "
                f"latency={latency_ms:.1f}ms"
            )

        if state == SystemState.LOW_LOAD:
            return (
                f"Low system load: "
                f"CPU={cpu:.1f}%, "
                f"memory={memory:.1f}%, "
                f"temperature={temperature if temperature is not None else 0:.1f}C"
            )

        return (
            f"Normal operating conditions: "
            f"CPU={cpu:.1f}%, "
            f"memory={memory:.1f}%, "
            f"temperature={temperature if temperature is not None else 0:.1f}C, "
            f"FPS={fps:.1f}, "
            f"latency={latency_ms:.1f}ms, "
            f"thermal={thermal_state.value}"
        )

    def _get_cpu(
        self,
        metrics: Dict[str, Any],
    ) -> float:

        value = metrics.get("cpu")

        if value is not None:
            return self._number(
                value,
                0.0,
            )

        try:
            return float(
                psutil.cpu_percent(
                    interval=None
                )
            )
        except Exception:
            return 0.0

    def _get_memory(
        self,
        metrics: Dict[str, Any],
    ) -> float:

        value = metrics.get("memory")

        if value is not None:
            return self._number(
                value,
                0.0,
            )

        try:
            return float(
                psutil.virtual_memory().percent
            )
        except Exception:
            return 0.0

    def _get_temperature(
        self,
        metrics: Dict[str, Any],
    ) -> Optional[float]:

        value = metrics.get(
            "temperature"
        )

        if value is not None:
            return self._number(
                value,
                0.0,
            )

        try:
            temperatures = (
                psutil.sensors_temperatures()
            )

            for entries in temperatures.values():

                for entry in entries:

                    if entry.current is not None:
                        return float(
                            entry.current
                        )

        except Exception:
            pass

        try:
            path = "/sys/class/thermal/thermal_zone0/temp"

            if os.path.exists(path):

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

                return value

        except Exception:
            pass

        return None

    def get_state(self) -> SystemState:
        return self.current_state

    def get_latest_evaluation(
        self,
    ) -> Optional[ResourceEvaluation]:
        return self._latest_evaluation

    def get_history(self):
        return list(self._history)

    def get_status(self) -> Dict[str, Any]:

        if self._latest_evaluation is None:

            return {
                "state": self.current_state.value,
                "evaluation": None,
                "candidate_state": (
                    self._candidate_state.value
                    if self._candidate_state
                    else None
                ),
                "candidate_count": self._candidate_count,
            }

        return {
            "state": self.current_state.value,
            "evaluation": (
                self._latest_evaluation.to_dict()
            ),
            "candidate_state": (
                self._candidate_state.value
                if self._candidate_state
                else None
            ),
            "candidate_count": self._candidate_count,
        }

    def reset(self) -> None:

        self.current_state = SystemState.NORMAL
        self._candidate_state = None
        self._candidate_count = 0
        self._latest_evaluation = None
        self._history.clear()

    @staticmethod
    def _number(
        value: Any,
        default: float,
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return float(default)