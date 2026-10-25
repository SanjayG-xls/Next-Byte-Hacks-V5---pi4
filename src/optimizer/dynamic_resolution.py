"""
BytePi4 Dynamic Resolution Controller

Adapts the target workload resolution based on normalized system state.

Responsibility:
    - Maintains an ordered list of resolution levels (highest to lowest).
    - Degrades resolution when the system is under high or critical load.
    - Gradually restores resolution when load decreases.
    - Uses hysteresis and minimum hold cycles to prevent oscillation.
    - Does NOT capture frames, resize images, or run inference.
    - Exposes a policy decision that the runtime will consume.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import yaml

from src.optimizer.resource_manager import SystemState

logger = logging.getLogger(__name__)

DEFAULT_DYNAMIC_RESOLUTION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "levels": [
        [640, 480],
        [512, 384],
        [480, 360],
        [320, 240],
    ],
    "default_level": 0,
    "recovery_step": 1,
    "degradation_step": 1,
    "minimum_hold_cycles": 3,
}


class DynamicResolution:
    """
    Controls the target workload resolution by selecting from a predefined
    set of resolution levels based on system state.

    Level index 0 is the highest resolution; the last index is the lowest.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = "configs/optimization.yaml",
    ):
        """
        Initialize DynamicResolution with a config dict or YAML config path.
        """
        self.config = self._load_config(config, config_path)

        self.enabled: bool = bool(self.config.get("enabled", True))

        raw_levels = self.config.get(
            "levels", DEFAULT_DYNAMIC_RESOLUTION_CONFIG["levels"]
        )
        self.levels: List[Tuple[int, int]] = self._parse_levels(raw_levels)

        if not self.levels:
            logger.warning("No valid resolution levels configured; using default.")
            self.levels = [(640, 480), (512, 384), (480, 360), (320, 240)]

        self.max_level_index: int = len(self.levels) - 1

        default_level = int(self.config.get("default_level", 0))
        self.current_level: int = max(0, min(default_level, self.max_level_index))

        self.recovery_step: int = max(1, int(self.config.get("recovery_step", 1)))
        self.degradation_step: int = max(1, int(self.config.get("degradation_step", 1)))
        self.minimum_hold_cycles: int = max(0, int(self.config.get("minimum_hold_cycles", 3)))

        self._hold_counter: int = 0

    def _load_config(
        self, config: Optional[Dict[str, Any]], config_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load dynamic resolution config from dict or YAML file."""
        loaded_config = None

        if config is not None:
            loaded_config = config
        elif config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        loaded_config = content.get("dynamic_resolution", {})
            except Exception as e:
                logger.warning(
                    "Failed to load dynamic_resolution config from %s: %s",
                    config_path,
                    e,
                )

        if not loaded_config:
            loaded_config = {}

        merged = dict(DEFAULT_DYNAMIC_RESOLUTION_CONFIG)
        merged.update(loaded_config)
        return merged

    @staticmethod
    def _parse_levels(raw_levels: Any) -> List[Tuple[int, int]]:
        """Parse resolution levels from config into list of (width, height) tuples."""
        parsed: List[Tuple[int, int]] = []
        if not isinstance(raw_levels, list):
            return parsed
        for entry in raw_levels:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                try:
                    w, h = int(entry[0]), int(entry[1])
                    if w > 0 and h > 0:
                        parsed.append((w, h))
                except (ValueError, TypeError):
                    continue
        return parsed

    def update(
        self,
        system_state: Union[SystemState, str],
        fps: Optional[float] = None,
        latency: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        Update the target resolution based on system state.

        Args:
            system_state: Current system state from ResourceManager.
            fps: Current FPS (informational, state already encodes pressure).
            latency: Current latency in ms (informational).

        Returns:
            The current target resolution as (width, height).
        """
        if not self.enabled:
            return self.levels[self.current_level]

        state_str = (
            system_state.value
            if isinstance(system_state, SystemState)
            else str(system_state).upper()
        )

        # Determine target level index based on state
        if state_str == "CRITICAL":
            target_level = self.max_level_index
        elif state_str == "HIGH_LOAD":
            target_level = min(
                self.current_level + self.degradation_step, self.max_level_index
            )
        elif state_str == "LOW_LOAD":
            target_level = 0
        else:  # NORMAL
            target_level = self.current_level

        old_level = self.current_level

        if target_level > self.current_level:
            # Degradation: move toward lower resolution immediately
            self.current_level = target_level
            self._hold_counter = 0
            if old_level != self.current_level:
                logger.info(
                    "Resolution degraded: %s -> %s",
                    self.levels[old_level],
                    self.levels[self.current_level],
                )
        elif target_level < self.current_level:
            # Recovery: hold, then step gradually toward higher resolution
            self._hold_counter += 1
            if self._hold_counter >= self.minimum_hold_cycles:
                new_level = max(target_level, self.current_level - self.recovery_step)
                self.current_level = max(0, new_level)
                self._hold_counter = 0
                if old_level != self.current_level:
                    logger.info(
                        "Resolution recovered: %s -> %s",
                        self.levels[old_level],
                        self.levels[self.current_level],
                    )
        else:
            # Stable â€” increment hold counter so future recovery can proceed
            self._hold_counter += 1

        return self.levels[self.current_level]

    def get_resolution(self) -> Tuple[int, int]:
        """Return the current target resolution as (width, height)."""
        return self.levels[self.current_level]

    def get_level_index(self) -> int:
        """Return the current resolution level index."""
        return self.current_level
