"""
BytePi4 Adaptive Frame Skip Controller

Dynamically adjusts inference frequency based on normalized system state
and performance metrics.

Responsibility:
    - Determines frame scheduling policy (skip 0, 1, 2, ...).
    - Does NOT capture camera frames or execute inference.
    - Uses hysteresis and gradual recovery to prevent rapid oscillation.
"""

import logging
import os
from typing import Dict, Any, Optional, Union
import yaml

from src.optimizer.resource_manager import SystemState

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_SKIP_CONFIG = {
    "min_skip": 0,
    "max_skip": 4,
    "recovery_step": 1,
    "minimum_hold_cycles": 2,
    "state_skip_mapping": {
        "LOW_LOAD": 0,
        "NORMAL": 0,
        "HIGH_LOAD": 2,
        "CRITICAL": 4,
    },
}


class AdaptiveFrameSkip:
    """
    Determines whether an inference workload should run on a given frame number
    and dynamically adjusts skip frequency based on system state.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = "configs/optimization.yaml",
    ):
        """
        Initialize AdaptiveFrameSkip with either a config dict or YAML config path.
        """
        self.config = self._load_config(config, config_path)

        self.min_skip = int(self.config.get("min_skip", 0))
        self.max_skip = int(self.config.get("max_skip", 4))
        self.recovery_step = max(1, int(self.config.get("recovery_step", 1)))
        self.minimum_hold_cycles = max(0, int(self.config.get("minimum_hold_cycles", 2)))
        self.state_skip_mapping = self.config.get(
            "state_skip_mapping", DEFAULT_ADAPTIVE_SKIP_CONFIG["state_skip_mapping"]
        )

        self.current_skip: int = self.min_skip
        self._hold_counter: int = 0
        self._frame_counter: int = 0

    def _load_config(
        self, config: Optional[Dict[str, Any]], config_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load adaptive frame skip parameters from config dictionary or YAML file.
        """
        loaded_config = None

        if config is not None:
            loaded_config = config
        elif config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        loaded_config = content.get("adaptive_skip", {})
            except Exception as e:
                logger.warning("Failed to load adaptive_skip config from %s: %s", config_path, e)

        if not loaded_config:
            loaded_config = {}

        merged_config = dict(DEFAULT_ADAPTIVE_SKIP_CONFIG)
        merged_config.update(loaded_config)
        return merged_config

    def update(
        self,
        system_state: Union[SystemState, str],
        fps: Optional[float] = None,
        latency: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Update frame skip decision based on system state and performance metrics.

        Returns:
            int: The updated current frame skip level.
        """
        state_str = (
            system_state.value
            if isinstance(system_state, SystemState)
            else str(system_state).upper()
        )

        # Determine target skip based on state
        if state_str == "CRITICAL":
            target_skip = self.max_skip
        elif state_str == "HIGH_LOAD":
            mapped_skip = int(self.state_skip_mapping.get("HIGH_LOAD", 2))
            target_skip = max(self.current_skip + 1, mapped_skip)
        elif state_str == "LOW_LOAD":
            target_skip = self.min_skip
        else:  # NORMAL
            target_skip = self.min_skip

        # Clamp target skip to allowed bounds
        target_skip = max(self.min_skip, min(self.max_skip, target_skip))

        old_skip = self.current_skip

        if target_skip > self.current_skip:
            # Escalating load: increase skip immediately to safeguard performance
            self.current_skip = target_skip
            self._hold_counter = 0
            logger.debug("Frame skip increased: %d -> %d (State: %s)", old_skip, self.current_skip, state_str)
        elif target_skip < self.current_skip:
            # Decreasing load: hold state to prevent rapid oscillation, then recover gradually
            self._hold_counter += 1
            if self._hold_counter >= self.minimum_hold_cycles:
                new_skip = max(target_skip, self.current_skip - self.recovery_step)
                self.current_skip = max(self.min_skip, min(self.max_skip, new_skip))
                self._hold_counter = 0
                logger.debug("Frame skip gradually recovered: %d -> %d (State: %s)", old_skip, self.current_skip, state_str)
        else:
            self._hold_counter += 1

        return self.current_skip

    def should_run(self, frame_number: Optional[int] = None) -> bool:
        """
        Determine if the specified or current frame should execute inference.

        Args:
            frame_number (int, optional): The frame index. If None, internal counter is used.

        Returns:
            bool: True if inference should run, False if frame should be skipped.
        """
        if frame_number is None:
            frame_idx = self._frame_counter
            self._frame_counter += 1
        else:
            frame_idx = int(frame_number)

        if self.current_skip <= 0:
            return True

        period = self.current_skip + 1
        return (frame_idx % period) == 0

    def get_skip(self) -> int:
        """Return the current frame skip level."""
        return self.current_skip

    def reset_frame_counter(self):
        """Reset internal frame counter to zero."""
        self._frame_counter = 0
