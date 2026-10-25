"""
BytePi4 Runtime Engine

Pi 4 deployment flow:

    Camera Frame
         |
         v
    Dynamic Resolution
         |
         v
    Adaptive Frame Skip
         |
         v
    YOLO INT8/TFLite
         |
         v
    Profiler
         |
         v
    Resource Manager
         |
         v
    Optimization Controller
         |
         +----------------------+
         |                      |
         v                      v
    Frame Skip            Resolution
         |                      |
         +----------+-----------+
                    |
                    v
              Next Camera Frame

Supports:
    - Original YOLO .pt
    - Quantized YOLO INT8 .tflite
    - Raspberry Pi 4 CPU inference
    - Laptop camera / OpenCV camera
    - Adaptive frame skipping
    - Dynamic resolution
    - Runtime profiling
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import cv2

from src.profiler.profiler import Profiler
from src.runtime.scheduler import OptimizationScheduler
from src.optimizer.controller import OptimizationController
from src.models.yolo_model import YOLOModel


logger = logging.getLogger("BytePi4Runtime")


class BytePi4Runtime:

    def __init__(
        self,
        profiler: Optional[Profiler] = None,
        scheduler: Optional[OptimizationScheduler] = None,
        controller: Optional[OptimizationController] = None,
        scheduler_interval: float = 1.0,
        model_path: Optional[str] = None,
        model_confidence: float = 0.25,
        model_image_size: int = 640,
        device: Optional[str] = None,
    ):
        """
        Initialize BytePi4 runtime.

        Parameters
        ----------
        model_path:
            Example:
                models/optimized/best_int8.tflite

            or:
                models/original/best.pt

        model_image_size:
            Initial YOLO inference size.

        device:
            For Pi 4 use:
                "cpu"
        """

        # ======================================================
        # CORE COMPONENTS
        # ======================================================

        self.profiler = profiler or Profiler()

        self.scheduler = (
            scheduler
            or OptimizationScheduler(
                interval_seconds=scheduler_interval
            )
        )

        self.controller = (
            controller
            or OptimizationController()
        )

        # ======================================================
        # YOLO MODEL
        # ======================================================

        self.model: Optional[YOLOModel] = None

        if model_path is not None:

            self.model = YOLOModel(
                model_path=model_path,
                task="detect",
                confidence=model_confidence,
                image_size=model_image_size,
                device=device,
            )

            logger.info(
                "YOLO model loaded: %s",
                model_path,
            )

        # ======================================================
        # RUNTIME STATE
        # ======================================================

        self.running = False

        self.start_time: Optional[float] = None

        self.frame_count = 0
        self.processed_frames = 0
        self.optimization_cycles = 0
        self.cycle_count = 0

        self.total_inferences = 0

        self.last_decision = None

        self.system_state = "UNKNOWN"

        # ======================================================
        # INFERENCE STATE
        # ======================================================

        self.last_inference_result: Optional[
            Dict[str, Any]
        ] = None

        self.last_inference_time_ms = 0.0

        # ======================================================
        # CAMERA / FRAME STATE
        # ======================================================

        self.last_frame: Optional[Any] = None

        self.last_processed_frame: Optional[Any] = None

        self.last_detection_count = 0

        # Current optimization policies.
        self.current_frame_skip = 0

        self.current_resolution: Optional[
            Tuple[int, int]
        ] = None

        logger.info(
            "BytePi4 Runtime initialized"
        )

    # ==========================================================
    # START
    # ==========================================================

    def start(self) -> None:
        """
        Start the runtime.
        """

        if self.running:
            return

        self.running = True

        self.start_time = time.monotonic()

        logger.info(
            "BytePi4 Runtime started"
        )

    # ==========================================================
    # STOP
    # ==========================================================

    def stop(self) -> None:
        """
        Stop the runtime.
        """

        if not self.running:
            return

        self.running = False

        logger.info(
            "BytePi4 Runtime stopped"
        )

    # ==========================================================
    # FRAME RESIZING
    # ==========================================================

    def resize_frame(
        self,
        frame: Any,
        resolution: Optional[Tuple[int, int]],
    ) -> Any:
        """
        Resize a camera frame according to the optimization policy.

        resolution:
            (width, height)
        """

        if frame is None:
            return frame

        if resolution is None:
            return frame

        width, height = resolution

        if width <= 0 or height <= 0:
            return frame

        try:

            return cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        except Exception as exc:

            logger.warning(
                "Frame resize failed: %s",
                exc,
            )

            return frame

    # ==========================================================
    # FRAME PROCESSING
    # ==========================================================

    def process_frame(
        self,
        frame: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Process one camera frame.

        Flow:

            Camera
              |
              v
         Frame Counter
              |
              v
         Adaptive Skip
              |
              v
        Dynamic Resolution
              |
              v
            YOLO
              |
              v
           Profiler

        Returns
        -------
        dict or None

        None means the frame was intentionally skipped.
        """

        if not self.running:

            raise RuntimeError(
                "BytePi4 Runtime is not running. "
                "Call start() first."
            )

        if frame is None:

            return None

        # ------------------------------------------------------
        # Store latest frame
        # ------------------------------------------------------

        self.last_frame = frame

        # ------------------------------------------------------
        # Count frame
        # ------------------------------------------------------

        frame_number = self.frame_count

        self.frame_count += 1

        # ------------------------------------------------------
        # Adaptive frame skipping
        # ------------------------------------------------------

        try:

            should_run = (
                self.controller.adaptive_skip.should_run(
                    frame_number
                )
            )

        except AttributeError:

            should_run = True

        if not should_run:

            return {
                "processed": False,
                "skipped": True,
                "frame_number": frame_number,
                "result": None,
            }

        # ------------------------------------------------------
        # Get dynamic resolution
        # ------------------------------------------------------

        try:

            resolution = (
                self.controller
                .dynamic_resolution
                .get_resolution()
            )

        except AttributeError:

            resolution = None

        self.current_resolution = resolution

        # ------------------------------------------------------
        # Resize frame
        # ------------------------------------------------------

        processed_frame = self.resize_frame(
            frame,
            resolution,
        )

        self.last_processed_frame = processed_frame

        # ------------------------------------------------------
        # YOLO inference
        # ------------------------------------------------------

        result = self.run_inference(
            processed_frame
        )

        self.processed_frames += 1

        return {
            "processed": True,
            "skipped": False,
            "frame_number": frame_number,
            "resolution": resolution,
            "result": result,
        }

    # ==========================================================
    # YOLO INFERENCE
    # ==========================================================

    def run_inference(
        self,
        frame: Any,
    ) -> Dict[str, Any]:
        """
        Run YOLO inference on one frame.
        """

        if self.model is None:

            raise RuntimeError(
                "No YOLO model is loaded. "
                "Initialize BytePi4Runtime with "
                "model_path=..."
            )

        if not self.running:

            raise RuntimeError(
                "BytePi4 Runtime is not running."
            )

        # ------------------------------------------------------
        # Run inference
        # ------------------------------------------------------

        result = self.model.predict(
            frame,
            verbose=False,
        )

        # ------------------------------------------------------
        # Store result
        # ------------------------------------------------------

        self.last_inference_result = result

        self.last_inference_time_ms = float(
            result.get(
                "inference_time_ms",
                0.0,
            )
        )

        self.total_inferences += 1

        self.last_detection_count = len(
            result.get(
                "detections",
                [],
            )
        )

        # ------------------------------------------------------
        # Update profiler
        # ------------------------------------------------------

        try:

            self.profiler.set_inference_time(
                self.last_inference_time_ms
            )

        except Exception as exc:

            logger.warning(
                "Profiler inference update failed: %s",
                exc,
            )

        return result

    # ==========================================================
    # OPTIMIZATION CYCLE
    # ==========================================================

    def run_cycle(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
    ):
        """
        Execute one optimization evaluation.

        Flow:

            Profiler
                |
                v
            Scheduler
                |
                v
            Controller
                |
                v
        Optimization Decision
                |
          +-----+-----+
          |           |
          v           v
       Skip       Resolution
        """

        if not self.running:

            raise RuntimeError(
                "BytePi4 Runtime is not running."
            )

        # ------------------------------------------------------
        # Current time
        # ------------------------------------------------------

        now = (
            time.monotonic()
            if current_time is None
            else float(current_time)
        )

        # ------------------------------------------------------
        # Collect metrics
        # ------------------------------------------------------

        if metrics is None:

            metrics = self._collect_metrics()

        else:

            metrics = dict(metrics)

        # ------------------------------------------------------
        # Include latest inference latency
        # ------------------------------------------------------

        if self.last_inference_time_ms > 0:

            metrics["inference_ms"] = (
                self.last_inference_time_ms
            )

        # ------------------------------------------------------
        # Scheduler
        # ------------------------------------------------------

        should_evaluate = (
            self.scheduler.should_evaluate(
                current_time=now
            )
        )

        if not should_evaluate:

            return None

        self.scheduler.mark_evaluated(
            current_time=now
        )

        # ------------------------------------------------------
        # Optimization controller
        # ------------------------------------------------------

        decision = self.controller.evaluate(
            metrics
        )

        self.last_decision = decision

        self.optimization_cycles += 1

        self.cycle_count = (
            self.optimization_cycles
        )

        self.system_state = decision.state

        # ------------------------------------------------------
        # Store current policies
        # ------------------------------------------------------

        try:

            self.current_frame_skip = (
                self.controller
                .adaptive_skip
                .get_skip()
            )

        except AttributeError:

            self.current_frame_skip = (
                getattr(
                    decision,
                    "frame_skip",
                    0,
                )
            )

        try:

            self.current_resolution = (
                self.controller
                .dynamic_resolution
                .get_resolution()
            )

        except AttributeError:

            self.current_resolution = (
                getattr(
                    decision,
                    "resolution",
                    None,
                )
            )

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        logger.info(
            "Optimization cycle %d | "
            "state=%s | "
            "action=%s | "
            "skip=%s | "
            "resolution=%s",
            self.optimization_cycles,
            decision.state,
            decision.action,
            self.current_frame_skip,
            self.current_resolution,
        )

        return decision

    # ==========================================================
    # METRIC COLLECTION
    # ==========================================================

    def _collect_metrics(
        self,
    ) -> Dict[str, Any]:
        """
        Collect metrics from Profiler.
        """

        try:

            metrics = self.profiler.get_metrics()

            if metrics is None:
                metrics = {}

            metrics = dict(metrics)

        except Exception as exc:

            logger.warning(
                "Profiler metric collection failed: %s",
                exc,
            )

            metrics = {}

        # ------------------------------------------------------
        # Always use latest YOLO latency
        # ------------------------------------------------------

        if self.last_inference_time_ms > 0:

            metrics["inference_ms"] = (
                self.last_inference_time_ms
            )

        return metrics

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Return complete runtime status.
        """

        metrics = self._collect_metrics()

        model_info = self.get_model_info()

        return {

            "running":
                self.running,

            "status":
                (
                    "running"
                    if self.running
                    else "stopped"
                ),

            "frame_count":
                self.frame_count,

            "processed_frames":
                self.processed_frames,

            "optimization_cycles":
                self.optimization_cycles,

            "cycle_count":
                self.cycle_count,

            "total_inferences":
                self.total_inferences,

            "system_state":
                self.system_state,

            "frame_skip":
                self.current_frame_skip,

            "resolution":
                self.current_resolution,

            "latest_decision":
                self.last_decision,

            "uptime_seconds":
                self._get_uptime(),

            "fps":
                metrics.get("fps", 0.0),

            "cpu":
                metrics.get("cpu", 0.0),

            "memory":
                metrics.get("memory", 0.0),

            "temperature":
                metrics.get("temperature"),

            "latency":
                (
                    self.last_inference_time_ms
                    if self.last_inference_time_ms > 0
                    else metrics.get(
                        "inference_ms",
                        0.0,
                    )
                ),

            "model_loaded":
                model_info.get(
                    "loaded",
                    False,
                ),

            "model_path":
                model_info.get(
                    "model_path"
                ),

            "model_type":
                model_info.get(
                    "model_type"
                ),

            "model_format":
                model_info.get(
                    "format"
                ),

            "model_size_bytes":
                model_info.get(
                    "model_size_bytes",
                    0,
                ),

            "model_size_mb":
                model_info.get(
                    "model_size_mb",
                    0.0,
                ),

            "inference_time_ms":
                self.last_inference_time_ms,

            "detections":
                self.last_detection_count,
        }

    # ==========================================================
    # MODEL INFORMATION
    # ==========================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        if self.model is None:

            return {
                "loaded": False,
                "model_path": None,
                "model_type": None,
                "format": None,
                "model_size_bytes": 0,
                "model_size_mb": 0.0,
            }

        info = self.model.get_model_info()

        info["loaded"] = True

        return info

    # ==========================================================
    # LATEST DECISION
    # ==========================================================

    def get_latest_decision(self):

        return self.last_decision

    # ==========================================================
    # LAST INFERENCE
    # ==========================================================

    def get_last_inference(
        self,
    ) -> Optional[Dict[str, Any]]:

        return self.last_inference_result

    # ==========================================================
    # RUNNING CHECK
    # ==========================================================

    def is_running(self) -> bool:

        return self.running

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def get_statistics(
        self,
    ) -> Dict[str, Any]:

        return {

            "status":
                self.get_status(),

            "running":
                self.running,

            "frame_count":
                self.frame_count,

            "processed_frames":
                self.processed_frames,

            "optimization_cycles":
                self.optimization_cycles,

            "cycle_count":
                self.cycle_count,

            "total_inferences":
                self.total_inferences,

            "system_state":
                self.system_state,

            "uptime_seconds":
                self._get_uptime(),

            "scheduler":
                self.scheduler.get_statistics(),

            "model":
                self.get_model_info(),

            "inference":
                {
                    "total":
                        self.total_inferences,

                    "last_time_ms":
                        self.last_inference_time_ms,

                    "detections":
                        self.last_detection_count,
                },
        }

    # ==========================================================
    # UPTIME
    # ==========================================================

    def _get_uptime(self) -> float:

        if self.start_time is None:

            return 0.0

        return (
            time.monotonic()
            - self.start_time
        )

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self) -> None:
        """
        Reset runtime statistics.
        """

        self.frame_count = 0

        self.processed_frames = 0

        self.optimization_cycles = 0

        self.cycle_count = 0

        self.total_inferences = 0

        self.last_decision = None

        self.system_state = "UNKNOWN"

        self.last_inference_result = None

        self.last_inference_time_ms = 0.0

        self.last_detection_count = 0

        self.current_frame_skip = 0

        self.current_resolution = None

        self.last_frame = None

        self.last_processed_frame = None

        try:

            self.scheduler.reset()

        except AttributeError:

            pass

        try:

            self.controller.adaptive_skip.reset_frame_counter()

        except AttributeError:

            pass

        if self.running:

            self.start_time = (
                time.monotonic()
            )

        else:

            self.start_time = None

        logger.info(
            "BytePi4 Runtime reset"
        )

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:
        """
        Cleanly shut down runtime.
        """

        self.stop()

        logger.info(
            "BytePi4 Runtime shutdown complete"
        )