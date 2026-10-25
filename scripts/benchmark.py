
"""
BytePi4 Benchmark Engine

Benchmarks the real YOLO workload through BytePi4Runtime.

This version is configured for the BASELINE model:

    models/original/best.pt

Results:

    benchmarks/baseline/results.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
import statistics
from typing import Any, Dict, List

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# BYTE PILOT IMPORT
# ============================================================

from src.runtime.runtime import BytePi4Runtime


# ============================================================
# MODEL
# ============================================================

BASELINE_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "original",
    "best.pt",
)


# ============================================================
# BENCHMARK ENGINE
# ============================================================

class Benchmark:

    def __init__(
        self,
        model_path: str,
        duration: float = 30.0,
        camera: int = 0,
        output: str = "benchmarks/baseline/results.csv",
        confidence: float = 0.25,
        image_size: int = 640,
        optimization_interval: float = 1.0,
        show_window: bool = True,
    ):

        self.model_path = os.path.abspath(
            model_path
        )

        self.duration = max(
            1.0,
            float(duration)
        )

        self.camera = int(
            camera
        )

        self.output = os.path.abspath(
            output
        )

        self.confidence = float(
            confidence
        )

        self.image_size = int(
            image_size
        )

        self.optimization_interval = max(
            0.1,
            float(
                optimization_interval
            )
        )

        self.show_window = bool(
            show_window
        )

        self.samples: List[
            Dict[str, Any]
        ] = []

        self.inference_times: List[
            float
        ] = []

        self.total_detections = 0
        self.total_frames = 0
        self.successful_inferences = 0
        self.optimization_evaluations = 0

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print()
        print("============================================")
        print("          BytePi4 Benchmark Engine")
        print("============================================")

        print(
            f"Model      : {self.model_path}"
        )

        print(
            f"Duration   : {self.duration:.1f} seconds"
        )

        print(
            f"Camera     : {self.camera}"
        )

        print(
            f"Opt. cycle : "
            f"{self.optimization_interval:.1f} seconds"
        )

        print(
            f"Output     : {self.output}"
        )

        print("============================================")
        print()

        # ----------------------------------------------------
        # Create runtime
        # ----------------------------------------------------

        runtime = BytePi4Runtime(
            model_path=self.model_path,
            model_confidence=self.confidence,
            model_image_size=self.image_size,
        )

        model_info = runtime.get_model_info()

        print(
            f"Model type : "
            f"{model_info.get('model_type')}"
        )

        print(
            f"Model size : "
            f"{model_info.get('model_size_mb', 0):.2f} MB"
        )

        # ----------------------------------------------------
        # Open camera
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            self.camera
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Unable to open camera "
                f"{self.camera}"
            )

        # ----------------------------------------------------
        # Start runtime
        # ----------------------------------------------------

        runtime.start()

        try:

            self._warmup(
                cap,
                runtime
            )

            summary = self._run_loop(
                cap,
                runtime,
                model_info
            )

        finally:

            cap.release()

            if self.show_window:
                cv2.destroyAllWindows()

            runtime.stop()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        self._save_results()

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        self._print_summary(
            summary
        )

        return summary

    # ========================================================
    # WARMUP
    # ========================================================

    def _warmup(
        self,
        cap,
        runtime,
    ):

        print(
            "Warming up YOLO model..."
        )

        for _ in range(3):

            ok, frame = cap.read()

            if not ok:
                continue

            runtime.run_inference(
                frame
            )

        print(
            "Warmup complete."
        )

        print()
        print(
            "Benchmark started."
        )
        print(
            "Press Q to stop early."
        )
        print()

    # ========================================================
    # BENCHMARK LOOP
    # ========================================================

    def _run_loop(
        self,
        cap,
        runtime,
        model_info,
    ):

        benchmark_start = (
            time.perf_counter()
        )

        last_optimization = (
            benchmark_start
            - self.optimization_interval
        )

        status = runtime.get_status()

        # ----------------------------------------------------
        # FPS window
        # ----------------------------------------------------

        fps_window_start = (
            benchmark_start
        )

        fps_window_frames = 0

        current_fps = 0.0

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

        while True:

            now = time.perf_counter()

            elapsed = (
                now - benchmark_start
            )

            if elapsed >= self.duration:
                break

            # ------------------------------------------------
            # Capture
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:
                continue

            self.total_frames += 1

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            inference_result = (
                runtime.run_inference(
                    frame
                )
            )

            self.successful_inferences += 1

            inference_ms = float(
                inference_result.get(
                    "inference_time_ms",
                    0.0
                )
            )

            self.inference_times.append(
                inference_ms
            )

            detections = (
                inference_result.get(
                    "detections",
                    []
                )
            )

            detection_count = len(
                detections
            )

            self.total_detections += (
                detection_count
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_window_frames += 1

            fps_elapsed = (
                now - fps_window_start
            )

            if fps_elapsed >= 1.0:

                current_fps = (
                    fps_window_frames
                    / fps_elapsed
                )

                fps_window_start = now
                fps_window_frames = 0

            # ------------------------------------------------
            # BytePi4 optimization
            #
            # Run periodically, NOT every frame.
            # ------------------------------------------------

            if (
                now - last_optimization
                >= self.optimization_interval
            ):

                try:

                    runtime.run_cycle(
                        metrics={
                            "fps": current_fps,
                            "inference_ms": (
                                inference_ms
                            ),
                        },
                        current_time=time.monotonic(),
                    )

                    self.optimization_evaluations += 1

                except Exception as exc:

                    print(
                        "[Benchmark] "
                        f"Optimization warning: {exc}"
                    )

                last_optimization = now

                try:

                    status = (
                        runtime.get_status()
                    )

                except Exception:

                    status = {}

            # ------------------------------------------------
            # Save sample
            # ------------------------------------------------

            self.samples.append(
                {
                    "timestamp": time.time(),

                    "elapsed_seconds": (
                        elapsed
                    ),

                    "model_type": (
                        model_info.get(
                            "model_type"
                        )
                    ),

                    "model_format": (
                        model_info.get(
                            "format"
                        )
                    ),

                    "model_size_mb": (
                        model_info.get(
                            "model_size_mb",
                            0.0
                        )
                    ),

                    "fps": current_fps,

                    "inference_latency_ms": (
                        inference_ms
                    ),

                    "cpu_percent": (
                        self._number(
                            status.get(
                                "cpu"
                            )
                        )
                    ),

                    "memory_mb": (
                        self._number(
                            status.get(
                                "memory"
                            )
                        )
                    ),

                    "temperature_c": (
                        self._number(
                            status.get(
                                "temperature"
                            )
                        )
                    ),

                    "frame_skip": (
                        status.get(
                            "frame_skip",
                            0
                        )
                    ),

                    "resolution": str(
                        status.get(
                            "resolution"
                        )
                    ),

                    "detections": (
                        detection_count
                    ),

                    "system_state": (
                        status.get(
                            "system_state",
                            "UNKNOWN"
                        )
                    ),
                }
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            if self.show_window:

                annotated = (
                    self._annotate_frame(
                        frame,
                        inference_result,
                        current_fps,
                        status,
                        model_info,
                    )
                )

                cv2.imshow(
                    "BytePi4 Benchmark",
                    annotated,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    print(
                        "\nBenchmark stopped "
                        "by user."
                    )

                    break

        # ----------------------------------------------------
        # Final calculation
        # ----------------------------------------------------

        total_elapsed = (
            time.perf_counter()
            - benchmark_start
        )

        actual_fps = (
            self.successful_inferences
            / total_elapsed
            if total_elapsed > 0
            else 0.0
        )

        return {
            "model_type": (
                model_info.get(
                    "model_type"
                )
            ),

            "model_path": (
                model_info.get(
                    "model_path"
                )
            ),

            "model_format": (
                model_info.get(
                    "format"
                )
            ),

            "model_size_mb": (
                model_info.get(
                    "model_size_mb",
                    0.0
                )
            ),

            "duration_seconds": (
                total_elapsed
            ),

            "total_frames": (
                self.total_frames
            ),

            "successful_inferences": (
                self.successful_inferences
            ),

            "total_detections": (
                self.total_detections
            ),

            "optimization_evaluations": (
                self.optimization_evaluations
            ),

            "fps": actual_fps,

            "avg_latency_ms": (
                self._average(
                    self.inference_times
                )
            ),

            "min_latency_ms": (
                self._minimum(
                    self.inference_times
                )
            ),

            "max_latency_ms": (
                self._maximum(
                    self.inference_times
                )
            ),

            "avg_cpu_percent": (
                self._average(
                    self._values(
                        "cpu_percent"
                    )
                )
            ),

            "avg_memory_mb": (
                self._average(
                    self._values(
                        "memory_mb"
                    )
                )
            ),

            "avg_temperature_c": (
                self._average(
                    self._values(
                        "temperature_c"
                    )
                ),
            ),
        }

    # ========================================================
    # ANNOTATE
    # ========================================================

    def _annotate_frame(
        self,
        frame,
        inference_result,
        fps,
        status,
        model_info,
    ):

        results = (
            inference_result.get(
                "results"
            )
        )

        if results:

            try:

                frame = (
                    results[0].plot()
                )

            except Exception:

                frame = frame.copy()

        else:

            frame = frame.copy()

        model_type = (
            model_info.get(
                "model_type",
                "UNKNOWN"
            )
        )

        latency = float(
            inference_result.get(
                "inference_time_ms",
                0.0
            )
        )

        cpu = status.get(
            "cpu"
        )

        memory = status.get(
            "memory"
        )

        lines = [
            f"Model: {model_type}",
            f"FPS: {fps:.2f}",
            f"Inference: {latency:.2f} ms",
            f"CPU: {self._display(cpu)}",
            f"RAM: {self._display(memory)}",
        ]

        y = 30

        for line in lines:

            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            y += 30

        return frame

    # ========================================================
    # SAVE
    # ========================================================

    def _save_results(self):

        output_dir = os.path.dirname(
            self.output
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        fields = [
            "timestamp",
            "elapsed_seconds",
            "model_type",
            "model_format",
            "model_size_mb",
            "fps",
            "inference_latency_ms",
            "cpu_percent",
            "memory_mb",
            "temperature_c",
            "frame_skip",
            "resolution",
            "detections",
            "system_state",
        ]

        with open(
            self.output,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()

            writer.writerows(
                self.samples
            )

        print()
        print(
            f"Results saved to:"
        )
        print(
            f"  {self.output}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def _print_summary(
        summary
    ):

        print()
        print("============================================")
        print("             BENCHMARK COMPLETE")
        print("============================================")

        print(
            f"Model type       : "
            f"{summary['model_type']}"
        )

        print(
            f"Model size       : "
            f"{summary['model_size_mb']:.2f} MB"
        )

        print(
            f"Frames           : "
            f"{summary['total_frames']}"
        )

        print(
            f"Inferences       : "
            f"{summary['successful_inferences']}"
        )

        print(
            f"Detections       : "
            f"{summary['total_detections']}"
        )

        print(
            f"Optimization eval: "
            f"{summary['optimization_evaluations']}"
        )

        print("--------------------------------------------")

        print(
            f"Actual FPS       : "
            f"{summary['fps']:.2f}"
        )

        print(
            f"Avg latency      : "
            f"{summary['avg_latency_ms']:.2f} ms"
        )

        print(
            f"Min latency      : "
            f"{summary['min_latency_ms']:.2f} ms"
        )

        print(
            f"Max latency      : "
            f"{summary['max_latency_ms']:.2f} ms"
        )

        print("--------------------------------------------")

        print(
            f"Average CPU      : "
            f"{Benchmark._display(summary['avg_cpu_percent'])}"
        )

        print(
            f"Average memory   : "
            f"{Benchmark._display(summary['avg_memory_mb'])}"
        )

        print(
            f"Average temp     : "
            f"{Benchmark._display(summary['avg_temperature_c'])}"
        )

        print("============================================")
        print()

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _number(
        value
    ):

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    def _values(
        self,
        key
    ):

        values = []

        for sample in self.samples:

            value = sample.get(
                key
            )

            if value is None:
                continue

            try:

                value = float(
                    value
                )

                if value == value:
                    values.append(
                        value
                    )

            except (
                TypeError,
                ValueError,
            ):

                continue

        return values

    @staticmethod
    def _average(
        values
    ):

        if not values:
            return 0.0

        return statistics.mean(
            values
        )

    @staticmethod
    def _minimum(
        values
    ):

        if not values:
            return 0.0

        return min(values)

    @staticmethod
    def _maximum(
        values
    ):

        if not values:
            return 0.0

        return max(values)

    @staticmethod
    def _display(
        value
    ):

        if value is None:
            return "N/A"

        try:

            return f"{float(value):.2f}"

        except (
            TypeError,
            ValueError,
        ):

            return str(value)


# ============================================================
# RUN BASELINE BENCHMARK
# ============================================================

if __name__ == "__main__":

    benchmark = Benchmark(
        model_path=BASELINE_MODEL,

        duration=30,

        camera=0,

        output=os.path.join(
            PROJECT_ROOT,
            "benchmarks",
            "baseline",
            "results.csv",
        ),

        confidence=0.25,

        image_size=640,

        optimization_interval=1.0,

        show_window=True,
    )

    benchmark.run()

