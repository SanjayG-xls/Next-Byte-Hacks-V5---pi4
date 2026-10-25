
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from ultralytics import YOLO

class YOLOModel:
    

    def __init__(
        self,
        model_path: str,
        task: str = "detect",
        confidence: float = 0.25,
        image_size: int = 640,
        device: Optional[str] = None,
    ):
        

        self.model_path = os.path.abspath(model_path)
        self.task = task
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        # Identify model type.
        extension = os.path.splitext(
            self.model_path
        )[1].lower()

        if extension not in {".pt", ".tflite"}:
            raise ValueError(
                f"Unsupported YOLO model format: {extension}. "
                "Expected .pt or .tflite."
            )

        self.model_type = (
            "baseline"
            if extension == ".pt"
            else "optimized"
        )

        self.extension = extension

        print(
            f"[YOLOModel] Loading {self.model_type} model:"
        )
        print(
            f"[YOLOModel] {self.model_path}"
        )

        # Ultralytics handles both .pt and .tflite.
        self.model = YOLO(
            self.model_path,
            task=self.task,
        )

        print(
            f"[YOLOModel] Loaded successfully "
            f"({self.model_type})"
        )

    def predict(
        self,
        frame: Any,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        

        start_time = time.perf_counter()

        predict_kwargs = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": verbose,
        }

        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(
            **predict_kwargs
        )

        inference_time = (
            time.perf_counter() - start_time
        )

        detections = self._extract_detections(
            results
        )

        return {
            "detections": detections,
            "results": results,
            "inference_time": inference_time,
            "inference_time_ms": inference_time * 1000.0,
            "model_type": self.model_type,
            "model_path": self.model_path,
        }

    # ========================================================
    # DETECTION EXTRACTION
    # ========================================================

    def _extract_detections(
        self,
        results: Any,
    ) -> List[Dict[str, Any]]:

        detections: List[Dict[str, Any]] = []

        if results is None:
            return detections

        for result in results:

            boxes = getattr(
                result,
                "boxes",
                None,
            )

            if boxes is None:
                continue

            for box in boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                confidence = float(
                    box.conf[0].cpu().item()
                )

                class_id = int(
                    box.cls[0].cpu().item()
                )

                class_name = str(
                    result.names.get(
                        class_id,
                        class_id,
                    )
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": xyxy,
                    }
                )

        return detections


    def get_model_info(self) -> Dict[str, Any]:
        

        try:
            model_size_bytes = os.path.getsize(
                self.model_path
            )
        except OSError:
            model_size_bytes = 0

        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "format": self.extension,
            "model_size_bytes": model_size_bytes,
            "model_size_mb": (
                model_size_bytes / (1024 * 1024)
            ),
            "task": self.task,
            "confidence": self.confidence,
            "image_size": self.image_size,
        }

    def warmup(
        self,
        frame: Any,
    ) -> None:
        """
        Perform one inference to warm up the model.

        This should be done before benchmarking so that
        model loading/warmup time does not distort the
        benchmark measurements.
        """

        print(
            f"[YOLOModel] Warming up "
            f"{self.model_type} model..."
        )

        self.predict(
            frame,
            verbose=False,
        )

        print(
            "[YOLOModel] Warmup complete."
        )


    def is_optimized(self) -> bool:

        return self.model_type == "optimized"

    def is_baseline(self) -> bool:

        return self.model_type == "baseline"

    def __repr__(self) -> str:
        return (
            f"YOLOModel("
            f"type='{self.model_type}', "
            f"format='{self.extension}', "
            f"path='{self.model_path}'"
            f")"
        )