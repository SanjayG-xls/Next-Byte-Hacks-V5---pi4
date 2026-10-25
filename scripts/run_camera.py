import time
import cv2

from src.runtime.runtime import BytePi4Runtime
from src.monitoring.prometheus_exporter import PrometheusExporter


MODEL_PATH = "models/optimized/best_int8.tflite"
CAMERA_INDEX = 0
EXPORTER_PORT = 8000


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    runtime = BytePi4Runtime(
        model_path=MODEL_PATH,
        model_confidence=0.25,
        model_image_size=640,
        device="cpu",
    )

    exporter = PrometheusExporter(
        port=EXPORTER_PORT
    )

    exporter.start()
    runtime.start()

    ret, frame = cap.read()

    if not ret:
        cap.release()
        runtime.stop()
        raise RuntimeError("Unable to read camera frame")

    runtime.model.warmup(frame)

    previous_time = time.monotonic()

    try:
        while runtime.is_running():
            ret, frame = cap.read()

            if not ret:
                continue

            now = time.monotonic()

            runtime.frame_count += 1

            resolution = runtime.controller.get_resolution()

            frame_skip = runtime.controller.get_frame_skip()

            should_infer = runtime.controller.adaptive_skip.should_run(
                runtime.frame_count
            )

            processed_frame = cv2.resize(
                frame,
                resolution
            )

            if should_infer:
                result = runtime.run_inference(
                    processed_frame
                )

                runtime.processed_frames += 1

                detections = result.get(
                    "detections",
                    []
                )

                scale_x = frame.shape[1] / resolution[0]
                scale_y = frame.shape[0] / resolution[1]

                for detection in detections:
                    x1, y1, x2, y2 = map(
                        int,
                        detection["bbox"]
                    )

                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)

                    label = (
                        f'{detection["class_name"]} '
                        f'{detection["confidence"]:.2f}'
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            elapsed = now - previous_time

            fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            previous_time = now

            runtime.profiler.update_fps()

            metrics = {
                "cpu": runtime.profiler.get_cpu(),
                "memory": runtime.profiler.get_memory(),
                "temperature": runtime.profiler.get_temperature(),
                "fps": fps,
                "inference_ms": runtime.last_inference_time_ms,
            }

            decision = runtime.run_cycle(
                metrics=metrics,
                current_time=now
            )

            current_state = runtime.system_state

            cv2.putText(
                frame,
                f"State: {current_state}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Skip: {runtime.controller.get_frame_skip()}",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            current_resolution = runtime.controller.get_resolution()

            cv2.putText(
                frame,
                f"Resolution: {current_resolution[0]}x{current_resolution[1]}",
                (10, 109),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Latency: {runtime.last_inference_time_ms:.2f} ms",
                (10, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            exporter.update(
                runtime.profiler,
                runtime
            )

            cv2.imshow(
                "BytePi4",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        runtime.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()