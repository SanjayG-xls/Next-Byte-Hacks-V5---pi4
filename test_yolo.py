import cv2
import time
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/optimized/best_int8.tflite"

CAMERA_INDEX = 0

CONFIDENCE = 0.25

IMAGE_SIZE = 640


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("BytePi4 - Raspberry Pi 4 YOLO Test")
print("=" * 60)

print(f"Model : {MODEL_PATH}")
print(f"Camera: /dev/video{CAMERA_INDEX}")

print("\nLoading YOLO model...")

model = YOLO(
    MODEL_PATH,
    task="detect",
)

print("YOLO model loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera /dev/video{CAMERA_INDEX}"
    )


# ------------------------------------------------------------
# Camera resolution
# ------------------------------------------------------------

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reduce camera buffering on Raspberry Pi
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print(
    f"Camera resolution: "
    f"{actual_width}x{actual_height}"
)

print("\nStarting inference...")
print("Press Q to quit.")
print("=" * 60)


# ============================================================
# FPS VARIABLES
# ============================================================

frame_count = 0

fps = 0.0

fps_start = time.perf_counter()

last_inference_ms = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read frame from camera."
            )

            break

        frame_count += 1

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        inference_end = time.perf_counter()

        last_inference_ms = (
            inference_end
            - inference_start
        ) * 1000.0

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        annotated_frame = results[0].plot()

        # ----------------------------------------------------
        # FPS calculation
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - fps_start
        )

        if elapsed >= 1.0:

            fps = frame_count / elapsed

            frame_count = 0

            fps_start = time.perf_counter()

        # ----------------------------------------------------
        # Display performance information
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Latency: {last_inference_ms:.2f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "BytePi4 - Raspberry Pi 4",
            annotated_frame,
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print("\nStopping...")


finally:

    cap.release()

    cv2.destroyAllWindows()

    print("\nCamera released.")

    print("BytePi4 YOLO test finished.")