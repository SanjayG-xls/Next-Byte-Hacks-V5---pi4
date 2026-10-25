import os
import sys
import cv2
import time

# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# BYTE PILOT IMPORT
# ============================================================

from src.models.yolo_model import YOLOModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "optimized",
    "best_int8.tflite",
)

CAMERA = 0
TEST_DURATION = 10


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLOModel(
    MODEL,
    task="detect",
    confidence=0.25,
    image_size=640,
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA}"
    )


# ============================================================
# WARMUP
# ============================================================

print()
print("Warming up YOLO...")

for _ in range(3):

    ok, frame = cap.read()

    if not ok:
        continue

    model.predict(
        frame,
        verbose=False,
    )

print("Warmup complete.")


# ============================================================
# FPS TEST
# ============================================================

print()
print(
    f"Running pure YOLO test "
    f"for {TEST_DURATION} seconds..."
)
print()

count = 0
total_inference = 0.0

start = time.perf_counter()

while (
    time.perf_counter() - start
    < TEST_DURATION
):

    ok, frame = cap.read()

    if not ok:
        continue

    result = model.predict(
        frame,
        verbose=False,
    )

    total_inference += result[
        "inference_time"
    ]

    count += 1


# ============================================================
# RESULTS
# ============================================================

elapsed = (
    time.perf_counter() - start
)

cap.release()

fps = (
    count / elapsed
    if elapsed > 0
    else 0
)

avg_latency = (
    total_inference / count * 1000
    if count > 0
    else 0
)


print()
print("==============================")
print("       Pure YOLO FPS Test")
print("==============================")
print(
    f"Model        : {MODEL}"
)
print(
    f"Frames       : {count}"
)
print(
    f"Time         : {elapsed:.2f} s"
)
print(
    f"FPS          : {fps:.2f}"
)
print(
    f"Avg latency  : {avg_latency:.2f} ms"
)
print("==============================")
print()