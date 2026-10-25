"""
BYTEPI4 Runtime Launcher

Starts the BYTEPI4 optimization runtime and continuously
monitors system performance.

Also exposes BYTEPI4 metrics for Prometheus on:

    http://localhost:8000/metrics

This version does NOT require a camera or ML model.

It is designed for developing and testing the optimization
engine on a laptop before deployment to Raspberry Pi 4.
"""

import argparse
import logging
import os
import sys
import time

from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
)

# ============================================================
# PROJECT PATH
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

from src.runtime.runtime import BYTEPI4Runtime

# ============================================================
# LOGGING
# ============================================================

def configure_logging(level="INFO"):

    numeric_level = getattr(
        logging,
        level.upper(),
        logging.INFO
    )

    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: "
            "%(message)s"
        ),
    )


# ============================================================
# PROMETHEUS METRICS
# ============================================================

BYTEPI4_CPU = Gauge(
    "BYTEPI4_cpu_usage_percent",
    "Current CPU usage percentage"
)

BYTEPI4_MEMORY = Gauge(
    "BYTEPI4_memory_usage_percent",
    "Current memory usage percentage"
)

BYTEPI4_TEMPERATURE = Gauge(
    "BYTEPI4_temperature_celsius",
    "Current system temperature in Celsius"
)

BYTEPI4_FPS = Gauge(
    "BYTEPI4_fps",
    "Current frames per second"
)

BYTEPI4_LATENCY = Gauge(
    "BYTEPI4_latency_ms",
    "Current processing latency in milliseconds"
)

BYTEPI4_FRAME_SKIP = Gauge(
    "BYTEPI4_frame_skip",
    "Current adaptive frame skip value"
)

BYTEPI4_RUNTIME_RUNNING = Gauge(
    "BYTEPI4_runtime_running",
    "Whether BYTEPI4 runtime is running"
)

BYTEPI4_RESOLUTION_WIDTH = Gauge(
    "BYTEPI4_resolution_width",
    "Current processing resolution width"
)

BYTEPI4_RESOLUTION_HEIGHT = Gauge(
    "BYTEPI4_resolution_height",
    "Current processing resolution height"
)

BYTEPI4_UPTIME = Gauge(
    "BYTEPI4_uptime_seconds",
    "BYTEPI4 runtime uptime in seconds"
)

BYTEPI4_FRAMES = Counter(
    "BYTEPI4_frames_total",
    "Total number of frames received"
)

BYTEPI4_PROCESSED_FRAMES = Counter(
    "BYTEPI4_processed_frames_total",
    "Total number of processed frames"
)

BYTEPI4_OPTIMIZATION_CYCLES = Counter(
    "BYTEPI4_optimization_cycles_total",
    "Total number of optimization cycles"
)

BYTEPI4_SYSTEM_STATE = Gauge(
    "BYTEPI4_system_state",
    "Current BYTEPI4 optimization state",
    ["state"]
)


# ============================================================
# SYSTEM STATE UPDATE
# ============================================================

def update_system_state(state):

    states = [
        "UNKNOWN",
        "NORMAL",
        "WARNING",
        "CRITICAL",
        "OPTIMIZING",
    ]

    for current_state in states:

        if current_state == state:
            BYTEPI4_SYSTEM_STATE.labels(
                state=current_state
            ).set(1)

        else:
            BYTEPI4_SYSTEM_STATE.labels(
                state=current_state
            ).set(0)


# ============================================================
# RESOLUTION UPDATE
# ============================================================

def update_resolution(resolution):

    if resolution is None:

        BYTEPI4_RESOLUTION_WIDTH.set(0)
        BYTEPI4_RESOLUTION_HEIGHT.set(0)

        return

    try:

        # Expected formats:
        #
        # (640, 480)
        # [640, 480]
        #
        # or similar two-value sequence.

        if isinstance(
            resolution,
            (tuple, list)
        ) and len(resolution) >= 2:

            width = float(
                resolution[0]
            )

            height = float(
                resolution[1]
            )

            BYTEPI4_RESOLUTION_WIDTH.set(
                width
            )

            BYTEPI4_RESOLUTION_HEIGHT.set(
                height
            )

            return

    except (
        TypeError,
        ValueError,
    ):
        pass

    BYTEPI4_RESOLUTION_WIDTH.set(0)
    BYTEPI4_RESOLUTION_HEIGHT.set(0)


# ============================================================
# VALUE CONVERSION
# ============================================================

def safe_float(value):

    if value is None:
        return None

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# PROMETHEUS UPDATE
# ============================================================

def update_prometheus(status):

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu = safe_float(
        status.get("cpu")
    )

    if cpu is not None:
        BYTEPI4_CPU.set(cpu)

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = safe_float(
        status.get("memory")
    )

    if memory is not None:
        BYTEPI4_MEMORY.set(memory)

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temperature = safe_float(
        status.get("temperature")
    )

    if temperature is not None:
        BYTEPI4_TEMPERATURE.set(
            temperature
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = safe_float(
        status.get("fps")
    )

    if fps is not None:
        BYTEPI4_FPS.set(fps)

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    latency = safe_float(
        status.get("latency")
    )

    if latency is None:

        latency = safe_float(
            status.get("inference_ms")
        )

    if latency is not None:
        BYTEPI4_LATENCY.set(
            latency
        )

    # --------------------------------------------------------
    # FRAME SKIP
    # --------------------------------------------------------

    frame_skip = safe_float(
        status.get(
            "frame_skip",
            0
        )
    )

    if frame_skip is not None:

        BYTEPI4_FRAME_SKIP.set(
            frame_skip
        )

    # --------------------------------------------------------
    # RUNTIME
    # --------------------------------------------------------

    running = status.get(
        "running",
        False
    )

    BYTEPI4_RUNTIME_RUNNING.set(
        1 if running else 0
    )

    # --------------------------------------------------------
    # UPTIME
    # --------------------------------------------------------

    uptime = safe_float(
        status.get(
            "uptime_seconds",
            status.get(
                "uptime",
                0
            )
        )
    )

    if uptime is not None:

        BYTEPI4_UPTIME.set(
            uptime
        )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    update_resolution(
        status.get("resolution")
    )

    # --------------------------------------------------------
    # SYSTEM STATE
    # --------------------------------------------------------

    state = status.get(
        "system_state",
        "UNKNOWN"
    )

    update_system_state(
        str(state)
    )


# ============================================================
# DISPLAY
# ============================================================

def print_status(status):

    print(
        "\033[2J\033[H",
        end=""
    )

    print(
        "=============================================="
    )

    print(
        "             BYTEPI4 Runtime"
    )

    print(
        "=============================================="
    )

    print(
        f"Runtime State       : "
        f"{'RUNNING' if status.get('running') else 'STOPPED'}"
    )

    print(
        f"System State        : "
        f"{status.get('system_state', 'N/A')}"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"FPS                 : "
        f"{format_value(status.get('fps'))}"
    )

    print(
        f"CPU Usage           : "
        f"{format_value(status.get('cpu'))} %"
    )

    print(
        f"Memory Usage        : "
        f"{format_value(status.get('memory'))} %"
    )

    print(
        f"Temperature         : "
        f"{format_value(status.get('temperature'))} Â°C"
    )

    print(
        f"Inference Latency   : "
        f"{format_value(status.get('latency'))} ms"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Frame Skip          : "
        f"{status.get('frame_skip', 0)}"
    )

    print(
        f"Resolution          : "
        f"{status.get('resolution', 'N/A')}"
    )

    print(
        f"Total Frames        : "
        f"{status.get('frame_count', 0)}"
    )

    print(
        f"Processed Frames    : "
        f"{status.get('processed_frames', 0)}"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Optimizer Cycles    : "
        f"{status.get('optimization_cycles', 0)}"
    )

    print(
        f"Runtime Uptime      : "
        f"{format_uptime(status.get('uptime_seconds', 0))}"
    )

    print(
        "=============================================="
    )

    print(
        "\nPrometheus metrics : "
        "http://localhost:8000/metrics"
    )

    print(
        "\nPress CTRL+C to stop."
    )


# ============================================================
# VALUE FORMATTERS
# ============================================================

def format_value(value):

    if value is None:
        return "N/A"

    try:

        return f"{float(value):.2f}"

    except (
        TypeError,
        ValueError,
    ):

        return str(value)


def format_uptime(seconds):

    try:

        seconds = int(seconds)

    except (
        TypeError,
        ValueError,
    ):

        return "N/A"

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# RUNTIME LOOP
# ============================================================

def run_runtime(
    update_interval=0.25,
    metrics_port=8000,
):

    logger = logging.getLogger(
        "BYTEPI4Launcher"
    )

    # --------------------------------------------------------
    # Start Prometheus HTTP server
    # --------------------------------------------------------

    start_http_server(
        metrics_port
    )

    logger.info(
        "Prometheus metrics server started "
        "on port %d",
        metrics_port,
    )

    # --------------------------------------------------------
    # Create runtime
    # --------------------------------------------------------

    runtime = BYTEPI4Runtime()

    runtime.start()

    logger.info(
        "BYTEPI4 runtime started."
    )

    try:

        while True:

            # ------------------------------------------------
            # Execute one optimization cycle
            # ------------------------------------------------

            runtime.run_cycle()

            # ------------------------------------------------
            # Get current runtime status
            # ------------------------------------------------

            status = runtime.get_status()

            # ------------------------------------------------
            # Collect profiler metrics
            # ------------------------------------------------

            try:

                profiler_metrics = (
                    runtime.profiler.get_metrics()
                )

                if profiler_metrics:

                    status.update(
                        {
                            "cpu":
                                profiler_metrics.get(
                                    "cpu"
                                ),

                            "memory":
                                profiler_metrics.get(
                                    "memory"
                                ),

                            "temperature":
                                profiler_metrics.get(
                                    "temperature"
                                ),

                            "fps":
                                profiler_metrics.get(
                                    "fps"
                                ),

                            "latency":
                                profiler_metrics.get(
                                    "inference_ms"
                                ),
                        }
                    )

            except Exception as exc:

                logger.debug(
                    "Profiler metrics unavailable: %s",
                    exc,
                )

            # ------------------------------------------------
            # Update Prometheus
            # ------------------------------------------------

            update_prometheus(
                status
            )

            # ------------------------------------------------
            # Display status
            # ------------------------------------------------

            print_status(
                status
            )

            # ------------------------------------------------
            # Wait
            # ------------------------------------------------

            time.sleep(
                max(
                    0.01,
                    update_interval
                )
            )

    except KeyboardInterrupt:

        print(
            "\n\nStopping BYTEPI4..."
        )

    finally:

        runtime.shutdown()

        print(
            "BYTEPI4 stopped."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Launch the BYTEPI4 "
            "optimization runtime "
            "with Prometheus metrics."
        )
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help=(
            "Runtime display/update "
            "interval in seconds."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help=(
            "Prometheus metrics "
            "HTTP port."
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
        help="Logging level.",
    )

    args = parser.parse_args()

    configure_logging(
        args.log_level
    )

    run_runtime(
        update_interval=args.interval,
        metrics_port=args.port,
    )


if __name__ == "__main__":

    main()