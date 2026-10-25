"""
BytePi4 Prometheus Monitoring Runner

Starts:
    BytePi4 Runtime
    BytePi4 Profiler
    Prometheus Exporter

and continuously publishes runtime metrics.
"""

import os
import sys
import time

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# BYTE PILOT IMPORTS
# ==========================================================

from src.runtime.runtime import BytePi4Runtime
from src.monitoring.prometheus_exporter import (
    PrometheusExporter,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

PROMETHEUS_PORT = 8000

UPDATE_INTERVAL = 0.5


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("========================================")
    print("      BytePi4 Prometheus Monitor")
    print("========================================")
    print(
        f"Prometheus port : {PROMETHEUS_PORT}"
    )
    print(
        f"Update interval : {UPDATE_INTERVAL}s"
    )
    print("========================================")
    print()

    # ------------------------------------------------------
    # Create BytePi4 runtime
    # ------------------------------------------------------

    runtime = BytePi4Runtime()

    # ------------------------------------------------------
    # Start runtime
    # ------------------------------------------------------

    runtime.start()

    # ------------------------------------------------------
    # Create Prometheus exporter
    # ------------------------------------------------------

    exporter = PrometheusExporter(
        port=PROMETHEUS_PORT
    )

    exporter.start()

    print()
    print("BytePi4 monitoring is running.")
    print()
    print(
        "Open Prometheus metrics at:"
    )
    print(
        f"http://localhost:{PROMETHEUS_PORT}/metrics"
    )
    print()
    print("Press CTRL+C to stop.")
    print()

    # ------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------

    try:

        while runtime.is_running():

            # ----------------------------------------------
            # Run one optimization cycle
            # ----------------------------------------------

            try:

                runtime.run_cycle()

            except Exception as exc:

                print(
                    f"Runtime cycle error: {exc}"
                )

            # ----------------------------------------------
            # Update Prometheus metrics
            # ----------------------------------------------

            try:

                exporter.update(
                    profiler=runtime.profiler,
                    runtime=runtime,
                )

            except Exception as exc:

                print(
                    f"Prometheus update error: {exc}"
                )

            # ----------------------------------------------
            # Wait before next update
            # ----------------------------------------------

            time.sleep(
                UPDATE_INTERVAL
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping BytePi4 monitoring..."
        )

    finally:

        runtime.shutdown()

        print(
            "BytePi4 monitoring stopped."
        )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()