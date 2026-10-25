"""
BytePi4 Benchmark Runner

Entry point for running the BytePi4 benchmark engine.

This script keeps command-line execution separate from
the benchmark implementation in benchmark.py.
"""

import argparse
import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# BENCHMARK IMPORT
# ============================================================

from scripts.benchmark import Benchmark


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the BytePi4 benchmark engine."
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help=(
            "Benchmark duration in seconds "
            "(default: 30)"
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help=(
            "Metric sampling interval in seconds "
            "(default: 0.05)"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "benchmarks/baseline/"
            "results.csv"
        ),
        help=(
            "Path where benchmark CSV "
            "will be saved."
        ),
    )

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create benchmark
    # --------------------------------------------------------

    benchmark = Benchmark(
        duration=args.duration,
        interval=args.interval,
        output=args.output,
    )

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    try:

        benchmark.run()

    except KeyboardInterrupt:

        print(
            "\nBenchmark stopped by user."
        )

        return 1

    except Exception as error:

        print(
            "\n========================================"
        )

        print(
            "        BENCHMARK ERROR"
        )

        print(
            "========================================"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        print(
            "========================================\n"
        )

        return 1

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )