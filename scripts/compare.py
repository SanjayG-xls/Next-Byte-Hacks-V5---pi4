
"""
BytePi4 Benchmark Comparison Engine

Compares:

    BASELINE
        benchmarks/baseline/results.csv

    OPTIMIZED
        benchmarks/optimized/results.csv

Calculates:

    - Model size reduction
    - FPS improvement
    - Latency reduction
    - CPU change
    - Memory change
    - Temperature change

Also saves a summary report to:

    reports/results/optimization_report.csv
"""

import csv
import os
import statistics


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# INPUT FILES
# ============================================================

BASELINE_FILE = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "baseline",
    "results.csv",
)

OPTIMIZED_FILE = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "optimized",
    "results.csv",
)


# ============================================================
# OUTPUT
# ============================================================

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "results",
)

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "optimization_report.csv",
)


# ============================================================
# CSV READER
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Benchmark file not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise RuntimeError(
            f"Benchmark file is empty:\n{path}"
        )

    return rows


# ============================================================
# NUMBER CONVERSION
# ============================================================

def number(value):

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
# AVERAGE
# ============================================================

def average(rows, field):

    values = []

    for row in rows:

        value = number(
            row.get(field)
        )

        if value is not None:
            values.append(value)

    if not values:
        return 0.0

    return statistics.mean(
        values
    )


# ============================================================
# MODEL SIZE
# ============================================================

def model_size(rows):

    values = []

    for row in rows:

        value = number(
            row.get(
                "model_size_mb"
            )
        )

        if value is not None:
            values.append(value)

    if not values:
        return 0.0

    return values[0]


# ============================================================
# MODEL TYPE
# ============================================================

def model_type(rows):

    for row in rows:

        value = row.get(
            "model_type"
        )

        if value:
            return value

    return "unknown"


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

def percentage_change(
    baseline,
    optimized,
):

    if baseline == 0:
        return 0.0

    return (
        (optimized - baseline)
        / baseline
    ) * 100.0


# ============================================================
# REDUCTION
# ============================================================

def percentage_reduction(
    baseline,
    optimized,
):

    if baseline == 0:
        return 0.0

    return (
        (baseline - optimized)
        / baseline
    ) * 100.0


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare():

    print()
    print("============================================")
    print("       BytePi4 Optimization Comparison")
    print("============================================")
    print()

    # --------------------------------------------------------
    # Read benchmark results
    # --------------------------------------------------------

    baseline_rows = read_csv(
        BASELINE_FILE
    )

    optimized_rows = read_csv(
        OPTIMIZED_FILE
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    baseline_model = model_type(
        baseline_rows
    )

    optimized_model = model_type(
        optimized_rows
    )

    baseline_size = model_size(
        baseline_rows
    )

    optimized_size = model_size(
        optimized_rows
    )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    baseline_fps = average(
        baseline_rows,
        "fps"
    )

    optimized_fps = average(
        optimized_rows,
        "fps"
    )

    baseline_latency = average(
        baseline_rows,
        "inference_latency_ms"
    )

    optimized_latency = average(
        optimized_rows,
        "inference_latency_ms"
    )

    baseline_cpu = average(
        baseline_rows,
        "cpu_percent"
    )

    optimized_cpu = average(
        optimized_rows,
        "cpu_percent"
    )

    baseline_memory = average(
        baseline_rows,
        "memory_mb"
    )

    optimized_memory = average(
        optimized_rows,
        "memory_mb"
    )

    baseline_temperature = average(
        baseline_rows,
        "temperature_c"
    )

    optimized_temperature = average(
        optimized_rows,
        "temperature_c"
    )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    size_reduction = percentage_reduction(
        baseline_size,
        optimized_size,
    )

    fps_improvement = percentage_change(
        baseline_fps,
        optimized_fps,
    )

    latency_reduction = percentage_reduction(
        baseline_latency,
        optimized_latency,
    )

    cpu_change = percentage_change(
        baseline_cpu,
        optimized_cpu,
    )

    memory_change = percentage_change(
        baseline_memory,
        optimized_memory,
    )

    temperature_change = percentage_change(
        baseline_temperature,
        optimized_temperature,
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print(
        f"Baseline model : "
        f"{baseline_model}"
    )

    print(
        f"Optimized model: "
        f"{optimized_model}"
    )

    print()
    print("--------------------------------------------")
    print("MODEL SIZE")
    print("--------------------------------------------")

    print(
        f"Baseline       : "
        f"{baseline_size:.2f} MB"
    )

    print(
        f"Optimized      : "
        f"{optimized_size:.2f} MB"
    )

    print(
        f"Reduction      : "
        f"{size_reduction:.2f}%"
    )

    print()
    print("--------------------------------------------")
    print("PERFORMANCE")
    print("--------------------------------------------")

    print(
        f"Baseline FPS   : "
        f"{baseline_fps:.2f}"
    )

    print(
        f"Optimized FPS  : "
        f"{optimized_fps:.2f}"
    )

    print(
        f"FPS improvement: "
        f"{fps_improvement:+.2f}%"
    )

    print()

    print(
        f"Baseline latency : "
        f"{baseline_latency:.2f} ms"
    )

    print(
        f"Optimized latency: "
        f"{optimized_latency:.2f} ms"
    )

    print(
        f"Latency reduction: "
        f"{latency_reduction:.2f}%"
    )

    print()
    print("--------------------------------------------")
    print("SYSTEM")
    print("--------------------------------------------")

    print(
        f"Baseline CPU   : "
        f"{baseline_cpu:.2f}%"
    )

    print(
        f"Optimized CPU  : "
        f"{optimized_cpu:.2f}%"
    )

    print(
        f"CPU change     : "
        f"{cpu_change:+.2f}%"
    )

    print()

    print(
        f"Baseline memory : "
        f"{baseline_memory:.2f}"
    )

    print(
        f"Optimized memory: "
        f"{optimized_memory:.2f}"
    )

    print(
        f"Memory change   : "
        f"{memory_change:+.2f}%"
    )

    print()

    print(
        f"Baseline temp : "
        f"{baseline_temperature:.2f} Â°C"
    )

    print(
        f"Optimized temp: "
        f"{optimized_temperature:.2f} Â°C"
    )

    print(
        f"Temperature change: "
        f"{temperature_change:+.2f}%"
    )

    print()
    print("============================================")
    print("             OPTIMIZATION RESULT")
    print("============================================")

    print(
        f"Model size reduction : "
        f"{size_reduction:.2f}%"
    )

    print(
        f"FPS improvement      : "
        f"{fps_improvement:+.2f}%"
    )

    print(
        f"Latency reduction    : "
        f"{latency_reduction:.2f}%"
    )

    print(
        f"CPU change           : "
        f"{cpu_change:+.2f}%"
    )

    print(
        f"Memory change        : "
        f"{memory_change:+.2f}%"
    )

    print(
        f"Temperature change   : "
        f"{temperature_change:+.2f}%"
    )

    print("============================================")
    print()

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    report = [
        {
            "metric": "model_size_mb",
            "baseline": baseline_size,
            "optimized": optimized_size,
            "change_percent": (
                percentage_change(
                    baseline_size,
                    optimized_size
                )
            ),
            "improvement_percent": (
                size_reduction
            ),
        },
        {
            "metric": "fps",
            "baseline": baseline_fps,
            "optimized": optimized_fps,
            "change_percent": (
                fps_improvement
            ),
            "improvement_percent": (
                fps_improvement
            ),
        },
        {
            "metric": "latency_ms",
            "baseline": baseline_latency,
            "optimized": optimized_latency,
            "change_percent": (
                percentage_change(
                    baseline_latency,
                    optimized_latency
                )
            ),
            "improvement_percent": (
                latency_reduction
            ),
        },
        {
            "metric": "cpu_percent",
            "baseline": baseline_cpu,
            "optimized": optimized_cpu,
            "change_percent": cpu_change,
            "improvement_percent": (
                -cpu_change
            ),
        },
        {
            "metric": "memory",
            "baseline": baseline_memory,
            "optimized": optimized_memory,
            "change_percent": memory_change,
            "improvement_percent": (
                -memory_change
            ),
        },
        {
            "metric": "temperature_c",
            "baseline": baseline_temperature,
            "optimized": optimized_temperature,
            "change_percent": (
                temperature_change
            ),
            "improvement_percent": (
                -temperature_change
            ),
        },
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
    ) as file:

        fields = [
            "metric",
            "baseline",
            "optimized",
            "change_percent",
            "improvement_percent",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            report
        )

    print(
        "Report saved to:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    compare()

