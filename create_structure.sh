#!/bin/bash

set -e

echo "Creating BytePi4 optimization structure..."

# Directories
mkdir -p \
src/runtime \
src/optimizer \
src/profiler \
src/monitoring \
configs \
benchmarks/baseline \
benchmarks/optimized \
dashboard/prometheus \
dashboard/grafana \
reports/graphs \
reports/results \
scripts \
tests/optimizer \
tests/profiler \
tests/runtime

# Runtime
touch \
src/runtime/__init__.py \
src/runtime/runtime.py \
src/runtime/scheduler.py

# Optimizer
touch \
src/optimizer/__init__.py \
src/optimizer/controller.py \
src/optimizer/adaptive_skip.py \
src/optimizer/dynamic_resolution.py \
src/optimizer/resource_manager.py

# Profiler
touch \
src/profiler/__init__.py \
src/profiler/profiler.py

# Monitoring
touch \
src/monitoring/__init__.py \
src/monitoring/metrics.py

# Configuration
touch \
configs/runtime.yaml \
configs/optimization.yaml

# Dashboard
touch \
dashboard/prometheus/prometheus.yml

# Scripts
touch \
scripts/run_runtime.py \
scripts/benchmark.py \
scripts/compare.py

# Project files
touch \
requirements.txt \
README.md \
LICENSE \
.gitignore

echo ""
echo "BytePi4 structure created successfully!"
echo ""

tree -L 4
