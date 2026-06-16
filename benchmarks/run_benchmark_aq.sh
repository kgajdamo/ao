#!/bin/bash
set -exo pipefail

# Run all affine-quantization benchmark scenarios in benchmark_aq.py.
#
# benchmark_aq.py takes NO command-line arguments: a single invocation runs
# every scenario it defines. Each quantized config is torch.compile'd with
# mode="max-autotune" and timed against a bf16 baseline over the shapes in
# `all_shapes` (default (M, N, K) = (20, 2048, 2048)), on the auto-detected
# device (last of get_available_devices(): xpu if available, else cuda).
#
# Scenarios covered by one run:
#   1. Int8DynamicActivationInt8WeightConfig   (int8 dynamic activation + int8 weight)
#   2. Int8WeightOnlyConfig                    (int8 weight-only)
#   3. Int4WeightOnlyConfig(group_size=32)     (int4 weight-only)
#
# Run from the repo root with the ao-kgajdamo venv already activated:
#   source /home/kgajdamo/ao-kgajdamo/bin/activate
#   ./benchmarks/run_benchmark_aq.sh

cd /home/kgajdamo/workspace/repos/ao-kgajdamo

BENCH="python benchmarks/benchmark_aq.py"

# Timestamped log next to the benchmark
LOG="benchmarks/benchmark_aq_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# Run all scenarios (int8 dynamic, int8 weight-only, int4 weight-only)
# ============================================================================
$BENCH 2>&1 | tee "$LOG"

echo "Saved benchmark log to: $LOG"
