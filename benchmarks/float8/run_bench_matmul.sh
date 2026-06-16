#!/bin/bash
set -ex

cd /home/kgajdamo/workspace/repos/ao-kgajdamo

BENCH="python benchmarks/float8/bench_matmul.py"

# ============================================================================
# 1. Default (pow2_extended, tensorwise, gpu kernel time)
# ============================================================================
$BENCH

# ============================================================================
# 2. Recipes (XPU-compatible: tensorwise, rowwise)
# ============================================================================
$BENCH --recipe tensorwise
$BENCH --recipe rowwise

# mxfp4_cutlass and nvfp4 are NVIDIA-only, skip on XPU
# $BENCH --recipe mxfp4_cutlass
# $BENCH --recipe nvfp4

# ============================================================================
# 3. Shape generators
# ============================================================================
$BENCH --shape_gen_name llama
$BENCH --shape_gen_name pow2
$BENCH --shape_gen_name pow2_extended
$BENCH --shape_gen_name custom --M 4096 --K 4096 --N 4096
$BENCH --shape_gen_name dsv3-671b

# sweep is O(n^3) — limit iterations
$BENCH --shape_gen_name sweep --n_limit 5

# ============================================================================
# 4. Recipes × shape generators
# ============================================================================
$BENCH --recipe rowwise --shape_gen_name llama
$BENCH --recipe rowwise --shape_gen_name pow2
$BENCH --recipe rowwise --shape_gen_name custom --M 2048 --K 8192 --N 7168
$BENCH --recipe rowwise --shape_gen_name dsv3-671b

# ============================================================================
# 5. GPU kernel time vs wall time
# ============================================================================
$BENCH --use_gpu_kernel_time True
$BENCH --use_gpu_kernel_time False

# ============================================================================
# 6. Limit iterations
# ============================================================================
$BENCH --n_limit 1
$BENCH --n_limit 3

# ============================================================================
# 7. Save results to CSV
# ============================================================================
$BENCH --out_filename results_bench_matmul_tensorwise.csv
$BENCH --recipe rowwise --out_filename results_bench_matmul_rowwise.csv

# ============================================================================
# 8. Combined: recipe + shapes + limit + CSV
# ============================================================================
$BENCH --recipe tensorwise --shape_gen_name llama --n_limit 2 \
    --out_filename results_matmul_tensorwise_llama.csv

$BENCH --recipe rowwise --shape_gen_name custom --M 8192 --K 8192 --N 8192 \
    --use_gpu_kernel_time False \
    --out_filename results_matmul_rowwise_custom.csv

# ============================================================================
# 9. DSV3 with limited shapes
# ============================================================================
$BENCH --shape_gen_name dsv3-671b --n_limit 3
$BENCH --recipe rowwise --shape_gen_name dsv3-671b --n_limit 3

# ============================================================================
# 10. Custom M override for dsv3-671b
# ============================================================================
$BENCH --shape_gen_name dsv3-671b --M 4096
$BENCH --recipe rowwise --shape_gen_name dsv3-671b --M 4096