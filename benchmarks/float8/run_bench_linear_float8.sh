#!/bin/bash
set -ex

cd /home/kgajdamo/workspace/repos/ao-kgajdamo

BENCH="python benchmarks/float8/bench_linear_float8.py"

# ============================================================================
# 1. Default configuration (llama shapes, dynamic tensorwise, compiled)
# ============================================================================
$BENCH

# ============================================================================
# 2. Shape generators
#    Options: llama, pow2, pow2_extended, sweep, custom, dsv3-671b
# ============================================================================
$BENCH --shape_gen_name llama
$BENCH --shape_gen_name pow2
$BENCH --shape_gen_name pow2_extended
$BENCH --shape_gen_name custom --M 4096 --K 4096 --N 4096
$BENCH --shape_gen_name dsv3-671b

# sweep generates O(n^3) combinations - limit iterations to keep it fast
$BENCH --shape_gen_name sweep -n 5

# ============================================================================
# 3. Scaling granularity
#    Options: tensorwise, axiswise
# ============================================================================
$BENCH --scaling_granularity tensorwise
$BENCH --scaling_granularity axiswise

# ============================================================================
# 4. Scaling types (input / weight / grad_output)
#    Options: dynamic, disabled
# ============================================================================

# All dynamic (default)
$BENCH --scaling_type_input dynamic --scaling_type_weight dynamic --scaling_type_grad_output dynamic

# Disable grad_output scaling
# Expected error:
# torch.mm "disabled" scaling option is supported only when all scales are disabled: 
# --scaling_type_input dynamic --scaling_type_weight dynamic --scaling_type_grad_output disabled
# AssertionError: Expecting both Float8TrainingTensor for mm inputs but found <class 'torch._subclasses.functional_tensor.FunctionalTensor'> and <class 'torchao.float8.float8_training_tensor.Float8TrainingTensor'>

# $BENCH --scaling_type_input dynamic --scaling_type_weight dynamic --scaling_type_grad_output disabled
$BENCH --scaling_type_input disabled --scaling_type_weight disabled --scaling_type_grad_output disabled

# Axiswise (rowwise) with dynamic scaling
$BENCH --scaling_type_input dynamic --scaling_type_weight dynamic --scaling_granularity axiswise

# ============================================================================
# 5. Compile vs no-compile
# ============================================================================
$BENCH --disable_compile
$BENCH -n 3  # compiled (default), limited iterations

# ============================================================================
# 6. Fast accumulation filter
# ============================================================================
$BENCH --fast_accum_filter True
$BENCH --fast_accum_filter False

# ============================================================================
# 7. Shape name filter (filter to a specific shape from the shape generator)
# ============================================================================
$BENCH --shape_gen_name llama --shape_name_filter "attn.wqkv"
$BENCH --shape_gen_name llama --shape_name_filter "ffn.w13"

# ============================================================================
# 8. Limit iterations
# ============================================================================
$BENCH -n 1
$BENCH -n 3

# ============================================================================
# 9. Save results to CSV
# ============================================================================
$BENCH -o results_bench_linear_float8.csv

# ============================================================================
# 10. Combined: custom shape, axiswise, no compile, save CSV
# ============================================================================
$BENCH --shape_gen_name custom --M 2048 --K 8192 --N 7168 \
    --scaling_granularity axiswise \
    --disable_compile \
    -o results_custom_axiswise.csv