#!/bin/bash
set -ex

cd /home/kgajdamo/workspace/repos/ao-kgajdamo/benchmarks/float8

# ============================================================
# Basic recipes — pow2 shapes (default), all iterations
# ============================================================

# Tensorwise (default shape_gen_name=pow2)
python float8_inference_roofline.py --recipe_name tensorwise

# Rowwise
python float8_inference_roofline.py --recipe_name rowwise

# ============================================================
# Block / fp4 recipes — roofline only
# These recipes need sm100+/cublas hardware to run real GPU kernels,
# so exercise their roofline math path with --do_benchmarks False.
# ============================================================

# python float8_inference_roofline.py --recipe_name mxfp8_cublas --do_benchmarks False
# python float8_inference_roofline.py --recipe_name mxfp4_cutlass --do_benchmarks False
# python float8_inference_roofline.py --recipe_name nvfp4 --do_benchmarks False
# python float8_inference_roofline.py --recipe_name nvfp4_static --do_benchmarks False
# python float8_inference_roofline.py --recipe_name nvfp4_no_global_scale --do_benchmarks False

# ============================================================
# Shape generators (each exercises a different shape set)
# ============================================================

# Custom: single user-specified shape
python float8_inference_roofline.py --recipe_name tensorwise --shape_gen_name custom --M 4096 --K 4096 --N 4096

# LLaMA shapes
python float8_inference_roofline.py --recipe_name rowwise --shape_gen_name llama

# pow2
python float8_inference_roofline.py --recipe_name tensorwise --shape_gen_name pow2

# pow2_extended
python float8_inference_roofline.py --recipe_name tensorwise --shape_gen_name pow2_extended

# sweep (roofline only — large shape set)
python float8_inference_roofline.py --recipe_name tensorwise --shape_gen_name sweep --do_benchmarks False

# ============================================================
# Limit iterations (quick smoke test)
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --n_limit 1
python float8_inference_roofline.py --recipe_name rowwise --n_limit 3

# ============================================================
# Roofline only (skip actual GPU benchmarks)
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --do_benchmarks False
python float8_inference_roofline.py --recipe_name rowwise --do_benchmarks False --shape_gen_name llama

# ============================================================
# Save results to CSV
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --outfile results_tensorwise.csv
python float8_inference_roofline.py --recipe_name rowwise --shape_gen_name llama --outfile results_rowwise_llama.csv

# ============================================================
# Fusion modeling (models activation read/write overhead)
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --enable_fusion_modeling True
python float8_inference_roofline.py --recipe_name rowwise --enable_fusion_modeling True --shape_gen_name llama

# ============================================================
# Condensed output (show only speedups)
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --skip_printing_detailed_metrics True
python float8_inference_roofline.py --recipe_name rowwise --skip_printing_detailed_metrics True --shape_gen_name llama

# ============================================================
# Save profiling traces (for chrome://tracing)
# ============================================================

python float8_inference_roofline.py --recipe_name tensorwise --save_profile_traces True --outfile trace_run --n_limit 2

# ============================================================
# Conv2d benchmarks (roofline only — fp8 conv2d not yet implemented)
# ============================================================
# Note: roofline (do_benchmarks=False) calculates expected time using hardware specs (peak TOPS, memory bandwidth) and math (FLOPs, bytes moved). The roofline path estimates performance without running any GPU kernels — it's pure math.

python float8_inference_roofline.py --recipe_name tensorwise --op_name conv2d --H 224 --W 224 --kernel_size 3 --shape_gen_name custom --M 1 --K 64 --N 128 --do_benchmarks False
python float8_inference_roofline.py --recipe_name tensorwise --op_name conv2d --H 56 --W 56 --kernel_size 3 --padding 1 --shape_gen_name custom --M 8 --K 256 --N 512 --do_benchmarks False

# ============================================================
# Conv3d benchmarks
# ============================================================

# Conv3d roofline only
python float8_inference_roofline.py --recipe_name tensorwise --op_name conv3d --D 16 --H 16 --W 16 --kernel_size 3 --shape_gen_name custom --M 2 --K 64 --N 128 --do_benchmarks False

# Conv3d with actual benchmarks (requires mslk and kernel_size > 1)
# Note: benchmarks (do_benchmarks=True) runs the actual convolution on GPU and measures time, which may differ from roofline estimates due to various factors (kernel efficiency, fusion, etc.). If mslk fp8 conv is not available, this will raise an error since the conv benchmarks rely on that operator.
# python float8_inference_roofline.py --recipe_name tensorwise --op_name conv3d --D 16 --H 16 --W 16 --kernel_size 3 --padding 1 --shape_gen_name custom --M 2 --K 64 --N 128

# Conv3d with stride
python float8_inference_roofline.py --recipe_name tensorwise --op_name conv3d --D 32 --H 32 --W 32 --kernel_size 3 --stride 2 --padding 1 --shape_gen_name custom --M 4 --K 32 --N 64 --do_benchmarks False