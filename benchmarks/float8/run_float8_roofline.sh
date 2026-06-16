#!/bin/bash

cd /home/kgajdamo/workspace/repos/ao-kgajdamo/benchmarks/float8

# NOTE: `outfile` is a required first argument for float8_roofline.py.
# Each scenario below writes to its own CSV so results don't clobber each other.

# ============================================================
# Float8 recipes — pow2 shapes (default), with benchmarks
# ============================================================

# Tensorwise (default recipe when none specified)
python float8_roofline.py results_tensorwise.csv --float8_recipe_name tensorwise

# Rowwise
python float8_roofline.py results_rowwise.csv --float8_recipe_name rowwise

# Rowwise with high-precision grad_weight gemm
python float8_roofline.py results_rowwise_gw_hp.csv --float8_recipe_name rowwise_with_gw_hp

# Default (no recipe specified -> falls back to tensorwise)
python float8_roofline.py results_default.csv

# ============================================================
# Shape generators (each exercises a different shape set)
# ============================================================

# llama
python float8_roofline.py results_tensorwise_llama.csv --float8_recipe_name tensorwise --shape_gen_name llama

# pow2
python float8_roofline.py results_tensorwise_pow2.csv --float8_recipe_name tensorwise --shape_gen_name pow2

# pow2_extended
python float8_roofline.py results_tensorwise_pow2ext.csv --float8_recipe_name tensorwise --shape_gen_name pow2_extended

# sweep (large shape set — roofline only)
python float8_roofline.py results_tensorwise_sweep.csv --float8_recipe_name tensorwise --shape_gen_name sweep --do_benchmarks False

# ============================================================
# Limit iterations (quick smoke test)
# ============================================================

python float8_roofline.py results_smoke1.csv --float8_recipe_name tensorwise --n_limit 1
python float8_roofline.py results_smoke3.csv --float8_recipe_name rowwise --n_limit 3

# ============================================================
# Roofline only (skip actual GPU benchmarks)
# ============================================================

python float8_roofline.py results_tensorwise_roofline.csv --float8_recipe_name tensorwise --do_benchmarks False
python float8_roofline.py results_rowwise_roofline_llama.csv --float8_recipe_name rowwise --do_benchmarks False --shape_gen_name llama

# ============================================================
# Fusion modeling (uses LNLinearSigmoid + models float8 overhead fusion)
# ============================================================

python float8_roofline.py results_tensorwise_fusion.csv --float8_recipe_name tensorwise --enable_fusion_modeling True
python float8_roofline.py results_rowwise_fusion_llama.csv --float8_recipe_name rowwise --enable_fusion_modeling True --shape_gen_name llama

# ============================================================
# Gemm benchmark cache (reuse measured gemm times across runs)
# ============================================================

python float8_roofline.py results_tensorwise_cached.csv --float8_recipe_name tensorwise --gemm_cache_filename gemm_cache.json

# ============================================================
# MX format recipes (CUDA / sm100+ only — NOT supported on XPU)
# These raise NotImplementedError on XPU, so use roofline-only on CUDA.
# ============================================================

# python float8_roofline.py results_mxfp8_flexible.csv --mx_recipe_name mxfp8_32x32_flexible_gemm_layout --do_benchmarks False
# python float8_roofline.py results_mxfp8_weight.csv --mx_recipe_name mxfp8_32x32_weight --do_benchmarks False
# python float8_roofline.py results_mxfp4_cutlass.csv --mx_recipe_name mxfp4_cutlass --do_benchmarks False
