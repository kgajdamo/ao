set -ex

cd /home/kgajdamo/workspace/repos/ao-kgajdamo/benchmarks/float8

mkdir -p ./traces

# ============================================================
# Basic run (profile_path_prefix is the only required arg)
# Defaults: compile=True, model_type=linear, experiment_filter=both,
#           mode_filter=fwd_bwd, no recipe (-> plain Float8LinearConfig)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/basic

# ============================================================
# Float8 recipes (all three supported recipe names)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/fp8_tensorwise --float8_recipe_name tensorwise
python profile_lowp_training.py --profile_path_prefix ./traces/fp8_rowwise --float8_recipe_name rowwise
python profile_lowp_training.py --profile_path_prefix ./traces/fp8_rowwise_gw_hp --float8_recipe_name rowwise_with_gw_hp

# ============================================================
# Model types (all four)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/m_linear --model_type linear
python profile_lowp_training.py --profile_path_prefix ./traces/m_ln_linear --model_type ln_linear
python profile_lowp_training.py --profile_path_prefix ./traces/m_norm_ffn_norm --model_type norm_ffn_norm
python profile_lowp_training.py --profile_path_prefix ./traces/m_norm_ffn_norm_small --model_type norm_ffn_norm_small

# ============================================================
# Experiment filter (profile only lowp, only ref, or both)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/exp_both --experiment_filter both
python profile_lowp_training.py --profile_path_prefix ./traces/exp_lowp --experiment_filter lowp
python profile_lowp_training.py --profile_path_prefix ./traces/exp_ref --experiment_filter ref

# ============================================================
# Mode filter (fwd_bwd and fwd are the float8-compatible modes)
# Note: cast_only / cast_with_to_blocked / cast_only_dim0_dim1 are
# MX-only modes (see "Not supported on XPU" section below).
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/mode_fwd_bwd --mode_filter fwd_bwd
python profile_lowp_training.py --profile_path_prefix ./traces/mode_fwd --mode_filter fwd

# ============================================================
# Forward only (no backward)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/forward_only --forward_only True

# ============================================================
# Compile on/off
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/compile_on --compile True
python profile_lowp_training.py --profile_path_prefix ./traces/compile_off --compile False

# ============================================================
# Activation checkpointing
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/act_ckpt --enable_activation_checkpointing True
python profile_lowp_training.py --profile_path_prefix ./traces/act_ckpt_ffn --model_type norm_ffn_norm --enable_activation_checkpointing True

# ============================================================
# Inductor metadata in trace (annotates trace with kernel metadata)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/inductor_meta --add_inductor_metadata_to_trace True

# ============================================================
# Combined scenarios (mix recipe + model + filters)
# ============================================================

python profile_lowp_training.py --profile_path_prefix ./traces/combo_rowwise_ffn --float8_recipe_name rowwise --model_type norm_ffn_norm --mode_filter fwd_bwd
python profile_lowp_training.py --profile_path_prefix ./traces/combo_tw_lowp_fwd --float8_recipe_name tensorwise --experiment_filter lowp --mode_filter fwd
python profile_lowp_training.py --profile_path_prefix ./traces/combo_no_compile_ref --compile False --experiment_filter ref --model_type ln_linear

# ============================================================
# Not supported on XPU (documented for CUDA / sm100+ use only)
# ============================================================

# MX recipes raise: NotImplementedError("MXFP8TrainingRecipe is not supported on XPU yet")
# float8_recipe_name and mx_recipe_name are mutually exclusive.
# python profile_lowp_training.py --profile_path_prefix ./traces/mx --mx_recipe_name mxfp8_emulated

# The cast_* modes use MX-only config fields (elem_dtype/block_size/gemm_kernel_choice),
# so they require an MX recipe and therefore do not run on XPU.
# cast_only additionally requires --experiment_filter lowp.
# python profile_lowp_training.py --profile_path_prefix ./traces/cast_only --mx_recipe_name mxfp8_emulated --mode_filter cast_only --experiment_filter lowp
# python profile_lowp_training.py --profile_path_prefix ./traces/cast_blocked --mx_recipe_name mxfp8_emulated --mode_filter cast_with_to_blocked --experiment_filter lowp
# python profile_lowp_training.py --profile_path_prefix ./traces/cast_dim01 --mx_recipe_name mxfp8_emulated --mode_filter cast_only_dim0_dim1 --experiment_filter lowp