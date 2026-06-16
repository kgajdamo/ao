# Benchmark output reference

This document lists, for each benchmark under `benchmarks/`, the "headers"/"results"
columns it emits (printed table and/or CSV) along with a short description of each.

Conventions used below:
- **us** = microseconds, **ms** = milliseconds, **s** = seconds.
- **GB/s** / **gbps** = gigabytes per second (memory bandwidth).
- **TOPS / TFLOP/s** = (tera) operations / floating-point operations per second.
- **% peak** = achieved throughput as a fraction of the hardware theoretical peak.
- "fwd" = forward pass, "bwd" = backward pass, "e2e" = end-to-end.
- Scripts that only print free-form text or profiler traces (no fixed table schema)
  are noted as such, with the names of the metrics they print.

---

## float8/

### float8/bench_matmul.py

Benchmarks raw low-precision (fp8 / fp4) matmuls against a bf16 reference matmul over a
sweep of shapes. Results print as a pandas DataFrame and optionally save to CSV (`--out_filename`).

| Header | Description |
| --- | --- |
| `fast_accum` | Whether `use_fast_accum=True` was passed to the scaled matmul (fp8 runs both `True`/`False`; fp4 only `False`). |
| `name` | Shape-config identifier — integer index for `pow2`/`pow2_extended`/`sweep`, or a layer name (e.g. `attn.wqkv`) for `llama`/`dsv3`. |
| `M` | GEMM M dimension (output rows). |
| `K` | GEMM K dimension (contraction / inner dimension). |
| `N` | GEMM N dimension (output columns). |
| `ref_pct_top_peak` | bf16 reference throughput as a fraction (0–1) of bf16 peak TOPS. |
| `pct_top_peak` | Low-precision (fp8/fp4) throughput as a fraction (0–1) of the fp8/fp4 peak TOPS. |
| `ref_time_s` | bf16 reference matmul time (seconds; GPU kernel time by default). |
| `time_s` | Low-precision (fp8/fp4) matmul time (seconds). |
| `fp8_speedup` | Speedup of low-precision over bf16 (`ref_time_s / time_s`). |

### float8/bench_linear_float8.py

Benchmarks fwd+bwd of a `Float8Linear` vs a bf16 `nn.Linear` over a shape sweep. Prints a
full DataFrame plus a simplified table; optionally saves the full DataFrame to CSV (`-o`).

| Header | Description |
| --- | --- |
| `name` | Shape-config identifier from the shape generator. |
| `M` | Input/activation rows (batch·seq_len). |
| `K` | Linear in_features (contraction dim). |
| `N` | Linear out_features. |
| `scaling_repr` | String describing the float8 scaling config (`extra_repr` of the layer). |
| `ref_dtype` | Reference dtype (bf16). |
| `compiled` | Whether `torch.compile` was applied. |
| `use_fast_accum` | Whether fast-accumulation was enabled for the fp8 matmul. |
| `ref_time_sec` | bf16 reference fwd+bwd time (seconds). |
| `pt_fp8_time_sec` | float8 fwd+bwd time (seconds). |
| `ref_tops_sec` | bf16 throughput (TOPS, counts 3 gemms for fwd+bwd). |
| `ref_pct_top_peak` | bf16 throughput as fraction of bf16 peak. |
| `pt_fp8_tops_sec` | float8 throughput (TOPS). |
| `pt_fp8_pct_top_peak` | float8 throughput as fraction of fp8 peak. |
| `pt_fp8_speedup` | float8 speedup over bf16 (`ref_time_sec / pt_fp8_time_sec`); added after table build. |
| `shape` | `(M, K, N)` tuple as a string (added after table build). |

### float8/bench_padding.py

Compares unpadded bf16, alignment-padded bf16, and fp8 (pad-first) matmuls across awkward
shapes. Prints two tabulate tables ("TOPs" and "Speed Results").

TOPs table:

| Header | Description |
| --- | --- |
| `Shape` | `(MxKxN)` problem size. |
| `Ref Dtype` | Reference (high-precision) output dtype. |
| `Ref Tops` | bf16 reference throughput (TOPS). |
| `Aligned BF16 Tops` | Throughput of alignment-padded bf16 matmul (TOPS). |
| `FP8 Tops` | fp8 (pad-first) matmul throughput (TOPS). |
| `Ref % Peak` | bf16 reference as fraction of peak. |
| `Aligned BF16 % Peak` | Padded bf16 as fraction of peak. |
| `FP8 % Peak` | fp8 as fraction of fp8 peak. |

Speed Results table:

| Header | Description |
| --- | --- |
| `Shape` | `(M, K, N)` problem size. |
| `Ref Dtype` | Reference output dtype. |
| `Ref Time` | bf16 reference time (microseconds). |
| `Aligned BF16 Time` | Padded bf16 time (microseconds). |
| `FP8 Time` | fp8 time (microseconds). |
| `Aligned BF16 Speedup` | `Ref Time / Aligned BF16 Time`. |
| `FP8 Speedup` | `Ref Time / FP8 Time`. |

### float8/float8_roofline.py

Estimates the benefit of converting a Linear to float8 in fwd+bwd, combining a roofline
model with optional measured benchmarks. Prints a DataFrame and saves it to CSV (`outfile`).

| Header | Description |
| --- | --- |
| `fwd_M` | Forward-pass M dimension. |
| `fwd_K` | Forward-pass K dimension. |
| `fwd_N` | Forward-pass N dimension. |
| `r_bf16_gemm_s` | Roofline-modeled bf16 gemm time (fwd+bwd, 3 gemms), seconds. |
| `r_fp8_gemm_s` | Roofline-modeled fp8 gemm time (fwd+bwd), seconds. |
| `r_fp8_ovhd_s` | Roofline-modeled fp8 overhead (quant/scale reads-writes), seconds. |
| `r_fp8_gemm_and_ovhd_s` | Roofline fp8 gemm + overhead total, seconds. |
| `r_fp8_gemm_and_ovhd_spdp` | Roofline speedup: `r_bf16_gemm_s / r_fp8_gemm_and_ovhd_s`. |
| `b_bf16_gemm_s` | Benchmarked bf16 gemm time (fwd+bwd), seconds (0 if `--do_benchmarks=False`). |
| `b_fp8_gemm_s` | Benchmarked fp8 gemm time (fwd+bwd), seconds. |
| `b_bf16_e2e_s` | Benchmarked bf16 end-to-end fwd+bwd time, seconds. |
| `b_fp8_e2e_s` | Benchmarked fp8 end-to-end fwd+bwd time, seconds. |
| `b_fp8_e2e_spdp` | Benchmarked e2e speedup (`b_bf16_e2e_s / b_fp8_e2e_s`; −1 if not run). |
| `rb_bf16_gemm_ratio` | Roofline-vs-benchmark bf16 gemm ratio (−1 if not benchmarked). |
| `rb_fp8_gemm_ratio` | Roofline-vs-benchmark fp8 gemm ratio (−1 if not benchmarked). |

### float8/float8_inference_roofline.py

Inference variant of the roofline (fwd only), supporting linear and conv2d/conv3d ops.
Prints a DataFrame and saves to CSV (`outfile`). For `op_name="linear"` the conv-only
columns `D`, `H`, `W`, `kernel_size` are dropped.

| Header | Description |
| --- | --- |
| `fwd_M` | Forward M (for conv: batch size). |
| `fwd_K` | Forward K (for conv: in_channels). |
| `fwd_N` | Forward N (for conv: out_channels). |
| `D` | Conv input depth (conv3d only). |
| `H` | Conv input height. |
| `W` | Conv input width. |
| `kernel_size` | Conv kernel size. |
| `r_bf16_gemm_s` | Roofline-modeled bf16 gemm time, seconds. |
| `r_fp8_gemm_s` | Roofline-modeled fp8 gemm time, seconds. |
| `r_bf16_ovhd_s` | Roofline bf16 overhead time (only with fusion modeling), seconds. |
| `r_fp8_ovhd_s` | Roofline fp8 overhead time, seconds. |
| `r_fp8_gemm_and_ovhd_s` | Roofline fp8 gemm + overhead total, seconds. |
| `r_fp8_gemm_and_ovhd_spdp` | Roofline speedup including overhead. |
| `b_bf16_gemm_s` | Benchmarked bf16 gemm time, seconds. |
| `b_fp8_gemm_s` | Benchmarked fp8 gemm time, seconds. |
| `b_bf16_e2e_s` | Benchmarked bf16 e2e time, seconds. |
| `b_fp8_e2e_s` | Benchmarked fp8 e2e time, seconds. |
| `b_fp8_e2e_spdp` | Benchmarked e2e speedup. |
| `rb_bf16_gemm_ratio` | Roofline-vs-benchmark bf16 gemm ratio. |
| `rb_fp8_gemm_ratio` | Roofline-vs-benchmark fp8 gemm ratio. |

### float8/measure_achievable_specs.py

Empirically measures the achievable fraction of peak gemm TOPS and peak memory bandwidth
(the roofline "fudge factors"). Prints per-shape sweep tables; optionally writes them to CSV
(`--out_filename`). Also prints suggested spec values.

bf16 / fp8 gemm sweep table:

| Header | Description |
| --- | --- |
| `dtype` | Data type of the measured gemm. |
| `M` / `K` / `N` | Square gemm dimensions (M=K=N). |
| `time_s` | Median gemm time (seconds). |
| `tops_sec` | Achieved throughput (TOPS). |

Memory-bandwidth sweep table:

| Header | Description |
| --- | --- |
| `numel` | Number of elements in the 1-D buffer. |
| `bytes_rw` | Total bytes read+written. |
| `time_s` | Median kernel time (seconds). |
| `gbps` | Achieved memory bandwidth (GB/s). |

Suggested-spec printed values: `pct_achievable_gemm_tops` (best gemm TOPS / peak) and
`pct_achievable_mem_bw` (best bandwidth / peak).

### float8/profile_lowp_training.py

Profiles fwd+bwd of a layer with float8/mx training and summarizes GPU time by kernel.
Prints two tables (no CSV).

"Summary of GPU time by CPU kernel" table:

| Header | Description |
| --- | --- |
| `experiment` | Which run the row belongs to (`0_ref` = bf16 reference, `1_lowp` = low-precision). |
| `kernel` | CPU kernel/op name. |
| `category` | Kernel category (e.g. gemm, overhead) from `kernel_name_to_category`. |
| `time_ms` | GPU time attributed to that kernel (ms). |
| `pct_gpu_time` | Kernel's share of total GPU time (%). |
| `bw_gpbs` | Achieved memory bandwidth for the kernel (GB/s), when parseable. |

"Summary of time (ms) by kernel category" pivot: rows are kernel categories, columns are the
experiments (`0_ref`, `1_lowp`); when both run it also adds `lowp_div_ref` and `ref_div_lowp`
(per-category time ratios).

---

## Top-level benchmarks/

### benchmark_aq.py

Affine-quantized tensor (int8dq / int8wo / int4wo) vs bf16 perf for a single linear. No table —
prints one free-form line per shape:
`(M, N, K): elapsed time: <quantized>, bf16 elapsed time: <bf16>` (times from `benchmark_model`).

### benchmark_blockwise_scaled_linear_triton.py

Blockwise fp8 Triton gemm vs fp16 linear. Builds two DataFrames, each saved to CSV
(`blockwise_triton_latency_results.csv`, `blockwise_triton_precision_results.csv`).

Latency results:

| Header | Description |
| --- | --- |
| `m` / `k` / `n` | Problem dimensions. |
| `block_size` | Blockwise quantization block size. |
| `dtype` | fp8 dtype used (e4m3 / e5m2). |
| `fp16_latency (ms)` | Reference fp16 `F.linear` latency. |
| `blockwise_latency (ms)` | Blockwise fp8 gemm latency. |
| `blockwise_speedup` | `fp16_latency / blockwise_latency`. |

Precision results:

| Header | Description |
| --- | --- |
| `m` / `k` / `n` | Problem dimensions. |
| `block_size` | Block size. |
| `dtype` | fp8 dtype. |
| `error_blockwise (dB)` | SQNR-style error of blockwise output vs reference (decibels). |

### benchmark_gptq.py

Times `gptq_quantize` for a single (K, N). No table — prints free-form: `K=…, N=…` and either
`gptq_quantize avg time: <s>` (timed run) or `gptq_quantize time: <s>` (profiling run).

### benchmark_gpu_sparsity.py

Dense vs sparse (semi-structured / block-sparse) linear or mm across shape sets. Prints a
DataFrame; optionally saves to CSV (`-save`).

| Header | Description |
| --- | --- |
| `test_function` | Op benchmarked (`linear` or `mm`). |
| `m` / `k` / `n` | Problem dimensions. |
| `dtype` | Data type. |
| `sparse` | Sparse eager time. |
| `dense` | Dense eager time. |
| `dense_c` | Dense `torch.compile` (max-autotune) time. |
| `sparse_c` | Sparse `torch.compile` time. |
| `speedup (d/s)` | Best-dense / best-sparse speedup. |

### benchmark_hqq.py

HQQ int4 mixed matmul: reference dequant vs Triton kernel vs tinygemm int4. Builds a DataFrame
printed as CSV (to stdout).

| Header | Description |
| --- | --- |
| `M` / `N` / `K` | Problem dimensions. |
| `group_size` | HQQ quantization group size. |
| `dtype` | Compute dtype. |
| `transposed` | Whether the transposed-matmul variant was benchmarked. |
| `ref` | Reference (dequantize + matmul) time (ms). |
| `triton` | Triton mixed int4 matmul time (ms). |
| `tinygemm` | Torch tinygemm int4 matmul time (ms; −1 if not run). |

### benchmark_low_bit_adam.py

Fine-tuning benchmark for low-bit optimizers (ViT/timm). No results table — logs metrics to
Weights & Biases each step and prints peak memory. Logged metrics:

| Metric | Description |
| --- | --- |
| `loss` | Training cross-entropy loss. |
| `lr` | Current learning rate. |
| `imgs_per_second` | Training throughput (images/sec). |
| `val_acc` | Validation accuracy (per epoch). |
| `max_memory_allocated` | Peak device memory allocated (GB). |

### benchmark_rowwise_scaled_linear_sparse_cutlass.py

Rowwise-scaled fp8 sparse CUTLASS linear vs fp16/fp8/cuSPARSELt. Builds a DataFrame saved to CSV
(`rowwise_scaled_linear_sparse_cutlass_time_results.csv`) and printed as markdown.

| Header | Description |
| --- | --- |
| `m` / `k` / `n` | Problem dimensions. |
| `fp16_latency (ms)` | Reference fp16 `F.linear` latency. |
| `fp8_latency (ms)` | Dense fp8 `_scaled_mm` latency. |
| `rowwise_scaled_linear_sparse_cutlass_f8f8 latency (ms)` | Rowwise-scaled fp8 sparse CUTLASS latency. |
| `cusparselt latency (ms)` | cuSPARSELt sparse matmul latency. |
| `f8f8 speedup (d/s)` | Dense-fp8 / sparse-fp8 speedup. |

### benchmark_semi_sparse_training.py

2:4 semi-structured sparse training vs dense (Linear / SAM ViT). Prints a DataFrame; optionally
saves to CSV (`--save`).

| Header | Description |
| --- | --- |
| `sparsity_config` | Variant label (e.g. `dense_linear`, `semi_sparse_linear`). |
| `mkn` | `(M, K, N)` case (modes `linear`/`llama3-8b`). |
| `model_type` | Model variant (mode `vit`). |
| `batch_size` | Batch size (mode `vit`). |
| `time` | Median fwd/bwd time (ms), or `OOM`. |
| `memory` | Peak memory used (GB), or `OOM`. |

(The case columns present depend on the `--mode`: `mkn` for linear/llama, `model_type`+`batch_size` for vit.)

### benchmark_sparse_conversion_cutlass.py

Times two semi-structured-sparse compression kernels. Builds a DataFrame saved to CSV and
printed as markdown.

| Header | Description |
| --- | --- |
| `cutlass_reference (ms)` | Reference CUTLASS sm9x f8 compression time. |
| `cutlass_custom (ms)` | Custom `sparse_semi_structured_tile` compression time. |

### benchmark_uintx.py

uintx (1–7 bit) weight-only quant vs fp16 across a few scales. No table — prints free-form:
`scale: <s> fp16 time:<ms>ms speedups:` then one line per bit width `int<nbits>: <x>x`.

### intmm.py

int8 matmul (`_int_mm`) and int-scaled matmul vs fp `torch.mm`, over shapes from a CSV. Prints
CSV-formatted rows to stdout with header `fn,m,k,n,fp_time,int_mm_time,ratio`.

| Column | Description |
| --- | --- |
| `fn` | Benchmark function (`run_int_mm_benchmark` or `run_int_scaled_mm_benchmark`). |
| `m` / `k` / `n` | Problem dimensions. |
| `fp_time` | Reference floating-point matmul time (ms). |
| `int_mm_time` | int8 (scaled) matmul time (ms). |
| `ratio` | `fp_time / int_mm_time` speedup. |

### print_config_shapes.py

Utility that prints the shapes of the autotuner's best-config cache. No benchmarking — prints
CSV with header `m,k,n` (one row per cached gemm config).

---

## inference/

### inference/bench_float8_inference.py

Times a single compiled float8 dynamic-activation/weight linear. No table — prints one line:
`time_us <value>` (median inference time in microseconds).

---

## microbenchmarks/ and dashboard/

### microbenchmarks/benchmark_runner.py (+ benchmark_inference.py)

Config-driven inference microbenchmark harness (quantization/sparsity recipes × shapes). Writes
a results CSV (`generate_results_csv`) and prints a tabulate table (`print_results`).

Printed table headers:

| Header | Description |
| --- | --- |
| `Name` | Benchmark/config name. |
| `Quantization` | Quantization recipe (or `baseline`). |
| `Sparsity` | Sparsity recipe (or `none`). |
| `Shape` | `shape_name (M, K, N)`. |
| `Eager Baseline Inference Time (ms)` | Baseline (unquantized) eager inference time. |
| `Eager Model Inference Time (ms)` | Quantized eager inference time. |
| `Eager Speedup` | Baseline-eager / quantized-eager speedup. |
| `Compile Baseline Inference Time (ms)` | Baseline compiled inference time. |
| `Compile Model Inference Time (ms)` | Quantized compiled inference time. |
| `Compile Speedup` | Baseline-compiled / quantized-compiled speedup. |
| `Eager vs Compile Speedup` | Quantized-eager / quantized-compiled speedup. |
| `Profiler Enabled` | Whether the profiler was enabled. |

The CSV additionally carries config fields (`name`, `quantization`, `sparsity`, `m`, `k`, `n`,
`high_precision_dtype`, `torch_compile_mode`, `device`, `model_type`, `output_dir`,
`enable_profiler`, `enable_memory_profiler`), the raw timing fields
(`baseline_model_eager_inference_time_in_ms`, `quantized_model_eager_inference_time_in_ms`,
`baseline_model_compiled_inference_time_in_ms`, `quantized_model_compiled_inference_time_in_ms`,
`eager speedup on baseline`, `compile speedup on baseline`, `eager vs compile speedup`), and
profiling outputs (`profiler_json_path`, `memory_profile_path`, `memory_visualization_path`,
`memory_stats`). `memory_stats` holds `allocated_bytes.all.peak`, `active_bytes.all.peak`,
`reserved_bytes.all.peak` (each in MB).

### dashboard/ci_microbenchmark_runner.py

CI wrapper that runs the same microbenchmarks and emits JSON in the PyTorch OSS benchmark-DB
format. Each entry carries one metric:

| Metric name | Description |
| --- | --- |
| `Fwd Speedup (x)` | Compiled quantized speedup over baseline (`compile_speedup_on_baseline`). |
| `Bfloat16 Fwd Time (ms)` | Baseline compiled forward time. |
| `Quantized Fwd Time (ms)` | Quantized compiled forward time. |
| `Allocated Memory (MB)` | Peak allocated memory (`allocated_bytes.all.peak`). |

---

## mx_formats/

### mx_formats/cast_bench.py

Benchmarks mx/nvfp4 cast (quantization) kernels in various modes. No table — prints config lines
plus two metrics:

| Metric | Description |
| --- | --- |
| `time_us` | Median kernel time (microseconds). |
| `mem_bw_gbps` | Achieved memory bandwidth (GB/s). |

---

## prototype/moe_training/

### prototype/moe_training/benchmark_scaled_grouped_mm_dq.py

bf16 vs scaled (fp8/mx) grouped-mm for MoE, fwd-only and fwd+bwd. Prints a tabulate table.

| Header | Description |
| --- | --- |
| `M,N,K,G` | Problem tuple: tokens M, out dim N, contraction K, number of groups/experts G. |
| `recipe` | Quantization recipe used for the scaled grouped mm. |
| `bf16_fwd_bwd_us` | bf16 fwd+bwd time (us). |
| `scaled_fwd_bwd_us` | Scaled (quantized) fwd+bwd time (us). |
| `scaled_fwd_bwd_speedup` | fwd+bwd speedup (`bf16 / scaled`), suffixed `x`. |
| `bf16_fwd_us` | bf16 forward-only time (us). |
| `scaled_fwd_us` | Scaled forward-only time (us). |
| `scaled_fwd_speedup` | forward-only speedup, suffixed `x`. |

### prototype/moe_training/bench_2d_3d_grouped_gemm.py

bf16 vs fp8-rowwise vs mxfp8 for a 2D×3D grouped gemm. Prints a tabulate table.

| Header | Description |
| --- | --- |
| `E` | Number of experts/groups. |
| `M` / `N` / `K` | Grouped-gemm dimensions. |
| `bf16_time_us` | bf16 grouped-mm time (us). |
| `fp8_rowwise_time_us` | fp8 rowwise grouped-mm time (us; `inf` if unsupported). |
| `mxfp8_time_us` | mxfp8 grouped-mm time (us; `inf` if unsupported). |
| `bf16_tflops` | bf16 throughput (TFLOP/s). |
| `fp8_rowwise_tflops` | fp8 rowwise throughput (TFLOP/s). |
| `mxfp8_tflops` | mxfp8 throughput (TFLOP/s). |
| `fp8_rowwise_speedup` | bf16 / fp8-rowwise speedup, suffixed `x`. |
| `mxfp8_speedup` | bf16 / mxfp8 speedup, suffixed `x`. |

### prototype/moe_training/bench_moe_layer.py

End-to-end MoE layer (bf16 vs a quantized recipe), fwd+bwd. No fixed table — prints free-form
lines with the shape config and `bf16 time` (ms), `<recipe> time` (ms), and `speedup` (`x`).

### prototype/moe_training/benchmark_moe_layer_fsdp.py

MoE layer under FSDP (multi-GPU). No fixed table — prints free-form `BF16 time` (us),
`Scaled time` (us), and `Speedup` (`x`).

### prototype/moe_training/fp8_rowwise/

These benchmark individual fp8-rowwise quantization/scaling Triton kernels vs torch references.
Each prints a tabulate table (no CSV) built from `ExperimentConfig`/`ExperimentResult`.

#### bench_triton_fp8_per_group_colwise_scales.py

| Header | Description |
| --- | --- |
| `Mg,K` | Input shape (total tokens Mg, cols K). |
| `n_groups` | Number of expert groups. |
| `high_precision_dtype` | Input high-precision dtype. |
| `torch_loop_time_us` | Compiled torch per-group loop time (us). |
| `triton_time_us` | Triton kernel time (us). |
| `torch_mem_bw_gbps` | Torch path bandwidth (GB/s). |
| `triton_mem_bw_gbps` | Triton path bandwidth (GB/s). |
| `triton_speedup` | torch / triton speedup, suffixed `x`. |

#### bench_triton_fp8_per_group_rowwise_scales.py

| Header | Description |
| --- | --- |
| `Mg,N` | Input shape (tokens Mg, cols N). |
| `n_groups` | Number of expert groups. |
| `torch_loop_time_us` | Compiled torch per-group loop time (us). |
| `triton_time_us` | Triton kernel time (us). |
| `triton_transpose_us` | Triton transpose-then-rowwise variant time (us). |
| `torch_mem_bw_gbps` | Torch path bandwidth (GB/s). |
| `triton_mem_bw_gbps` | Triton path bandwidth (GB/s). |
| `triton_transpose_mem_bw_gbps` | Triton-transpose path bandwidth (GB/s). |
| `triton_speedup` | torch / triton speedup, suffixed `x`. |
| `triton_transpose_speedup` | torch / triton-transpose speedup, suffixed `x`. |

#### bench_triton_fp8_per_group_colwise_scales_dual.py

| Header | Description |
| --- | --- |
| `M` | Shared rows of both tensors. |
| `N1` | Cols of tensor 1. |
| `N2` | Cols of tensor 2. |
| `n_groups` | Number of expert groups. |
| `dtype` | Input high-precision dtype. |
| `two calls (us)` | Two sequential single-tensor calls (baseline), us. |
| `dual (us)` | Single fused dual-tensor call, us. |
| `speedup` | two-calls / dual speedup, suffixed `x`. |

#### bench_triton_fp8_colwise_3d_scale_and_cast.py

| Header | Description |
| --- | --- |
| `shape (E, K, N)` | 3D input shape (after transpose). |
| `dtype` | Input high-precision dtype. |
| `torch.compile (us)` | Compiled native 3-op sequence time (us). |
| `triton (us)` | Fused Triton kernel time (us). |
| `speedup` | torch.compile / triton speedup. |
| `torch.compile BW (GB/s)` | torch.compile path bandwidth. |
| `triton BW (GB/s)` | Triton path bandwidth. |

#### bench_triton_fp8_rowwise_3d_transpose_rhs.py

| Header | Description |
| --- | --- |
| `input_shape` | 3D input shape. |
| `power_of_2_scales` | Whether scales are rounded to powers of 2. |
| `torch_time_us` | Compiled torch path time (us). |
| `triton_atomic_time_us` | Triton atomic-reduction kernel time (us). |
| `triton_reduction_time_us` | Triton reduction kernel time (us). |
| `torch_mem_bw_gbps` | Torch path bandwidth (GB/s). |
| `triton_atomic_mem_bw_gbps` | Triton-atomic path bandwidth (GB/s). |
| `triton_reduction_mem_bw_gbps` | Triton-reduction path bandwidth (GB/s). |
| `triton_atomic_speedup` | torch / triton-atomic speedup, `x`. |
| `triton_reduction_speedup` | torch / triton-reduction speedup, `x`. |

#### bench_triton_fp8_rowwise_2d_fused_scale_and_cast.py

| Header | Description |
| --- | --- |
| `input_shape` | 2D input shape (string). |
| `dtype` | Input high-precision dtype. |
| `torch.compile (us)` | Compiled native sequence time (us). |
| `triton (us)` | Fused Triton kernel time (us). |
| `speedup` | torch.compile / triton speedup, `x`. |
| `torch.compile BW (GB/s)` | torch.compile path bandwidth. |
| `triton BW (GB/s)` | Triton path bandwidth. |

#### bench_colwise_block_configs.py

Sweeps Triton block/warp configs for the colwise-scales kernel. Prints a "Best per Shape" and a
"Full Results" tabulate table sharing these headers:

| Header | Description |
| --- | --- |
| `Mg` | Total token rows (jagged dim). |
| `K` | Hidden cols. |
| `n_groups` | Number of expert groups. |
| `block_size_n` | Triton tile size over N. |
| `block_size_k` | Triton inner-loop tile size over K. |
| `num_warps` | Warps per kernel launch. |
| `time (us)` | Median kernel time (us); best/minimum in the per-shape table. |

### prototype/moe_training/mxfp8/

Each benchmarks an mxfp8 MoE helper kernel vs a torch/triton/cuda reference and prints a
tabulate table (no CSV), except `roofline_unified.py` (see below).

#### bench_dequantize.py

| Header | Description |
| --- | --- |
| `input_shape` | `(local_bs, seq_len, dim)` bf16 input. |
| `torch_us` | Compiled torch dequant time (us). |
| `triton_us` | Triton dequant time (us). |
| `torch_gbps` | Torch path bandwidth (GB/s). |
| `triton_gbps` | Triton path bandwidth (GB/s). |
| `triton_speedup` | torch / triton speedup. |

#### bench_quantize_3d.py

| Header | Description |
| --- | --- |
| `input_shape` | `(E, N, K)` bf16 input shape. |
| `scaling_mode` | Scale rounding mode (FLOOR / RCEIL). |
| `variant` | Layout variant (`32x1_n`, `32x1_t`, `32x32_n`, `32x32_t`). |
| `cuda_2d_us` | CUDA 2D-path quantize time (us; NaN unless `32x1_n`). |
| `cutedsl_3d_us` | CuTeDSL 3D kernel time (us). |
| `to_mx_us` | Reference `to_mx` time (us). |
| `cuda_2d_gbps` | CUDA 2D bandwidth (GB/s). |
| `cutedsl_3d_gbps` | CuTeDSL 3D bandwidth (GB/s). |
| `to_mx_gbps` | Reference bandwidth (GB/s). |

#### bench_cutedsl_quantize_2d_1x32.py

| Header | Description |
| --- | --- |
| `input_shape` | `(M, K)` bf16 input. |
| `scaling_mode` | Scale rounding mode (floor / rceil). |
| `num_groups` | Number of jagged groups. |
| `cutedsl_blocked_us` | CuTeDSL blocked-output kernel time (us). |
| `triton+rearrange_us` | Triton quant + rearrange time (us). |
| `speedup` | triton+rearrange / cutedsl ratio, `x`. |
| `cutedsl_gbps` | CuTeDSL bandwidth (GB/s). |
| `triton+rearrange_gbps` | Triton+rearrange bandwidth (GB/s). |

#### bench_cutedsl_quantize_2d_32x1.py

| Header | Description |
| --- | --- |
| `input_shape` | `(M, K)` bf16 input. |
| `scaling_mode` | Scale rounding mode (rceil). |
| `num_groups` | Number of jagged groups. |
| `cutedsl_blocked_us` | CuTeDSL blocked-output kernel time (us). |
| `cuda+rearrange_us` | CUDA quant + rearrange time (us). |
| `speedup` | cuda+rearrange / cutedsl ratio, `x`. |
| `cutedsl_gbps` | CuTeDSL bandwidth (GB/s). |
| `cuda+rearrange_gbps` | CUDA+rearrange bandwidth (GB/s). |

#### bench_pad_token_groups.py / bench_unpad_token_groups.py

(Identical schema; pad vs unpad of token groups.)

| Header | Description |
| --- | --- |
| `num_tokens` | Total token count. |
| `dim` | Feature dimension. |
| `num_groups` | Number of token groups. |
| `torch_us` | Torch eager time (us). |
| `cuda_us` | CUDA kernel time (us). |
| `torch_mem_bw_gbps` | Torch bandwidth (GB/s). |
| `cuda_mem_bw_gbps` | CUDA bandwidth (GB/s). |
| `cuda_vs_torch` | torch / cuda speedup (`x` or `N/A`). |

#### bench_triton_mx_block_rearrange_per_group_3d.py

| Header | Description |
| --- | --- |
| `input_shape` | `(E, N, K)` scale tensor shape. |
| `torch_time_us` | Compiled torch rearrange time (us). |
| `triton_time_us` | Triton kernel time (us). |
| `torch_mem_bw_gbps` | Torch bandwidth (GB/s). |
| `triton_mem_bw_gbps` | Triton bandwidth (GB/s). |
| `triton_speedup` | torch / triton speedup. |

#### bench_mx_block_rearrange_2d_K_groups.py

| Header | Description |
| --- | --- |
| `input_shape` | `(M, scale_cols)` scale tensor shape. |
| `num_groups` | Number of jagged groups along K. |
| `torch_time_us` | Torch rearrange time (us). |
| `triton_time_us` | Triton kernel time (us). |
| `triton_speedup` | torch / triton speedup. |
| `torch_mem_bw_gbps` | Torch bandwidth (GB/s). |
| `triton_mem_bw_gbps` | Triton bandwidth (GB/s). |

#### bench_mx_block_rearrange_2d_M_groups.py

| Header | Description |
| --- | --- |
| `input_shape` | `(Mg, K)` scale tensor shape. |
| `chunks_per_tb` | Chunks-per-threadblock tuning param for the CUDA kernel. |
| `torch_time_us` | Compiled torch rearrange time (us). |
| `triton_time_us` | Triton kernel time (us). |
| `cuda_time_us` | CUDA kernel time (us). |
| `triton_speedup` | torch / triton speedup, `x`. |
| `cuda_speedup` | torch / cuda speedup, `x`. |

#### bench_all_to_all_v.py

| Header | Description |
| --- | --- |
| `input_shape` | `(batch_size, seq_len, dim)` input. |
| `num_splits` | Number of all-to-all splits (= world size). |
| `fwd_bf16_ms` | bf16 all-to-all forward time (ms). |
| `fwd_mxfp8_ms` | mxfp8 all-to-all forward time (ms). |
| `bwd_bf16_ms` | bf16 backward time (ms). |
| `bwd_mxfp8_ms` | mxfp8 backward time (ms). |

#### bench_ep_pipeline.py

Expert-parallel pipeline (bf16 vs mxfp8), rank-0 prints a tabulate table:

| Header | Description |
| --- | --- |
| `tokens` | Number of tokens. |
| `dim` | Model dimension. |
| `hidden_dim` | Expert hidden dimension. |
| `num_experts` | Number of experts. |
| `fwd_bf16_ms` | bf16 forward time (ms). |
| `fwd_mxfp8_ms` | mxfp8 forward time (ms). |
| `fwd_speedup` | Forward bf16 / mxfp8 speedup, `x`. |
| `bwd_bf16_ms` | bf16 backward time (ms). |
| `bwd_mxfp8_ms` | mxfp8 backward time (ms). |
| `bwd_speedup` | Backward speedup, `x`. |
| `total_speedup` | Total fwd+bwd speedup, `x`. |

#### roofline_unified.py

Unified mxfp8 MoE roofline: builds several DataFrames, writes 3 CSVs, prints a free-form
SUMMARY STATISTICS block, and saves a PNG plot.

`speedup_results.csv` (e2e roofline vs measured grouped-mm):

| Column | Description |
| --- | --- |
| `M` | Local batch×seq length (varied dimension). |
| `K` | Reduction dimension. |
| `N` | Output dimension per group. |
| `num_groups` | Number of groups. |
| `shape_label` | Shape label (e.g. `M=16384`). |
| `bf16_roofline_time_ms` | Modeled bf16 fwd+bwd time. |
| `mxfp8_roofline_quant_time_ms` | Modeled mxfp8 quantization time. |
| `mxfp8_roofline_gemm_time_ms` | Modeled mxfp8 gemm time. |
| `mxfp8_roofline_total_time_ms` | Modeled mxfp8 total time. |
| `roofline_speedup` | Modeled bf16 / mxfp8 speedup. |
| `roofline_quant_overhead_pct` | Quant time as % of mxfp8 total. |
| `bf16_actual_time_ms` | Measured bf16 grouped-mm fwd+bwd time. |
| `mxfp8_actual_time_ms` | Measured mxfp8 grouped-mm fwd+bwd time. |
| `actual_speedup` | Measured bf16 / mxfp8 speedup. |

`quant_2d_results.csv` (2D quantize kernel efficiency): `M`, `K`, `shape_label`,
`roofline_time_ms`, `peak_bandwidth_gbps`, `total_bytes_gb`, `triton_to_mxfp8_dim0_us`,
`triton_dim0_bandwidth_gbps`, `triton_dim0_efficiency_pct`, `to_mxfp8_dim1_cuda_us`,
`cuda_dim1_bandwidth_gbps`, `cuda_dim1_efficiency_pct` — modeled vs measured time, achieved
bandwidth, and % of peak for the dim0 (triton) and dim1 (cuda) casts.

`quant_3d_results.csv` (3D quantize kernel efficiency): `num_experts`, `N`, `K`, `shape_label`,
`roofline_time_ms`, `peak_bandwidth_gbps`, `total_bytes_gb`, `mxfp8_quantize_cuda_3d_us`,
`cuda_3d_bandwidth_gbps`, `cuda_3d_efficiency_pct` — modeled time, measured CUDA 3D quantize
time, achieved bandwidth, and % of peak.

(Additional in-memory DataFrames drive the PNG plot — block-rearrange and gemm-efficiency rows
with `roofline_*`, measured `*_us`/`*_ms`, `*_bandwidth_gbps`/`*_tflops`, and `*_efficiency_pct`
columns.)

---

## prototype/blockwise_fp8_training/

### bench_1x128_128x128_gemms.py

bf16 vs blockwise-fp8 (`1x128_128x128`) Triton gemm vs `_scaled_mm`. Prints a tabulate table.

| Header | Description |
| --- | --- |
| `M` / `N` / `K` | GEMM dimensions. |
| `out_dtype` | Output dtype. |
| `bf16_mm_us` | bf16 `torch.mm` time (us). |
| `fp8_triton_us` | fp8 Triton `1x128_128x128` gemm time (us). |
| `fp8_scaled_mm_us` | fp8 `_scaled_mm` time (us). |
| `bf16 tflops/sec` | bf16 throughput (TFLOP/s). |
| `triton tflops/sec` | Triton fp8 throughput (TFLOP/s). |
| `scaled_mm tflops/sec` | `_scaled_mm` fp8 throughput (TFLOP/s). |

### bench_1x128_128x1_gemms.py

Same schema as above but the Triton path is the `1x128_128x1` gemm (simulating
`grad_weight = grad_output_t @ input`): `M`, `N`, `K`, `out_dtype`, `bf16_mm_us`,
`fp8_triton_us`, `fp8_scaled_mm_us`, `bf16 tflops/sec`, `triton tflops/sec`,
`scaled_mm tflops/sec`.

### bench_linear_fwd_bwd.py

Blockwise-fp8 Linear fwd+bwd (Triton vs scaled_mm backends) vs bf16. Prints a tabulate table.

| Header | Description |
| --- | --- |
| `M` / `N` / `K` | Problem dimensions. |
| `out_dtype` | Output dtype. |
| `bf16_mm_linear_us` | bf16 linear fwd+bwd time (us). |
| `fp8_triton_linear_us` | Blockwise-fp8 (Triton backend) fwd+bwd time (us). |
| `fp8_scaled_mm_linear_us` | Blockwise-fp8 (`_scaled_mm` backend) fwd+bwd time (us). |

### benchmark_quant_kernel_bandwidth.py

Blockwise-fp8 quant-kernel bandwidth study. Prints three tabulate tables and optionally a CSV.

Per-shape table: `kernel`, `shape` (`MxK`), `kernel_us`, `effective_logical_io_gbps`,
`logical_io_vs_achievable_%` — kernel name, shape, kernel time, modeled read+write bandwidth,
and bandwidth as % of the GPU's achievable bandwidth.

Overall table: `kernel`, `avg_effective_logical_io_gbps`, `avg_logical_io_vs_achievable_%`,
`worst_case_logical_io_vs_achievable_%` — per-kernel averages and worst-case % across shapes.

Skipped table: `skipped_kernel`, `shape`, `reason` — kernel/shape combos skipped and why.

CSV columns: `kernel`, `m`, `k`, `kernel_us`, `effective_logical_io_gbps`,
`logical_io_vs_achievable_%`, `achievable_bandwidth_gbps`, plus the achievable-vs-peak figure and
its source string.

### bench_linear_roofline.py

Blockwise-fp8 Linear roofline (modeled + measured). Prints a DataFrame and optionally writes CSV.

| Column | Description |
| --- | --- |
| `name` | Shape/config name. |
| `fp8_backend` | `blockwise_triton_gemm` or `blockwise_scaled_mm`. |
| `compiled` | Whether `torch.compile` was used. |
| `fwd_M` / `fwd_K` / `fwd_N` | Forward dimensions. |
| `r_bf16_gemm_s` | Roofline bf16 gemm time (s). |
| `r_fp8_gemm_s` | Roofline fp8 gemm time (s). |
| `r_fp8_ovhd_s` | Roofline fp8 overhead time (s). |
| `r_fp8_gemm_and_ovhd_s` | Roofline fp8 gemm + overhead total (s). |
| `r_fp8_gemm_and_ovhd_spdp` | Roofline-predicted speedup. |
| `b_bf16_e2e_s` | Measured bf16 e2e fwd+bwd time (s). |
| `b_fp8_e2e_s` | Measured fp8 e2e fwd+bwd time (s). |
| `b_fp8_e2e_spdp` | Measured e2e speedup (`b_bf16_e2e_s / b_fp8_e2e_s`). |
| `b_fp8_e2e_spdp_ratio_of_r` | Measured speedup ÷ roofline-predicted speedup. |

---

## prototype/nvfp4_training/

These three time NVFP4 cast/Hadamard kernels. Each prints a tabulate table; in
"representative-models" mode `model` and `shape` columns are prepended.

### bench_quantize_2d.py

| Header | Description |
| --- | --- |
| `model` | (representative-models mode) model name. |
| `shape` | (representative-models mode) tensor role/shape label. |
| `M` | Rows. |
| `N` | Columns. |
| `time_us` | Kernel time (us). |
| `gbps` | Effective bandwidth (read + packed fp4 writes + fp8 scale writes), GB/s. |

### bench_hadamard_amax.py

| Header | Description |
| --- | --- |
| `model` / `shape` | (representative-models mode) labels. |
| `M` | Rows. |
| `N` | Columns. |
| `time_us` | Kernel time (us). |
| `gbps` | Effective read bandwidth (GB/s). |

### bench_hadamard_quantize_row_col.py

| Header | Description |
| --- | --- |
| `model` / `shape` | (representative-models mode) labels. |
| `M` | Rows. |
| `N` | Columns. |
| `rounding` | Rounding mode (`rtne` or `rs` stochastic). |
| `time_us` | Kernel time (us). |
| `gbps` | Effective bandwidth (read + col write + row write), GB/s. |
| `pct_peak_mem_bw` | Bandwidth as % of peak memory bandwidth (or `n/a`). |

---

## prototype/attention/

### benchmark_sdpa.py

Compares two SDPA backends across sequence lengths. Prints a manually formatted table and
returns a list of result dicts.

| Column | Description |
| --- | --- |
| `SeqLen` / `seq_len` | Sequence length S. |
| `<baseline> (ms)` / `baseline_time_ms` | Baseline backend median runtime (ms). |
| `<test> (ms)` / `test_time_ms` | Test backend median runtime (ms). |
| `Speedup` / `speedup` | `baseline_time / test_time`, suffixed `x`. |
| `SQNR (dB)` / `sqnr_db` | Signal-to-quantization-noise ratio of test vs baseline output (dB). |

### eval_llama3_model.py

Llama-3 eval: perplexity + per-seq runtime. No structured table — free-form metrics plus a
manual runtime table.

| Metric / column | Description |
| --- | --- |
| `<baseline> perplexity` | Baseline WikiText-2 word perplexity. |
| `<test> perplexity` | Test backend perplexity. |
| `Perplexity delta` | Test − baseline perplexity. |
| `SeqLen` | Sequence length. |
| `<baseline> (ms)` | Baseline forward median latency (ms). |
| `<test> (ms)` | Test forward median latency (ms). |
| `Speedup` | `baseline_ms / test_ms`, `x`. |

### eval_flux_model.py

Flux image-gen eval: LPIPS quality + timing. No structured table — free-form metrics and a
returned dict.

| Metric | Description |
| --- | --- |
| `Mean` (LPIPS) | Mean LPIPS perceptual distance (baseline vs test images). |
| `Std Dev` | Std dev of LPIPS values. |
| `Min` / `Max` | Min / max LPIPS values. |
| `Avg <baseline> time` | Mean baseline generation time per image (ms). |
| `Avg <test> time` | Mean test generation time per image (ms). |
| `Speedup` | Baseline / test time, `x`. |

---

## quantization/

### parse_log.py

Parses a quantization run log into a comparison table (tabulate) and CSV. Columns (table label /
CSV field):

| Column | Description |
| --- | --- |
| `Recipe` / `recipe` | Quantization recipe for the row (e.g. `None` baseline, `float8_rowwise`). |
| `Checkpoint (GB)` / `checkpoint_size_gb` | On-disk size of the saved quantized checkpoint (GB). |
| `Wikitext Perplexity` / `wikitext_word_perplexity` | WikiText word perplexity (lower is better). |
| `Winogrande Acc` / `winogrande_acc` | Winogrande accuracy (higher is better). |
| `Winogrande Stderr` / `winogrande_acc_stderr` | Std error of Winogrande accuracy. |
| `Prefill tok/s` / `prefill_total_tokens_per_sec` | vLLM prefill throughput (tokens/sec). |
| `Decode tok/s` / `decode_total_tokens_per_sec` | vLLM decode throughput (tokens/sec). |
| `Speedup Prefill` / `speedup_prefill` | Prefill throughput vs the `None` baseline (ratio). |
| `Speedup Decode` / `speedup_decode` | Decode throughput vs the `None` baseline (ratio). |

### create_quantized_model.py and calibration_based/create_quantized_model.py

Save a quantized model. No table — print free-form, the only metric being `Size: <…> GB`
(saved-model directory size).

### eval_accuracy_and_perf_of_flux.py

Quantized-Flux accuracy/perf eval; writes long-format key/value CSVs with header
`metric,value`. The `metric` rows depend on the mode:

| `metric` row | Description |
| --- | --- |
| `mode` | Run mode (`accuracy`, `performance_hp`, `performance_quant`). |
| `gpu_rank` / `world_size` | Producing rank and total processes. |
| `total_linear_layers_quantized` | Count of quantized Linear layers (accuracy/perf-quant). |
| `prompts_tested` | Number of prompts evaluated (accuracy). |
| `average_lpips` / `max_lpips` / `min_lpips` | LPIPS distance stats (accuracy / aggregated). |
| `lpips_prompt_<i>` | Per-prompt LPIPS value (accuracy / aggregated). |
| `average_baseline_time` / `average_quantized_time` | Mean generation time, baseline vs quantized (s). |
| `num_iterations` / `batch_size` | Timing config (performance modes). |
| `average_time` | Mean generation time (s) for the timed mode. |
| `time_iter_<i>` | Per-iteration time (s). |
| `num_gpus` / `total_prompts` | Aggregation metadata (aggregated CSV). |

`utils.py` files in `quantization/` and `quantization/calibration_based/` are helper modules
(recipe→config maps, directory-size helper); they define no output columns.

---

## quantized_training/

### benchmark_int8mm.py

int8 matmul speedups (cuBLAS `_int_mm` and Triton int8-dequant) vs bf16, over Llama-8B shapes.
Builds a DataFrame printed as markdown.

| Header | Description |
| --- | --- |
| `M` / `N` / `K` | Problem dimensions. |
| `CuBLAS INT8 speedup` | `bf16_time / int8_time` (cuBLAS `_int_mm`). |
| `Triton INT8 dequant speedup` | `bf16_time / int8_dequant_time` (Triton fused dequant). |

### pretrain_llama2.py

Llama-2 pretraining benchmark. No results table — logs metrics to Weights & Biases:

| Metric | Description |
| --- | --- |
| `loss` | Training cross-entropy loss for the step. |
| `lr` | Current optimizer learning rate. |
| `max_memory_allocated` | Peak CUDA memory allocated (GB). |
| `max_memory_reserved` | Peak CUDA memory reserved (GB). |
| `tokens_per_second` | Training throughput (tokens/sec). |
