# MoE Training Test Results

**Tested on PyTorch:** 2.12.0.dev20260406+xpu (ao-kgajdamo venv) and 2.10.0+xpu (ao-2c venv) | **ao main base:** d26bbae1c | **ao commit (4-GPU run):** 7787da618 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices available:** 4

## Issues (Jira format)

||#||Issue||Type||Tests Affected||Number of Tests Affected||Fix / Issue||Tested on PyTorch||ao main base||branch||
|1.|NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device.|torchao|• test_training.py::test_moe_training (recipe_config1: auto+target_fqns0, auto+target_fqns2, emulated+target_fqns0, emulated+target_fqns2) \\ • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu \\ • ep/test_integration.py::test_full_pipeline_device_type_xpu \\ • test_distributed.py::test_moe_training_parallel (recipe_config0 all 10, recipe_config1 all 10)|26|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|2.|RuntimeError: Invalid scaling configuration.|torch-xpu-ops / aten|• test_training.py::test_moe_training (recipe_config1: auto+target_fqns1, auto+True+target_fqns0/1/2, emulated+target_fqns1, emulated+True+target_fqns0/1/2)|8|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|3.|RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)|OOM / driver|• test_training.py::test_moe_training (recipe_config2: emulated variants, recipe_config3: all variants)|14|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|4.|torch.OutOfMemoryError: XPU out of memory.|OOM|• test_training.py::test_moe_training (recipe_config2: auto+target_fqns1/2, auto+True+target_fqns0/1)|4|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|5.|AssertionError: Torch not compiled with CUDA enabled — device="cuda" hardcoded in test|torchao (test)|• test_kernels.py::test_triton_fp8_rowwise_2d_scale_and_cast|6|Need to add XPU device support to the test (hardcoded device = "cuda" on line 623)|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|6.|RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM|torch|• mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest \\ • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest|2|XPU_SHMEM backend not registered in PyTorch. Confirmed with 4 XPU devices — device count is not the issue.|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|7.|AssertionError: SQNR must be >= 26.5, got 23.625 / 25.125 (output SQNR too low)|numerical accuracy|• test_distributed.py::test_moe_training_parallel (recipe_config2-False: expert_parallel, expert_tensor_parallel, fsdp)|3|MXFP8 emulated mode output SQNR below threshold on XPU with EP/ETP/FSDP parallelization. tensor_parallel passes.|2.12.0.dev20260406+xpu|7787da618|kgajdamo/xpu-moe-debug|
|7a.|Fatal Python error: Aborted (SIGABRT) during fsdp_tp test|crash / OOM|• test_distributed.py::test_moe_training_parallel (recipe_config2-False-fsdp_tp)|1|Process aborts with SIGABRT in generate_permute_indices during reference model forward. Likely OOM.|2.12.0.dev20260406+xpu|7787da618|kgajdamo/xpu-moe-debug|
|8.|Skipped: XPU support not yet available|Expected skip|• test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all configs) \\ • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic|129|Intentional skip via @skip_if_xpu decorator|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|9.|Skipped: MXFP8 requires CUDA SM 10.x / cutedsl kernels not available / CUDA kernel requires sm_100|Expected skip|• test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) \\ • test_kernels.py::test_cuda_mx_dim1_3d_numerics (200) \\ • test_kernels.py::test_cuda_fused_pad_token_groups (16) \\ • test_kernels.py::test_cuda_fused_unpad_token_groups (12) \\ • test_kernels.py::test_cuda_mxfp8_quantize_cutedsl_3d (36)|275|Expected — CUDA SM 10.x kernels not applicable on XPU|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|10.|Skipped: FP8 rowwise doesn't support per group token padding yet / compile+EMULATED not supported / FakeTensor issue #4048|Expected skip|• test_training.py::test_moe_training (recipe_config0 variants, compile=True+EMULATED, compile=True+auto) \\ • test_distributed.py::test_moe_training_parallel (recipe_config2+compile=True, 5 variants)|23|Expected skips (feature not implemented / known issue)|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|11.|Skipped: XPU support not yet available|Expected skip|• test_fp8_grouped_mm.py::test_fp8_rowwise_scaled_grouped_mm|4|Intentional skip|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|12.|Skipped: mm doesn't support batching|Expected skip|• test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (num_groups+mm combos)|2|Expected skip|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|

## Issues (Markdown format)

| # | Issue | Type | Tests Affected | Number of Tests Affected | Fix / Issue | Tested on PyTorch | ao main base | branch |
|---|---|---|---|---|---|---|---|---|
| 1. | NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device. | torchao | • test_training.py::test_moe_training (recipe_config1: auto+target_fqns0, auto+target_fqns2, emulated+target_fqns0, emulated+target_fqns2) <br> • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu <br> • ep/test_integration.py::test_full_pipeline_device_type_xpu <br> • test_distributed.py::test_moe_training_parallel (recipe_config0 all 10, recipe_config1 all 10) | 26 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 2. | RuntimeError: Invalid scaling configuration. | torch-xpu-ops / aten | • test_training.py::test_moe_training (recipe_config1: auto+target_fqns1, auto+True+target_fqns0/1/2, emulated+target_fqns1, emulated+True+target_fqns0/1/2) | 8 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 3. | RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) | OOM / driver | • test_training.py::test_moe_training (recipe_config2: emulated variants, recipe_config3: all variants) | 14 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 4. | torch.OutOfMemoryError: XPU out of memory. | OOM | • test_training.py::test_moe_training (recipe_config2: auto+target_fqns1/2, auto+True+target_fqns0/1) | 4 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 5. | AssertionError: Torch not compiled with CUDA enabled — `device="cuda"` hardcoded in test | torchao (test) | • test_kernels.py::test_triton_fp8_rowwise_2d_scale_and_cast | 6 | Need to add XPU device support to the test (hardcoded `device = "cuda"` on line 623) | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 6. | RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM | torch | • mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest <br> • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest | 2 | XPU_SHMEM backend not registered in PyTorch. Confirmed with 4 XPU devices — device count is not the issue. | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 7. | AssertionError: SQNR must be >= 26.5, got 23.625 / 25.125 (output SQNR too low) | numerical accuracy | • test_distributed.py::test_moe_training_parallel (recipe_config2-False: expert_parallel, expert_tensor_parallel, fsdp) | 3 | MXFP8 emulated mode output SQNR below threshold on XPU with EP/ETP/FSDP parallelization. tensor_parallel passes. | 2.12.0.dev20260406+xpu | 7787da618 | kgajdamo/xpu-moe-debug |
| 7a. | Fatal Python error: Aborted (SIGABRT) during fsdp_tp test | crash / OOM | • test_distributed.py::test_moe_training_parallel (recipe_config2-False-fsdp_tp) | 1 | Process aborts with SIGABRT in `generate_permute_indices` during reference model forward. Likely OOM. | 2.12.0.dev20260406+xpu | 7787da618 | kgajdamo/xpu-moe-debug |
| 8. | Skipped: XPU support not yet available | Expected skip | • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all configs) <br> • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic | 129 | Intentional skip via `@skip_if_xpu` decorator | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 9. | Skipped: MXFP8 requires CUDA SM 10.x / cutedsl kernels not available / CUDA kernel requires sm_100 | Expected skip | • test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) <br> • test_kernels.py::test_cuda_mx_dim1_3d_numerics (200) <br> • test_kernels.py::test_cuda_fused_pad_token_groups (16) <br> • test_kernels.py::test_cuda_fused_unpad_token_groups (12) <br> • test_kernels.py::test_cuda_mxfp8_quantize_cutedsl_3d (36) | 275 | Expected — CUDA SM 10.x kernels not applicable on XPU | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 10. | Skipped: FP8 rowwise doesn't support per group token padding yet / compile+EMULATED not supported / FakeTensor issue #4048 | Expected skip | • test_training.py::test_moe_training (recipe_config0 variants, compile=True+EMULATED, compile=True+auto) <br> • test_distributed.py::test_moe_training_parallel (recipe_config2+compile=True, 5 variants) | 23 | Expected skips (feature not implemented / known issue) | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 11. | Skipped: XPU support not yet available | Expected skip | • test_fp8_grouped_mm.py::test_fp8_rowwise_scaled_grouped_mm | 4 | Intentional skip | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 12. | Skipped: mm doesn't support batching | Expected skip | • test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (num_groups+mm combos) | 2 | Expected skip | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |

## Pass Rate (Jira format)

**Last updated:** 08.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** 7787da618 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

||TEST NAME||PASS||FAIL / ERROR||SKIP||XFAIL||TOTAL||
|ep/test_a2a_dispatch.py|1|0|0|0|1|
|ep/test_compile.py|0|1|0|0|1|
|ep/test_integration.py|0|1|0|0|1|
|ep/test_kernels.py|9|0|0|0|9|
|ep/test_permute.py|1|0|0|0|1|
|mxfp8/test_mxfp8_a2a.py|0|2|0|0|2|
|test_distributed.py|1|24|5|0|30|
|test_fp8_grouped_mm.py|8|0|4|0|12|
|test_kernels.py|118|0|275|0|393|
|test_mxfp8_grouped_mm.py|16|0|129|0|145|
|test_tensor.py|8|0|2|0|10|
|test_training.py|6|24|18|0|48|
|*TOTAL*|*168*|*52*|*433*|*0*|*653*|

## Pass Rate (Markdown format)

**Last updated:** 08.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** 7787da618 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

| TEST NAME | PASS | FAIL / ERROR | SKIP | SKIP (CUDA KNOWN ISSUE) | XFAIL | TOTAL |
|---|---|---|---|---|---|
| ep/test_a2a_dispatch.py | 1 | 0 | 0 | 0 | 0 | 1 |
| ep/test_compile.py | 0 | 1 | 0 | 0 | 0 | 1 |
| ep/test_integration.py | 0 | 1 | 0 | 0 | 0 | 1 |
| ep/test_kernels.py | 9 | 0 | 0 | 0 | 0 | 9 |
| ep/test_permute.py | 1 | 0 | 0 | 0 | 0 | 1 |
| mxfp8/test_mxfp8_a2a.py | 0 | 2 | 0 | 0 | 0 | 2 |
| test_distributed.py | 1 | 24 | 5 | 0 | 0 | 30 |
| test_fp8_grouped_mm.py | 8 | 0 | 0 | 4 | 0 | 12 |
| test_kernels.py | 118 | 0 | 275 | 0 | 393 |
| test_mxfp8_grouped_mm.py | 16 | 0 | 129 | 0 | 145 |
| test_tensor.py | 8 | 0 | 2 | 0 | 10 |
| test_training.py | 6 | 24 | 18 | 0 | 48 |
| **TOTAL** | **168** | **52** | **433** | **0** | **653** |
