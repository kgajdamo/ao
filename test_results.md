# MoE Training Test Results

**Tested on PyTorch:** 2.12.0.dev20260406+xpu (ao-kgajdamo venv) and 2.10.0+xpu (ao-2c venv) | **ao main base:** d26bbae1c | **ao commit (4-GPU run):** 7787da618 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices available:** 4

## Issues (Jira format)

||#||Issue||Type||Tests Affected||Number of Tests Affected||Fix / Issue||Tested on PyTorch||ao main base||branch||
|1.|NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device.|torchao|• test_training.py::test_moe_training (recipe_config1/2: auto/emulated+False+target_fqns0/2) \\ • test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) \\ • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu \\ • ep/test_integration.py::test_full_pipeline_device_type_xpu \\ • test_distributed.py::test_moe_training_parallel (recipe_config0 all 10, recipe_config1 all 10) \\ • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all 128 configs)|31|The op is responsible for preparing data for the scaled_grouped_mm kernel. scaled_grouped_mm should be implemented first at the lower levels. The issue to enable scaled_grouped_mm is open: torch-xpu-ops@3060|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|2.|RuntimeError: Invalid scaling configuration. XPU _scaled_mm does not support MX blockwise (float8_e8m0fnu) scales.|torch-xpu-ops / aten|• test_training.py::test_moe_training (recipe_config1/2: auto/emulated+True+target_fqns0/1/2, auto/emulated+False+target_fqns1) \\ • test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (CUDA+CUTEDSL, MXFP8_RCEIL, all num_groups, matmul/linear/mm-None)|30|Add blockwise FP8 support for scaled_mm_*: [pytorch@176064|https://github.com/pytorch/pytorch/pull/176064], [pytorch@176043|https://github.com/pytorch/pytorch/pull/176043], [pytorch@173630|https://github.com/pytorch/pytorch/pull/173630]|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|3.|Missing xpu kernel: mxfp8_quantize_cuda_3d|torchao|• test_kernels.py::test_cuda_mx_dim1_3d_numerics (all 200 parametrizations)|200|mxfp8_quantize_cuda_3d torchao implementation required for xpu|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|4.|Missing xpu kernel: mxfp8_quantize_2d_1x32_cutedsl|torchao|• test_kernels.py::test_cuda_mx_dim0_2d_numerics (36)|36|mxfp8_quantize_2d_1x32_cutedsl torchao implementation required for xpu|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|5.|Missing xpu kernel: mxfp8_quantize_2d_32x1_cutedsl|torchao|• test_kernels.py::test_cuda_mx_dim1_2d_numerics_32x1 (96)|96|mxfp8_quantize_2d_32x1_cutedsl torchao implementation required for xpu|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|6.|NotImplementedError: The operator 'torchao::fused_pad_token_groups' / 'torchao::fused_unpad_token_groups' is not currently implemented for the XPU device.|torchao|• test_kernels.py::test_cuda_fused_pad_token_groups (16) \\ • test_kernels.py::test_cuda_fused_unpad_token_groups (12)|28|fused_pad_token_groups and fused_unpad_token_groups implementation required for xpu|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|7.|AssertionError: fp8 data not equal|numerical accuracy|• test_kernels.py::test_row_major_with_jagged_rowwise_scales (2 parametrizations)|2|Hardcoded incorrect value of stride output rowed caused the results mismatch. Fixed locally by: STRIDE_OUTPUT_ROW=1 -> STRIDE_OUTPUT_ROW=output_buffer.stride(0)|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|8.|tl.exp2() not supported for the XPU device|torchao (test)|test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic (1)|1|Fixed by replacing tl.exp2() with libdevice.exp2()|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|9.|RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM|torch|• mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest \\ • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest|2|XPU_SHMEM backend not registered in PyTorch|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|10.|AssertionError: SQNR must be >= 26.5, got 23.625 / 25.125 (output SQNR too low)|numerical accuracy|• test_distributed.py::test_moe_training_parallel (recipe_config2-False: expert_parallel, expert_tensor_parallel, fsdp)|3|MXFP8 emulated mode output SQNR below threshold on XPU with EP/ETP/FSDP parallelization. tensor_parallel passes.|2.12.0.dev20260406+xpu|7787da618|kgajdamo/xpu-moe-debug|
|11.|Fatal Python error: Aborted (SIGABRT) during fsdp_tp test|crash / OOM|• test_distributed.py::test_moe_training_parallel (recipe_config2-False-fsdp_tp)|1|Process aborts with SIGABRT in generate_permute_indices during reference model forward. Likely OOM.|2.12.0.dev20260406+xpu|7787da618|kgajdamo/xpu-moe-debug|

## Issues (Markdown format)

| # | Issue | Type | Tests Affected | Number of Tests Affected | Fix / Issue | Tested on PyTorch | ao main base | branch |
|---|---|---|---|---|---|---|---|---|
| 1. | NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device. | torchao | • test_training.py::test_moe_training (recipe_config1/2: auto/emulated+False+target_fqns0/2) <br> • test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) <br> • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu <br> • ep/test_integration.py::test_full_pipeline_device_type_xpu <br> • test_distributed.py::test_moe_training_parallel (recipe_config0 all 10, recipe_config1 all 10) <br> • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all 128 configs) | 31 | The op is responsible for preparing data for the scaled_grouped_mm kernel. scaled_grouped_mm should be implemented first at the lower levels. The issue to enable scaled_grouped_mm is open: torch-xpu-ops@3060 | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 2. | RuntimeError: Invalid scaling configuration. XPU `_scaled_mm` does not support MX blockwise (`float8_e8m0fnu`) scales. | torch-xpu-ops / aten | • test_training.py::test_moe_training (recipe_config1/2: auto/emulated+True+target_fqns0/1/2, auto/emulated+False+target_fqns1) <br> • test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (CUDA+CUTEDSL, MXFP8_RCEIL, all num_groups, matmul/linear/mm-None) | 30 | Add blockwise FP8 support for scaled_mm_*: [pytorch@176064](https://github.com/pytorch/pytorch/pull/176064), [pytorch@176043](https://github.com/pytorch/pytorch/pull/176043), [pytorch@173630](https://github.com/pytorch/pytorch/pull/173630) | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 3. | Missing xpu kernel: mxfp8_quantize_cuda_3d | torchao | • test_kernels.py::test_cuda_mx_dim1_3d_numerics (all 200 parametrizations) | 200 | mxfp8_quantize_cuda_3d torchao implementation required for xpu | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 4. | Missing xpu kernel: mxfp8_quantize_2d_1x32_cutedsl | torchao | • test_kernels.py::test_cuda_mx_dim0_2d_numerics (36) | 36 | mxfp8_quantize_2d_1x32_cutedsl torchao implementation required for xpu | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 5. | Missing xpu kernel: mxfp8_quantize_2d_32x1_cutedsl | torchao | • test_kernels.py::test_cuda_mx_dim1_2d_numerics_32x1 (96) | 96 | mxfp8_quantize_2d_32x1_cutedsl torchao implementation required for xpu | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 6. | NotImplementedError: The operator 'torchao::fused_pad_token_groups' / 'torchao::fused_unpad_token_groups' is not currently implemented for the XPU device. | torchao | • test_kernels.py::test_cuda_fused_pad_token_groups (16) <br> • test_kernels.py::test_cuda_fused_unpad_token_groups (12) | 28 | fused_pad_token_groups and fused_unpad_token_groups implementation required for xpu | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 7. | AssertionError: fp8 data not equal | numerical accuracy | • test_kernels.py::test_row_major_with_jagged_rowwise_scales (2 parametrizations) | 2 | Hardcoded incorrect value of stride output rowed caused the results mismatch. Fixed locally by: STRIDE_OUTPUT_ROW=1 -> STRIDE_OUTPUT_ROW=output_buffer.stride(0) | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 8. | tl.exp2() not supported for the XPU device | torchao (test) | test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic (1) | 1 | Fixed by replacing tl.exp2() with libdevice.exp2() | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 9. | RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM | torch | • mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest <br> • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest | 2 | XPU_SHMEM backend not registered in PyTorch | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 10. | AssertionError: SQNR must be >= 26.5, got 23.625 / 25.125 (output SQNR too low) | numerical accuracy | • test_distributed.py::test_moe_training_parallel (recipe_config2-False: expert_parallel, expert_tensor_parallel, fsdp) | 3 | MXFP8 emulated mode output SQNR below threshold on XPU with EP/ETP/FSDP parallelization. tensor_parallel passes. | 2.12.0.dev20260406+xpu | 7787da618 | kgajdamo/xpu-moe-debug |
| 11. | Fatal Python error: Aborted (SIGABRT) during fsdp_tp test | crash / OOM | • test_distributed.py::test_moe_training_parallel (recipe_config2-False-fsdp_tp) | 1 | Process aborts with SIGABRT in `generate_permute_indices` during reference model forward. Likely OOM. | 2.12.0.dev20260406+xpu | 7787da618 | kgajdamo/xpu-moe-debug |

## Pass Rate (Jira format)

**Last updated:** 22.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** f810c2c08 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

||TEST NAME||PASS||FAIL / ERROR||SKIP||SKIP (CUDA KNOWN ISSUE)||XFAIL||TOTAL||
|ep/test_a2a_dispatch.py|1|0|0|0|0|1|
|ep/test_compile.py|0|1|0|0|0|1|
|ep/test_integration.py|0|1|0|0|0|1|
|ep/test_kernels.py|9|0|0|0|0|9|
|ep/test_permute.py|1|0|0|0|0|1|
|mxfp8/test_mxfp8_a2a.py|0|2|0|0|0|2|
|test_distributed.py|1|24|5|0|0|30|
|test_fp8_grouped_mm.py|8|0|0|4|0|12|
|test_kernels.py|118|374|0|0|0|492|
|test_mxfp8_grouped_mm.py|17|128|0|0|0|145|
|test_nvfp4_grouped_mm.py|14|0|0|0|0|14|
|test_tensor.py|15|14|8|0|0|37|
|test_training.py|6|24|18|0|0|48|
|*TOTAL*|*190*|*568*|*31*|*4*|*793*|

PASS - The test ran and succeeded. Everything behaved exactly as expected.
FAIL / ERROR - The test ran but failed. Indicates a bug, incorrect logic, or unexpected behavior. Needs investigation and fixing. Passes on cuda, fails on xpu. Reason of failure not yet found or is under investigation.
SKIP - The test was intentionally not executed. Feature not implemented. Missing dependency or environment condition
Platform-specific limitation. The test is not planned to be enabled.
SKIP (CUDA KNOWN ISSUE) - The test was intentionally not executed. Known cuda bug (not hardware limitation).
XFAIL - Passes on cuda, fails on xpu. The issue is known and reported.
TOTAL - sum of all tests.



## Pass Rate (Markdown format)

**Last updated:** 22.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** f810c2c08 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

| TEST NAME | PASS | FAIL / ERROR | SKIP | SKIP (CUDA KNOWN ISSUE) | XFAIL | TOTAL |
|---|---|---|---|---|---|---|
| ep/test_a2a_dispatch.py | 1 | 0 | 0 | 0 | 0 | 1 |
| ep/test_compile.py | 0 | 1 | 0 | 0 | 0 | 1 |
| ep/test_integration.py | 0 | 1 | 0 | 0 | 0 | 1 |
| ep/test_kernels.py | 9 | 0 | 0 | 0 | 0 | 9 |
| ep/test_permute.py | 1 | 0 | 0 | 0 | 0 | 1 |
| mxfp8/test_mxfp8_a2a.py | 0 | 2 | 0 | 0 | 0 | 2 |
| test_distributed.py | 1 | 24 | 5 | 0 | 0 | 30 |
| test_fp8_grouped_mm.py | 8 | 0 | 0 | 4 | 0 | 12 |
| test_kernels.py | 118 | 374 | 0 | 0 | 0 | 492 |
| test_mxfp8_grouped_mm.py | 17 | 128 | 0 | 0 | 0 | 145 |
| test_nvfp4_grouped_mm.py | 14 | 0 | 0 | 0 | 0 | 14 |
| test_tensor.py | 15 | 14 | 8 | 0 | 0 | 37 |
| test_training.py | 6 | 24 | 18 | 0 | 0 | 48 |
| **TOTAL** | **190** | **568** | **31** | **4** | **0** | **793** |

PASS - The test ran and succeeded. Everything behaved exactly as expected.
FAIL / ERROR - The test ran but failed. Indicates a bug, incorrect logic, or unexpected behavior. Needs investigation and fixing. Passes on cuda, fails on xpu. Reason of failure not yet found or is under investigation.
SKIP - The test was intentionally not executed. Feature not implemented. Missing dependency or environment condition
Platform-specific limitation. The test is not planned to be enabled.
SKIP (CUDA KNOWN ISSUE) - The test was intentionally not executed. Known cuda bug (not hardware limitation).
XFAIL - Passes on cuda, fails on xpu. The issue is known and reported.
TOTAL - sum of all tests.

## Compatibility with CUDA CI (Jira format)

**Last updated:** 22.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** f810c2c08 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

||TEST NAME||XPU PASS||CUDA PASS||XPU FAIL / ERROR||CUDA FAIL / ERROR||XPU SKIP||CUDA SKIP||XPU TOTAL||CUDA TOTAL||DIFF||
|test_fp8_grouped_mm.py|8|8|0|0|4|4|12|12|THE SAME|
|test_mxfp8_grouped_mm.py|17|65|128|0|0|80|145|145|test_mxfp8_grouped_gemm_with_dq_fwd_bwd - on xpu all test cases hang (128/128 - FAIL), on cuda (80/128 - SKIPPED, 48/128 - PASSED)|
|test_training.py|6|6|24|0|18|42|48|48|PASS - same configs, FAIL on XPU = skip in CUDA CI|
|*TOTAL*|*31*|*79*|*152*|*0*|*22*|*126*|*215*|*215*|



## Compatibility with CUDA CI (Markdown format)

**Last updated:** 22.04.2026 | **PyTorch:** 2.12.0.dev20260406+xpu | **ao commit:** f810c2c08 | **branch:** kgajdamo/xpu-moe-debug | **XPU devices:** 4

| TEST NAME | XPU PASS | CUDA PASS | XPU FAIL / ERROR | CUDA FAIL / ERROR | XPU SKIP | CUDA SKIP | XPU TOTAL | CUDA TOTAL | DIFF |
|---|---|---|---|---|---|---|---|---|---|
| test_fp8_grouped_mm.py | 8 | 8 | 0 | 0 | 4 | 4 | 12 | 12 | The same results |
| test_mxfp8_grouped_mm.py | 17 | 65 | 128 | 0 | 0 | 80 | 145 | 145 | test_mxfp8_grouped_gemm_with_dq_fwd_bwd - on xpu all test cases hang (128/128 - FAIL), on cuda (80/128 - SKIPPED, 48/128 - PASSED) |
| test_training.py | 6 | 6 | 24 | 0 | 18 | 42 | 48 | 48 | PASS - same configs, FAIL on XPU = skip in CUDA CI |
| **TOTAL** | **31** | **79** | **152** | **0** | **22** | **126** | **215** | **215** | |
