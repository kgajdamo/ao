# MoE Training Test Results

**Tested on PyTorch:** 2.12.0.dev20260406+xpu (ao-kgajdamo venv) and 2.10.0+xpu (ao-2c venv) | **ao main base:** d26bbae1c | **branch:** kgajdamo/xpu-moe-debug | **XPU devices available:** 2

## Issues (Jira format)

||#||Issue||Type||Tests Affected||Number of Tests Affected||Fix / Issue||Tested on PyTorch||ao main base||branch||
|1.|NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device.|torchao|• test_training.py::test_moe_training (recipe_config1: auto+target_fqns0, auto+target_fqns2, emulated+target_fqns0, emulated+target_fqns2) \\ • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu \\ • ep/test_integration.py::test_full_pipeline_device_type_xpu|6|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|2.|RuntimeError: Invalid scaling configuration.|torch-xpu-ops / aten|• test_training.py::test_moe_training (recipe_config1: auto+target_fqns1, auto+True+target_fqns0/1/2, emulated+target_fqns1, emulated+True+target_fqns0/1/2)|8|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|3.|RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)|OOM / driver|• test_training.py::test_moe_training (recipe_config2: emulated variants, recipe_config3: all variants)|14|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|4.|torch.OutOfMemoryError: XPU out of memory.|OOM|• test_training.py::test_moe_training (recipe_config2: auto+target_fqns1/2, auto+True+target_fqns0/1)|4|TBD|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|5.|AssertionError: Torch not compiled with CUDA enabled — device="cuda" hardcoded in test|torchao (test)|• test_kernels.py::test_triton_fp8_rowwise_2d_scale_and_cast|6|Need to add XPU device support to the test (hardcoded device = "cuda" on line 623)|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|6.|RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM / device index out of range (test requires world_size=4, only 2 XPU devices available)|torch / HW config|• mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest \\ • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest|2|XPU_SHMEM backend not registered in PyTorch. Also requires 4 XPU devices (only 2 available on this machine).|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|7.|AssertionError: This test requires world_size=4, but got world_size=2|HW config|• test_distributed.py (all 30 parametrized tests)|30|Machine only has 2 XPU devices; test requires 4. Cannot run on this hardware.|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|8.|Skipped: XPU support not yet available|Expected skip|• test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all configs) \\ • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic|129|Intentional skip via @skip_if_xpu decorator|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|9.|Skipped: MXFP8 requires CUDA SM 10.x / cutedsl kernels not available / CUDA kernel requires sm_100|Expected skip|• test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) \\ • test_kernels.py::test_cuda_mx_dim1_3d_numerics (200) \\ • test_kernels.py::test_cuda_fused_pad_token_groups (16) \\ • test_kernels.py::test_cuda_fused_unpad_token_groups (12) \\ • test_kernels.py::test_cuda_mxfp8_quantize_cutedsl_3d (36)|275|Expected — CUDA SM 10.x kernels not applicable on XPU|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|10.|Skipped: FP8 rowwise doesn't support per group token padding yet / compile+EMULATED not supported / FakeTensor issue #4048|Expected skip|• test_training.py::test_moe_training (recipe_config0 variants, compile=True+EMULATED, compile=True+auto)|18|Expected skips (feature not implemented / known issue)|2.12.0.dev20260406+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|11.|Skipped: XPU support not yet available|Expected skip|• test_fp8_grouped_mm.py::test_fp8_rowwise_scaled_grouped_mm|4|Intentional skip|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|
|12.|Skipped: mm doesn't support batching|Expected skip|• test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (num_groups+mm combos)|2|Expected skip|2.10.0+xpu|d26bbae1c|kgajdamo/xpu-moe-debug|

## Issues (Markdown format)

| # | Issue | Type | Tests Affected | Number of Tests Affected | Fix / Issue | Tested on PyTorch | ao main base | branch |
|---|---|---|---|---|---|---|---|---|
| 1. | NotImplementedError: The operator 'torchao::mx_block_rearrange_2d_M_groups' is not currently implemented for the XPU device. | torchao | • test_training.py::test_moe_training (recipe_config1: auto+target_fqns0, auto+target_fqns2, emulated+target_fqns0, emulated+target_fqns2) <br> • ep/test_compile.py::test_full_pipeline_compiled_device_type_xpu <br> • ep/test_integration.py::test_full_pipeline_device_type_xpu | 6 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 2. | RuntimeError: Invalid scaling configuration. | torch-xpu-ops / aten | • test_training.py::test_moe_training (recipe_config1: auto+target_fqns1, auto+True+target_fqns0/1/2, emulated+target_fqns1, emulated+True+target_fqns0/1/2) | 8 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 3. | RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) | OOM / driver | • test_training.py::test_moe_training (recipe_config2: emulated variants, recipe_config3: all variants) | 14 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 4. | torch.OutOfMemoryError: XPU out of memory. | OOM | • test_training.py::test_moe_training (recipe_config2: auto+target_fqns1/2, auto+True+target_fqns0/1) | 4 | TBD | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 5. | AssertionError: Torch not compiled with CUDA enabled — `device="cuda"` hardcoded in test | torchao (test) | • test_kernels.py::test_triton_fp8_rowwise_2d_scale_and_cast | 6 | Need to add XPU device support to the test (hardcoded `device = "cuda"` on line 623) | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 6. | RuntimeError: SymmetricMemory does not find allocation backend XPU_SHMEM / device index out of range (test requires world_size=4, only 2 XPU devices available) | torch / HW config | • mxfp8/test_mxfp8_a2a.py::MXFP8OnDeviceAllToAllVTest <br> • mxfp8/test_mxfp8_a2a.py::ToMXFP8AllToAllVDequantTest | 2 | XPU_SHMEM backend not registered in PyTorch. Also requires 4 XPU devices (only 2 available on this machine). | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 7. | AssertionError: This test requires world_size=4, but got world_size=2 | HW config | • test_distributed.py (all 30 parametrized tests) | 30 | Machine only has 2 XPU devices; test requires 4. Cannot run on this hardware. | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 8. | Skipped: XPU support not yet available | Expected skip | • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_with_dq_fwd_bwd (all configs) <br> • test_mxfp8_grouped_mm.py::test_mxfp8_grouped_gemm_from_qdata_and_scales_matches_dynamic | 129 | Intentional skip via `@skip_if_xpu` decorator | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 9. | Skipped: MXFP8 requires CUDA SM 10.x / cutedsl kernels not available / CUDA kernel requires sm_100 | Expected skip | • test_kernels.py::test_cuda_mx_block_rearrange_2d_M_groups (11) <br> • test_kernels.py::test_cuda_mx_dim1_3d_numerics (200) <br> • test_kernels.py::test_cuda_fused_pad_token_groups (16) <br> • test_kernels.py::test_cuda_fused_unpad_token_groups (12) <br> • test_kernels.py::test_cuda_mxfp8_quantize_cutedsl_3d (36) | 275 | Expected — CUDA SM 10.x kernels not applicable on XPU | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 10. | Skipped: FP8 rowwise doesn't support per group token padding yet / compile+EMULATED not supported / FakeTensor issue #4048 | Expected skip | • test_training.py::test_moe_training (recipe_config0 variants, compile=True+EMULATED, compile=True+auto) | 18 | Expected skips (feature not implemented / known issue) | 2.12.0.dev20260406+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 11. | Skipped: XPU support not yet available | Expected skip | • test_fp8_grouped_mm.py::test_fp8_rowwise_scaled_grouped_mm | 4 | Intentional skip | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |
| 12. | Skipped: mm doesn't support batching | Expected skip | • test_tensor.py::test_mxfp8_training_tensor_ops_fwd_bwd (num_groups+mm combos) | 2 | Expected skip | 2.10.0+xpu | d26bbae1c | kgajdamo/xpu-moe-debug |

## Per-file Summary

| File | Passed | Failed | Skipped | Errors |
|---|---|---|---|---|
| test_training.py | 0 | 30 | 18 | 0 |
| test_fp8_grouped_mm.py | 8 | 0 | 4 | 0 |
| test_mxfp8_grouped_mm.py | 16 | 0 | 129 | 0 |
| test_kernels.py | 112 | 6 | 275 | 0 |
| test_tensor.py | 8 | 0 | 2 | 0 |
| ep/test_compile.py | 0 | 1 | 0 | 0 |
| ep/test_integration.py | 0 | 1 | 0 | 0 |
| ep/test_a2a_dispatch.py | 1 | 0 | 0 | 0 |
| ep/test_kernels.py | 9 | 0 | 0 | 0 |
| ep/test_permute.py | 1 | 0 | 0 | 0 |
| mxfp8/test_mxfp8_a2a.py | 0 | 2 | 0 | 0 |
| test_distributed.py | 0 | 0 | 0 | 30 (fixture fail: need 4 GPUs) |
| **Total** | **155** | **40** | **428** | **30** |
