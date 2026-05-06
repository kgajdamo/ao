"""
Simple unit test to debug and understand to_mx() behavior.
"""

import torch
from torchao.prototype.mx_formats.mx_tensor import to_mx, MXTensor
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.constants import SUPPORTED_ELEM_DTYPES


def test_basic_to_mx_fp8():
    """Basic: convert a small bf16 tensor to MXFP8."""
    data = torch.randn(2, 64, dtype=torch.bfloat16)
    scale, data_lp = to_mx(data, torch.float8_e4m3fn, block_size=32)

    print("=== Basic MXFP8 ===")
    print(f"Input shape: {data.shape}, dtype: {data.dtype}")
    print(f"Scale shape: {scale.shape}, dtype: {scale.dtype}")
    print(f"Data_lp shape: {data_lp.shape}, dtype: {data_lp.dtype}")
    print(f"Scale values: {scale}")
    print(f"First block input:  {data[0, :32]}")
    print(f"First block output: {data_lp[0, :32]}")
    print()


def test_scaling_modes():
    """Compare FLOOR vs RCEIL scaling modes."""
    data = torch.tensor([[1.0, 2.0, 0.5, 0.25] * 8], dtype=torch.float32)

    for mode in [ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL]:
        scale, data_lp = to_mx(data, torch.float8_e4m3fn, block_size=32, scaling_mode=mode)
        print(f"=== Scaling mode: {mode.name} ===")
        print(f"Scale (e8m0 biased): {scale.view(torch.uint8)}")
        print(f"Data_lp[:8]: {data_lp[0, :8]}")
        print()


def test_roundtrip():
    """Quantize then dequantize – check reconstruction error."""
    data = torch.randn(4, 128, dtype=torch.bfloat16)
    mx = MXTensor.to_mx(data, torch.float8_e4m3fn, block_size=32)
    recon = mx.dequantize(torch.bfloat16)

    abs_err = (data.float() - recon.float()).abs()
    rel_err = abs_err / (data.float().abs() + 1e-8)

    print("=== Roundtrip (MXFP8 e4m3) ===")
    print(f"Max abs error: {abs_err.max().item():.6f}")
    print(f"Mean abs error: {abs_err.mean().item():.6f}")
    print(f"Mean rel error: {rel_err.mean().item():.4f}")
    print(f"SQNR: {(data.float().norm() / (data.float() - recon.float()).norm()).item():.2f}")
    print()


def test_nan_and_zero_blocks():
    """Check how to_mx handles NaN and all-zero blocks."""
    # All zeros
    zeros = torch.zeros(1, 32, dtype=torch.float32)
    scale_z, data_z = to_mx(zeros, torch.float8_e4m3fn, block_size=32)
    print("=== All-zero block ===")
    print(f"Scale (uint8): {scale_z.view(torch.uint8)}")
    print(f"Data_lp: {data_z}")
    print()

    # Block with NaN
    nan_data = torch.ones(1, 32, dtype=torch.float32)
    nan_data[0, 0] = float("nan")
    scale_n, data_n = to_mx(nan_data, torch.float8_e4m3fn, block_size=32)
    print("=== Block with NaN ===")
    print(f"Scale (uint8): {scale_n.view(torch.uint8)} (255 = NaN sentinel)")
    print(f"Data_lp[:4]: {data_n[0, :4]}")
    print()


def test_different_elem_dtypes():
    """Try all supported elem_dtypes."""
    data = torch.randn(1, 32, dtype=torch.float32)
    for dtype in SUPPORTED_ELEM_DTYPES:
        try:
            scale, data_lp = to_mx(data, dtype, block_size=32)
            print(f"{str(dtype):30s} -> scale {scale.shape}, data_lp {data_lp.shape} ({data_lp.dtype})")
        except Exception as e:
            print(f"{str(dtype):30s} -> ERROR: {e}")
    print()


if __name__ == "__main__":
    test_basic_to_mx_fp8()
    test_scaling_modes()
    test_roundtrip()
    test_nan_and_zero_blocks()
    test_different_elem_dtypes()
