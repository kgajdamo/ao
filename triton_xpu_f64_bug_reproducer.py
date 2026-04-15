"""
Reproducer: Triton XPU compiler incorrectly elides f32→f64→f32 round-trip.

Expected: both kernels produce the same result as PyTorch f64 reference.
Actual (on XPU): the kernel WITHOUT device_print matches f32 precision,
                 the kernel WITH device_print matches f64 precision.
"""

import torch
import triton
import triton.language as tl

NUMERATOR = 448.0  # e.g. torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _scale_no_print(input_ptr, output_ptr, NUMERATOR: tl.constexpr):
    """f32 → f64 division → f32, no side-effects. Compiler may elide f64."""
    idx = tl.program_id(0)
    val = tl.load(input_ptr + idx)  # f32
    # result = NUMERATOR / val.to(tl.float64)  # should be f64 division
    result = tl.math.div_rn(NUMERATOR, val)
    tl.store(output_ptr + idx, result.to(tl.float32))


@triton.jit
def _scale_with_print(input_ptr, output_ptr, NUMERATOR: tl.constexpr):
    """f32 → f64 division → f32, with device_print forcing f64 materialization."""
    idx = tl.program_id(0)
    val = tl.load(input_ptr + idx)  # f32
    result = NUMERATOR / val.to(tl.float64)  # should be f64 division
    tl.device_print("result", result)  # side-effect prevents elision
    tl.store(output_ptr + idx, result.to(tl.float32))


@triton.jit
def _scale_f32_only(input_ptr, output_ptr, NUMERATOR: tl.constexpr):
    """Pure f32 division (baseline). No f64 upcast at all."""
    idx = tl.program_id(0)
    val = tl.load(input_ptr + idx)  # f32
    result = NUMERATOR / val  # f32 division
    tl.store(output_ptr + idx, result.to(tl.float32))


def main():
    device = "xpu"
    print(f"Device: {device}")

    # Values chosen to produce visible f32 vs f64 precision differences.
    # 3.84375 is a realistic amax from bf16 MoE weight tensors.
    test_values = torch.tensor(
        [3.84375, 1.23456789, 0.00137, 255.5, 7.77777],
        dtype=torch.float32,
        device=device,
    )

    n = test_values.shape[0]
    out_no_print = torch.empty(n, dtype=torch.float32, device=device)
    out_with_print = torch.empty(n, dtype=torch.float32, device=device)
    out_f32_only = torch.empty(n, dtype=torch.float32, device=device)

    grid = (n,)

    # Run all three kernels
    _scale_no_print[grid](test_values, out_no_print, NUMERATOR=NUMERATOR)
    # Note: the kernel below will emit device_print output (set TRITON_DEBUG=1 to see it)
    _scale_with_print[grid](test_values, out_with_print, NUMERATOR=NUMERATOR)
    _scale_f32_only[grid](test_values, out_f32_only, NUMERATOR=NUMERATOR)

    # PyTorch references
    ref_f64 = (NUMERATOR / test_values.to(torch.float64)).to(torch.float32)
    ref_f32 = (NUMERATOR / test_values)

    print()
    print(f"{'value':>14s}  {'no_print':>22s}  {'with_print':>22s}  "
          f"{'f32_only':>22s}  {'pytorch_f64':>22s}  {'pytorch_f32':>22s}")
    print("-" * 130)

    any_mismatch = False
    for i in range(n):
        v = test_values[i].item()
        a = out_no_print[i].item()
        b = out_with_print[i].item()
        c = out_f32_only[i].item()
        d = ref_f64[i].item()
        e = ref_f32[i].item()

        # Flag if no_print matches f32 instead of f64
        match_f32 = "== f32" if a == e else ""
        match_f64 = "== f64" if a == d else ""
        flag = match_f64 if match_f64 else match_f32
        if a != d:
            any_mismatch = True

        print(f"{v:14.7f}  {a:22.15f}  {b:22.15f}  "
              f"{c:22.15f}  {d:22.15f}  {e:22.15f}  {flag}")

    print()
    if any_mismatch:
        print("BUG CONFIRMED: 'no_print' kernel does NOT match PyTorch f64 reference.")
        print("The f64 upcast is being elided by the Triton XPU compiler when there")
        print("are no side-effects between the f64 division and the f32 downcast.")
    else:
        print("All results match f64 reference. Bug may not reproduce on this backend.")


if __name__ == "__main__":
    main()
