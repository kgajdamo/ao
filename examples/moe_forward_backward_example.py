# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified MoE (Mixture of Experts) layer: forward and backward pass example.

This example demonstrates how grouped GEMMs work in an MoE layer with 8 experts,
showing the exact matrix shapes and operations at each stage.

MoE layer overview:
    1. A router assigns each token to one expert.
    2. Tokens are reordered so that all tokens for expert 0 come first, then
       expert 1, etc. The `offs` tensor records the cumulative boundaries.
    3. Forward pass: output = x @ W^T            (2D x 3D grouped GEMM)
    4. Backward pass:
       - dgrad:  grad_x = grad_output @ W        (2D x 3D grouped GEMM)
       - wgrad:  grad_W = grad_output^T @ x      (2D x 2D grouped GEMM)
"""

import torch

# ============================================================================
# Configuration
# ============================================================================
num_experts = 8
M = 16  # 1024  # total number of tokens (across all experts)
K = 8  # 512  # input hidden dimension
N = 4  # 256  # output hidden dimension (per expert)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print(f"Device: {device}, dtype: {dtype}")
print(f"num_experts={num_experts}, M={M}, K={K}, N={N}")
print()

# ============================================================================
# Step 1: Simulate router token assignment
# ============================================================================
# In a real MoE, a gating network (router) assigns each token to an expert.
# Here we simulate this by randomly assigning tokens, then sorting by expert.
#
# After sorting, `offs` contains the cumulative end-index for each expert's
# token group. For example, if expert 0 gets 130 tokens and expert 1 gets 120:
#   offs = [130, 250, ...]

tokens_per_expert = torch.multinomial(
    torch.ones(num_experts, device=device),
    num_samples=M,
    replacement=True,
).bincount(minlength=num_experts)

# Ensure all experts get at least some tokens
tokens_per_expert = torch.tensor([3, 1, 1, 3, 1, 2, 1, 4])
# Adjust to exactly M tokens total
diff = M - tokens_per_expert.sum().item()
tokens_per_expert[0] += diff

offs = tokens_per_expert.cumsum(dim=0).to(torch.int32)

print("Step 1: Router token assignment")
print(f"  Tokens per expert: {tokens_per_expert.tolist()}")
print(f"  offs (cumulative):  {offs.tolist()}")
print(f"  Total tokens:       {offs[-1].item()}")
print()

# ============================================================================
# Step 2: Create input data and expert weights
# ============================================================================
# x:  (M, K) — all tokens stacked, reordered by expert assignment
# W:  (num_experts, N, K) — each expert has its own weight matrix
# W^T (used in forward): each expert's (K, N) slice

x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
W = torch.randn(num_experts, N, K, dtype=dtype, device=device)
W_t = W.transpose(-2, -1).contiguous().requires_grad_(True)  # (num_experts, K, N)

print("Step 2: Input data and weights")
print(f"  x (activations):    {tuple(x.shape)}  — (M, K)")
print(f"  W (expert weights): {tuple(W.shape)}  — (num_experts, N, K)")
print(f"  W^T (transposed):   {tuple(W_t.shape)} — (num_experts, K, N)")
print()

# ============================================================================
# Step 3: Forward pass — 2D x 3D grouped GEMM
# ============================================================================
# For each expert i, with token range [start_i, end_i):
#   output[start_i:end_i] = x[start_i:end_i] @ W_t[i]
#                         = (tokens_i, K) @ (K, N) -> (tokens_i, N)
#
# torch._grouped_mm handles this efficiently in a single fused kernel.
# The `offs` tensor tells it where each expert's token group starts/ends.

print("Step 3: Forward pass (2D x 3D grouped GEMM)")
print(f"  x @ W^T: ({M}, {K}) @ ({num_experts}, {K}, {N}) -> ({M}, {N})")
print("  Conceptually, for each expert i:")
print("    output[start_i:end_i] = x[start_i:end_i] @ W_t[i]")
print()

chunks = []
start = 0
for i in range(num_experts):
    end = offs[i].item()
    chunks.append((x[start:end] @ W_t[i]).to(dtype))
    start = end
output = torch.cat(chunks, dim=0)

print(f"  output shape: {tuple(output.shape)}")
print()

# Verify by computing manually per expert
start = 0
manual_output = torch.zeros_like(output)
for i in range(num_experts):
    end = offs[i].item()
    manual_output[start:end] = x[start:end].detach() @ W_t[i].detach()
    start = end

max_diff = (output.detach() - manual_output).abs().max().item()
print(f"  Manual vs grouped_mm max diff: {max_diff:.2e} (should be ~0)")
print()

# ============================================================================
# Step 4: Backward pass — compute gradients
# ============================================================================
# Simulate a loss and backpropagate.
# MSE loss against a target of ones (simplified).

labels = torch.ones_like(output)
loss = torch.nn.functional.mse_loss(output, labels)
loss.backward()

print("Step 4: Backward pass")
print(f"  Loss: {loss.item():.4f}")
print()

# ============================================================================
# Step 4a: dgrad — input gradient (2D x 3D grouped GEMM)
# ============================================================================
# For each expert i:
#   grad_x[start_i:end_i] = grad_output[start_i:end_i] @ W[i]
#                          = (tokens_i, N) @ (N, K) -> (tokens_i, K)
#
# This is another 2D x 3D grouped GEMM, same shape pattern as forward.

print("Step 4a: dgrad — input gradient (2D x 3D grouped GEMM)")
print(f"  grad_output @ W: ({M}, {N}) @ ({num_experts}, {N}, {K}) -> ({M}, {K})")
print(f"  grad_x shape: {tuple(x.grad.shape)}")
print()

# Verify manually
grad_output = 2.0 * (output.detach() - labels) / labels.numel()  # MSE grad
start = 0
manual_grad_x = torch.zeros_like(x.detach())
for i in range(num_experts):
    end = offs[i].item()
    # grad_x = grad_output @ W  (note: W[i] is (N, K), no transpose needed)
    manual_grad_x[start:end] = grad_output[start:end] @ W[i].detach()
    start = end

dgrad_diff = (x.grad.detach() - manual_grad_x).abs().max().item()
print(f"  Manual vs autograd dgrad max diff: {dgrad_diff:.2e} (should be ~0)")
print()

# ============================================================================
# Step 4b: wgrad — weight gradient (2D x 2D grouped GEMM)
# ============================================================================
# For each expert i:
#   grad_W_t[i] = grad_output[start_i:end_i]^T @ x[start_i:end_i]
#               = (N, tokens_i) @ (tokens_i, K) -> (N, K)
#
# THIS is the 2D x 2D grouped GEMM:
#   - grad_output^T is (N, M) — 2D, with M partitioned by offs
#   - x is (M, K)             — 2D, with M partitioned by offs
#   - Both operands share the M (token) dimension, which is the "grouped" dim.
#
# Unlike forward/dgrad where one operand is 3D (per-expert weights),
# here BOTH operands are flat 2D tensors that must be sliced by offs.
#
# Note: since W_t has shape (num_experts, K, N), its gradient also has
# shape (K, N) per expert. The underlying math is:
#   grad_W_t[i] = x[start:end]^T @ grad_output[start:end]
#               = (K, tokens_i) @ (tokens_i, N) -> (K, N)
#
# Equivalently, for the original W (num_experts, N, K):
#   grad_W[i] = grad_output[start:end]^T @ x[start:end]
#             = (N, tokens_i) @ (tokens_i, K) -> (N, K)
# This second form is the 2D x 2D grouped GEMM tested by
# test_emulate_mxfp8_grouped_gemm_2d_2d.

print("Step 4b: wgrad — weight gradient (2D x 2D grouped GEMM)")
print("  For W_t (K, N): x^T @ grad_output per expert")
print(f"    ({K}, {M}) @ ({M}, {N}) -> ({num_experts}, {K}, {N})")
print("  Equivalently for W (N, K): grad_output^T @ x per expert")
print(f"    ({N}, {M}) @ ({M}, {K}) -> ({num_experts}, {N}, {K})")
print(f"  grad_W_t shape: {tuple(W_t.grad.shape)}")
print()

# Verify manually
start = 0
manual_grad_W_t = torch.zeros_like(W_t.detach())
for i in range(num_experts):
    end = offs[i].item()
    # grad_W_t[i] = x[start:end]^T @ grad_output[start:end]
    #             = (K, tokens_i) @ (tokens_i, N) -> (K, N)
    manual_grad_W_t[i] = x[start:end].detach().t() @ grad_output[start:end]
    start = end

wgrad_diff = (W_t.grad.detach() - manual_grad_W_t).abs().max().item()
print(f"  Manual vs autograd wgrad max diff: {wgrad_diff:.2e} (should be ~0)")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: Grouped GEMMs in MoE forward + backward")
print("=" * 70)
print()
print("  FORWARD (2D x 3D):")
print("    x @ W^T per expert")
print(f"    ({M}, {K}) @ ({num_experts}, {K}, {N}) -> ({M}, {N})")
print("    LHS is 2D (tokens), RHS is 3D (per-expert weights)")
print()
print("  BACKWARD — dgrad (2D x 3D):")
print("    grad_output @ W per expert")
print(f"    ({M}, {N}) @ ({num_experts}, {N}, {K}) -> ({M}, {K})")
print("    Same shape pattern as forward")
print()
print("  BACKWARD — wgrad (2D x 2D):")
print(
    f"    For W (N,K):  grad_output^T @ x  = ({N}, {M}) @ ({M}, {K}) -> ({num_experts}, {N}, {K})"
)
print(
    f"    For W^T (K,N): x^T @ grad_output  = ({K}, {M}) @ ({M}, {N}) -> ({num_experts}, {K}, {N})"
)
print("    BOTH operands are 2D, partitioned along the shared M dim by offs")
print("    This is the operation tested by test_emulate_mxfp8_grouped_gemm_2d_2d")
print()
print("  offs partitions the token dimension M across experts:")
print(f"    {offs.tolist()}")
start = 0
for i in range(num_experts):
    end = offs[i].item()
    print(f"    Expert {i}: tokens [{start:>4}, {end:>4}) — {end - start} tokens")
    start = end
