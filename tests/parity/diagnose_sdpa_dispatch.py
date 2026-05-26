#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Show which kernel ``F.scaled_dot_product_attention`` actually dispatches
to, on whatever device is available (NPU preferred, then CUDA, then CPU).

Run on the NPU host *without* the parity fp32 patch active::

    PARITY_DISABLE_FP32_ATTN=1 python tests/parity/diagnose_sdpa_dispatch.py

The script prints:

1. version info (torch, torch_npu, transformers)
2. where Llama's SDPA forward lives in this version
3. a TorchDispatchMode trace of a *single* ``scaled_dot_product_attention``
   call with OpenVLA-OFT-ish shapes (B=1, num_heads=32, num_kv_heads=8,
   seq=64, head_dim=128, GQA expanded, additive bf16 causal mask).
   The trace shows every aten / npu / cuda op called, with their actual
   argument shapes/dtypes/scalars -- the line where SDPA picks a backend
   shows up as ``aten::_scaled_dot_product_flash_attention`` /
   ``_scaled_dot_product_efficient_attention`` / ``_math`` on CUDA, or
   the equivalent torch_npu op on NPU (e.g. ``npu::npu_fusion_attention``
   or ``npu::npu_fused_attention_score``).
4. backend registrations for SDPA on PrivateUse1 (NPU's dispatch key).

The point is to learn the *exact name and arguments* of the NPU kernel
that the unpatched eval is calling, so we can compare it to what the
manual fp32 patch does and identify the dispatch discrepancy.
"""
from __future__ import annotations

import os
import sys

# Belt-and-braces: make sure the parity loader, if any .pth installed it,
# refuses to patch. We want the *unpatched* dispatch path.
os.environ.setdefault("PARITY_DISABLE_FP32_ATTN", "1")

import inspect

import torch
import torch.nn.functional as F


# ── 1. version info ─────────────────────────────────────────────────────────
print(f"python:        {sys.version.split()[0]}")
print(f"torch:         {torch.__version__}")
print(f"cuda avail:    {torch.cuda.is_available()}")

try:
    import torch_npu  # noqa: F401  -- importing registers the npu backend
    _has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    _npu_ver = getattr(torch_npu, "__version__", "(version attr missing)")
    print(f"torch_npu:     {_npu_ver}")
    print(f"npu avail:     {_has_npu}")
except ImportError:
    _has_npu = False
    print("torch_npu:     (not installed -- this is probably a GPU host)")

try:
    import transformers
    print(f"transformers:  {transformers.__version__}")
except ImportError:
    print("transformers:  (not installed)")

if _has_npu:
    DEVICE = torch.device(f"npu:{torch.npu.current_device()}")
elif torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    DEVICE = torch.device("cpu")
print(f"device:        {DEVICE}")
print()


# ── 2. Llama's SDPA call site ───────────────────────────────────────────────
print("=== where Llama's SDPA forward lives ===")
try:
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
    fn = LlamaSdpaAttention.forward
    print(f"  LlamaSdpaAttention.forward")
    print(f"  src:    {inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}")
except ImportError:
    try:
        from transformers.models.llama import modeling_llama as m
        for name in ("sdpa_attention_forward", "eager_attention_forward"):
            f = getattr(m, name, None)
            if f is not None:
                try:
                    src = f"{inspect.getsourcefile(f)}:{inspect.getsourcelines(f)[1]}"
                except (TypeError, OSError):
                    src = "(builtin)"
                print(f"  {name}\n  src:    {src}")
    except Exception as e:
        print(f"  (failed to locate: {e})")
print()


# ── 3. Trace dispatch of a representative SDPA call ────────────────────────
from torch.utils._python_dispatch import TorchDispatchMode


def _fmt(arg) -> str:
    if isinstance(arg, torch.Tensor):
        return f"T{tuple(arg.shape)}@{str(arg.dtype).removeprefix('torch.')}"
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return repr(arg)
    return f"<{type(arg).__name__}>"


class TraceDispatch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        parts = [_fmt(a) for a in args]
        parts += [f"{k}={_fmt(v)}" for k, v in kwargs.items()]
        print(f"  [disp] {str(func):65s}  ({', '.join(parts)})")
        return func(*args, **kwargs)


print("=== SDPA dispatch trace (OpenVLA-OFT-ish shape, GQA expanded) ===")
torch.manual_seed(0)
B, H, KV_H, S, D = 1, 32, 8, 64, 128
q = torch.randn(B, H,    S, D, dtype=torch.bfloat16, device=DEVICE)
k = torch.randn(B, KV_H, S, D, dtype=torch.bfloat16, device=DEVICE)
v = torch.randn(B, KV_H, S, D, dtype=torch.bfloat16, device=DEVICE)
# Llama does repeat_kv BEFORE the SDPA call, so we feed the expanded K/V too.
k = k.repeat_interleave(H // KV_H, dim=1)
v = v.repeat_interleave(H // KV_H, dim=1)

# Llama-style 4D additive causal mask: [B, 1, S, S], bf16, -inf above diag.
mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16, device=DEVICE)
upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=DEVICE), diagonal=1)
mask.masked_fill_(upper, float("-inf"))

with TraceDispatch(), torch.no_grad():
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

print(f"  output:        {tuple(out.shape)} @ {out.dtype}  on {out.device}")
print(f"  output stats:  abs.max={out.abs().max().item():.4e}, "
      f"isnan={out.isnan().any().item()}, isinf={out.isinf().any().item()}")
print()


# ── 4. Backend registrations for the SDPA aten ops ─────────────────────────
print("=== backend registrations for SDPA aten ops ===")
sdpa_ops = [
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_attention_math",
    "aten::_scaled_dot_product_cudnn_attention",
]
for op_name in sdpa_ops:
    print(f"  {op_name}:")
    try:
        dump = torch._C._dispatch_dump(op_name)
        # Highlight PrivateUse1 (NPU), CUDA, CPU lines
        keep = []
        for ln in (dump or "").splitlines():
            if any(k in ln for k in ("PrivateUse1", "CUDA", "CPU", "Meta", "Autograd")):
                keep.append(ln.strip())
        if keep:
            for ln in keep[:10]:
                print(f"      {ln}")
        else:
            print(f"      (no relevant backend keys found)")
    except Exception as e:
        print(f"      (dump failed: {e})")


# ── 5. Quick numerical check against a manual fp32 reference ────────────────
print()
print("=== manual fp32 vs SDPA output (same inputs, same device) ===")
import math
with torch.no_grad():
    qf, kf, vf = q.float(), k.float(), v.float()
    attn = (qf @ kf.transpose(-2, -1)) * (1.0 / math.sqrt(D)) + mask.float()
    attn = F.softmax(attn, dim=-1)
    out_manual = (attn @ vf).to(q.dtype)

diff = (out - out_manual).abs()
of = out.flatten().float()
mf = out_manual.flatten().float()
cos = torch.nn.functional.cosine_similarity(of, mf, dim=0).item()
print(f"  max abs diff:  {diff.max().item():.4e}")
print(f"  mean abs diff: {diff.mean().item():.4e}")
print(f"  cosine sim:    {cos:.6f}")
if cos < 0.99:
    print("  ⚠  SDPA output deviates from a manual fp32 reference on this device.")
    print("     If this is NPU, that is the dispatch divergence the eval is hitting.")
else:
    print("  ✓  SDPA and manual fp32 agree on toy input. If real eval still fails,")
    print("     the bug is triggered by a specific shape/mask pattern, not the kernel itself.")

print()
print("=== done ===")
