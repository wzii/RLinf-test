#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Verify install_fp32_attention on the current device.

Checks three things:
  1. Self-consistency: two patched runs give bit-identical outputs.
  2. Validity: patched outputs don't contain NaN / Inf.
  3. Drift from bf16 baseline: compares patched prev_logprobs/actions against
     an existing unpatched GPU golden (openvla_oft_rlinf_gpu.pt). The drift
     should be small (~1e-3 level) if fp32 matmul is the only change.

Usage
-----
    python tests/parity/openvla_oft/run_fp32attn_check.py
    python tests/parity/openvla_oft/run_fp32attn_check.py \
        --baseline tests/parity/goldens/openvla_oft_rlinf_gpu.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    GOLDENS_DIR,
    NUM_SAMPLES,
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    compare_tensors,
    device_label,
    format_diff_table,
    install_fp32_attention,
    pick_device,
    print_env_banner,
    save_golden,
    set_determinism,
)
from run_rlinf import build_cfg, build_env_obs, run_forward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=OPENVLA_OFT_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--baseline", type=Path,
                    default=GOLDENS_DIR / "openvla_oft_rlinf_gpu.pt",
                    help="Unpatched golden to compare against (bf16 baseline)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "rlinf-fp32attn-check", "ckpt": str(args.model_path)})

    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    cfg = build_cfg(args.model_path)
    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"[check] model loaded in {time.time() - t0:.1f}s")

    # Apply fp32 attention patch.
    install_fp32_attention(model)

    inputs = collect_libero_image_inputs(n=args.num_samples)
    print(f"[check] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs, device)

    # Run twice for self-consistency.
    runs = []
    for i in range(2):
        set_determinism()
        t1 = time.time()
        with torch.no_grad():
            out = run_forward(model, env_obs)
        print(f"[check] run {i}: {time.time() - t1:.2f}s")
        runs.append(out)

    print("\n=== 1. self-consistency (patched run0 vs run1) ===")
    diffs_self = [compare_tensors(runs[0][k], runs[1][k], name=k) for k in runs[0]]
    print(format_diff_table(diffs_self))
    any_drift = any(not d.bit_exact for d in diffs_self)
    if any_drift:
        print("[check] WARN: self-consistency FAILED — determinism flags did not fully pin the patched math.")
    else:
        print("[check] self-consistency OK ✓")

    # Check for NaN / Inf.
    has_nan = any(torch.isnan(v).any() or torch.isinf(v).any()
                  for v in runs[0].values() if torch.is_tensor(v))
    print(f"\n=== 2. NaN/Inf check: {'FAIL ✗' if has_nan else 'OK ✓'} ===")
    if has_nan:
        for k, v in runs[0].items():
            if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"   {k}: contains NaN/Inf")

    # Compare against unpatched bf16 baseline.
    if args.baseline.exists():
        print(f"\n=== 3. drift from bf16 baseline ({args.baseline.name}) ===")
        bl = torch.load(args.baseline, map_location="cpu", weights_only=False)
        bl_tensors = bl["tensors"]
        keys = [k for k in runs[0] if k in bl_tensors and torch.is_tensor(runs[0][k])]
        diffs_bl = [compare_tensors(bl_tensors[k], runs[0][k], name=k) for k in keys]
        print(format_diff_table(diffs_bl))
        logprob_d = next((d for d in diffs_bl if d.name == "prev_logprobs"), None)
        if logprob_d:
            if logprob_d.max_abs < 1e-1:
                print(f"[check] logprob drift {logprob_d.max_abs:.2e} < 0.1 — patch is a small numeric change ✓")
            else:
                print(f"[check] WARN: logprob drift {logprob_d.max_abs:.2e} is large — patch may have changed model behaviour")
    else:
        print(f"\n[check] baseline {args.baseline} not found, skipping drift check")

    out_path = args.out or (GOLDENS_DIR / f"openvla_oft_rlinf_fp32attn_{device_label(device)}.pt")
    save_golden({
        "schema": "openvla_oft_rlinf_fp32attn_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "tensors": runs[0],
        "self_consistent": not any_drift,
        "has_nan": has_nan,
    }, out_path)
    print(f"\n[check] wrote {out_path}")
    return 0 if (not any_drift and not has_nan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
