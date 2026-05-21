#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Diff two parity golden files produced by the OpenVLA-OFT / Wan scripts.

Usage
-----
    python tests/parity/compare.py A.pt B.pt
        [--rtol-strict 1e-5] [--atol-strict 1e-5]
        [--rtol-bf16 1e-2]   [--atol-bf16 1e-2]
        [--report report.json]

Exit code
---------
0 if every tensor is bit-exact.
1 if any tensor diverges past the strict tolerance.
2 if any tensor diverges past the bf16 tolerance.

The two files must share the same ``schema`` (e.g. both ``openvla_oft_rlinf_v1``).
Their ``input_fingerprint`` is also compared — if those differ the input
collectors disagreed, which makes any output diff meaningless.

This script does both the bit-exact / allclose checks AND the semantic
sanity checks (chunk-action L1, action-token agreement, reward L1) so a
single run gives you both the "is the math the same" and the "would the
policy behave the same" answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/parity/

import torch  # noqa: E402

from common import (  # noqa: E402
    compare_tensors,
    format_diff_table,
    load_golden,
    tensor_hash,
)


def semantic_metrics(schema: str, a: dict, b: dict) -> dict[str, float]:
    """Per-schema downstream metrics — these answer 'would the policy act
    differently?' rather than 'are tensors equal?'."""
    out: dict[str, float] = {}
    ta, tb = a["tensors"], b["tensors"]

    if schema.startswith("openvla_oft_"):
        # Mean L1 over decoded 7D actions and token-agreement rate.
        act_a = ta["actions"].to(torch.float32)
        act_b = tb["actions"].to(torch.float32)
        out["action_mean_l1"] = float((act_a - act_b).abs().mean().item())
        out["action_max_l1"] = float((act_a - act_b).abs().max().item())
        # cosine over flattened per-sample action vectors
        flat_a = act_a.reshape(act_a.shape[0], -1)
        flat_b = act_b.reshape(act_b.shape[0], -1)
        cos = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=-1)
        out["action_cos_mean"] = float(cos.mean().item())
        out["action_cos_min"] = float(cos.min().item())
        # token agreement
        tok_a = ta["action_tokens"]
        tok_b = tb["action_tokens"]
        if tok_a.shape == tok_b.shape:
            out["token_agreement"] = float((tok_a == tok_b).float().mean().item())
        else:
            out["token_agreement"] = float("nan")

    elif schema == "wan_rlinf_env_v1":
        ra = ta["rewards"].to(torch.float32)
        rb = tb["rewards"].to(torch.float32)
        out["reward_mean_l1"] = float((ra - rb).abs().mean().item())
        out["reward_max_l1"] = float((ra - rb).abs().max().item())
        out["term_agreement"] = float(
            (ta["terminations"] == tb["terminations"]).float().mean().item()
        )
        img_a = ta["main_images_after"].to(torch.float32)
        img_b = tb["main_images_after"].to(torch.float32)
        # uint8 image; report mean abs diff in 0..255 scale and PSNR-ish
        diff = (img_a - img_b).abs()
        out["image_mean_abs"] = float(diff.mean().item())
        out["image_max_abs"] = float(diff.max().item())

    elif schema == "wan_upstream_pipeline_v1":
        v_a = ta["video"].to(torch.float32)
        v_b = tb["video"].to(torch.float32)
        diff = (v_a - v_b).abs()
        out["video_mean_abs"] = float(diff.mean().item())
        out["video_max_abs"] = float(diff.max().item())
        mse = ((v_a - v_b) ** 2).mean().item()
        # values are in [-1, 1] so peak signal is 2.0
        out["video_psnr"] = float(
            float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(4.0 / mse)).item()
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--rtol-strict", type=float, default=1e-5)
    ap.add_argument("--atol-strict", type=float, default=1e-5)
    ap.add_argument("--rtol-bf16", type=float, default=1e-2)
    ap.add_argument("--atol-bf16", type=float, default=1e-2)
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path for a machine-readable JSON report.")
    args = ap.parse_args()

    a = load_golden(args.a)
    b = load_golden(args.b)
    if a["schema"] != b["schema"]:
        print(f"[parity][ERROR] schema mismatch: {a['schema']} vs {b['schema']}",
              file=sys.stderr)
        return 3

    print(f"[parity] schema    : {a['schema']}")
    print(f"[parity] devices   : {a.get('device_label')} vs {b.get('device_label')}")
    print(f"[parity] N samples : {a.get('num_samples')} vs {b.get('num_samples')}")
    fp_a = a.get("input_fingerprint")
    fp_b = b.get("input_fingerprint")
    print(f"[parity] input fp  : {'MATCH' if fp_a == fp_b else 'MISMATCH'}")
    if fp_a != fp_b:
        print(f"[parity][WARN] input fingerprints differ — outputs are not comparable.")
        print(f"  a: {fp_a}")
        print(f"  b: {fp_b}")

    # Per-tensor diff
    diffs = []
    for k in a["tensors"].keys():
        if k not in b["tensors"]:
            print(f"[parity][WARN] tensor {k} missing in B")
            continue
        diffs.append(
            compare_tensors(
                a["tensors"][k], b["tensors"][k], name=k,
                rtol_strict=args.rtol_strict, atol_strict=args.atol_strict,
                rtol_bf16=args.rtol_bf16, atol_bf16=args.atol_bf16,
            )
        )
    print("\n=== tensor diffs ===")
    print(format_diff_table(diffs))

    sem = semantic_metrics(a["schema"], a, b)
    print("\n=== semantic metrics ===")
    for k, v in sem.items():
        print(f"  {k}: {v}")

    # Per-sample hash comparison for human-readable diagnostics.
    if "per_sample" in a and "per_sample" in b:
        diverged = [
            i for i, (pa, pb) in enumerate(zip(a["per_sample"], b["per_sample"]))
            if pa.get("hash") != pb.get("hash")
        ]
        print(f"\n[parity] per-sample mismatches: {len(diverged)} / {len(a['per_sample'])}")
        for i in diverged[:8]:
            pa, pb = a["per_sample"][i], b["per_sample"][i]
            print(f"  [{i}] A: mean={pa['mean']:.6f} std={pa['std']:.6f}  "
                  f"B: mean={pb['mean']:.6f} std={pb['std']:.6f}")
        if len(diverged) > 8:
            print(f"  …and {len(diverged) - 8} more")

    if args.report is not None:
        report = {
            "a": str(args.a),
            "b": str(args.b),
            "schema": a["schema"],
            "device_a": a.get("device_label"),
            "device_b": b.get("device_label"),
            "input_fingerprint_match": fp_a == fp_b,
            "tensor_diffs": [d.__dict__ for d in diffs],
            "semantic_metrics": sem,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"[parity] report written to {args.report}")

    bit_exact = all(d.bit_exact for d in diffs)
    strict_ok = all(d.allclose_strict for d in diffs)
    bf16_ok = all(d.allclose_bf16 for d in diffs)
    print(f"\n[parity] result: bit_exact={bit_exact} strict={strict_ok} bf16={bf16_ok}")
    if bit_exact:
        return 0
    if strict_ok:
        return 0
    if not bf16_ok:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
