#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Diff two ``openvla_oft_layers_*.pt`` goldens to localise the FIRST diverging
layer between two devices (e.g. GPU vs NPU).

Usage
-----
    python tests/parity/compare_layers.py \
        tests/parity/goldens/openvla_oft_layers_gpu.pt \
        tests/parity/goldens/openvla_oft_layers_npu.pt \
        [--rel 1e-3] [--report report.json] [--top 25]

Reading the output
------------------
Layers are printed in execution order. For each layer:
  - ``bit`` : output checksum identical on both devices (bit-identical layout),
  - ``in✓`` : the layer's INPUT was (near-)identical -- so any output diff is
    produced BY THIS LAYER, not inherited,
  - ``relL2``: ||a-b|| / ||a|| over the fp32 stats-derived signal,
  - ``Δabsmax`` / ``maxAbsDiff``: magnitude (the latter is the true elementwise
    max difference, only available where both goldens stored full tensors).

The FIRST layer flagged ``>>> FIRST DIVERGENCE`` is where to look: its input
matched but its output didn't, so the op it runs (conv / attention / layernorm /
silu / matmul) is the kernel that disagrees across hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def rel_l2_from_stats(a: dict, b: dict) -> float:
    """Cheap divergence proxy from the scalar fingerprint (works even when no
    full tensor was stored): treat (mean, std, l2) as a 3-vector signal."""
    va = torch.tensor([a["mean"], a["std"], a["l2"]])
    vb = torch.tensor([b["mean"], b["std"], b["l2"]])
    denom = va.norm().item()
    return (va - vb).norm().item() / denom if denom > 1e-12 else (va - vb).norm().item()


def full_max_abs_diff(ra: dict, rb: dict, key: str):
    ta, tb = ra.get(key), rb.get(key)
    if ta is None or tb is None or tuple(ta.shape) != tuple(tb.shape):
        return None
    return (ta.float() - tb.float()).abs().max().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--rel", type=float, default=1e-3,
                    help="relL2 above this counts as diverged")
    ap.add_argument("--top", type=int, default=30, help="rows to print")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    if A.get("schema") != "openvla_oft_layers_v1" or B.get("schema") != A.get("schema"):
        print(f"[compare] schema mismatch: {A.get('schema')} vs {B.get('schema')}", file=sys.stderr)
        return 3
    if A.get("input_fingerprint") != B.get("input_fingerprint"):
        print("[compare][WARN] input fingerprints differ — inputs diverged, "
              "layer diffs below are NOT attributable to kernels.", file=sys.stderr)

    ra, rb = A["records"], B["records"]
    common = [n for n in ra if n in rb and "fp_out" in ra[n] and "fp_out" in rb[n]]
    common.sort(key=lambda n: ra[n]["order"])

    rows = []
    first_div = None
    for n in common:
        oa, ob = ra[n]["fp_out"], rb[n]["fp_out"]
        bit = oa["chk"] == ob["chk"]
        rel_out = rel_l2_from_stats(oa, ob)
        # input agreement (did divergence start here or was it inherited?)
        ia, ib = ra[n].get("fp_in"), rb[n].get("fp_in")
        if ia and ib:
            in_ok = (ia["chk"] == ib["chk"]) or rel_l2_from_stats(ia, ib) < args.rel
        else:
            in_ok = None
        mad = full_max_abs_diff(ra[n], rb[n], "full_out")
        diverged = (not bit) and rel_out >= args.rel
        if diverged and first_div is None and (in_ok is None or in_ok):
            first_div = n
        rows.append({
            "name": n, "order": ra[n]["order"], "bit": bit, "in_ok": in_ok,
            "rel_out": rel_out, "absmax_a": oa["absmax"], "absmax_b": ob["absmax"],
            "d_absmax": abs(oa["absmax"] - ob["absmax"]), "max_abs_diff": mad,
            "shape": oa["shape"], "dtype": oa["dtype"], "diverged": diverged,
        })

    # ---- print ----
    n_div = sum(r["diverged"] for r in rows)
    print(f"\n{args.a.name}  vs  {args.b.name}")
    print(f"{len(common)} common modules | {n_div} diverged (relL2 >= {args.rel})")
    if first_div:
        fr = next(r for r in rows if r["name"] == first_div)
        print(f"\n>>> FIRST DIVERGENCE: {first_div}")
        print(f"      (input matched, output relL2={fr['rel_out']:.2e}, "
              f"maxAbsDiff={fr['max_abs_diff']}, shape={fr['shape']}, dtype={fr['dtype']})")
        print("      => this layer's op is the kernel that disagrees across hardware.")
    else:
        print("\n>>> no layer exceeded the threshold — devices agree at this tolerance.")

    hdr = f"\n{'ord':>4} {'bit':>4} {'in':>4} {'relL2':>10} {'Δabsmax':>10} {'maxAbsDiff':>11}  layer"
    print(hdr)
    print("-" * (len(hdr) + 40))
    # Print the window around the first divergence (most informative), else head.
    if first_div:
        center = next(i for i, r in enumerate(rows) if r["name"] == first_div)
        lo = max(0, center - 3)
        window = rows[lo:lo + args.top]
    else:
        window = rows[:args.top]
    for r in window:
        flag = ">>" if r["name"] == first_div else ("! " if r["diverged"] else "  ")
        inmark = "—" if r["in_ok"] is None else ("✓" if r["in_ok"] else "✗")
        mad = "—" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.3e}"
        print(f"{flag}{r['order']:>3} {('Y' if r['bit'] else 'n'):>4} {inmark:>4} "
              f"{r['rel_out']:>10.3e} {r['d_absmax']:>10.3e} {mad:>11}  {r['name']}")

    if args.report:
        args.report.write_text(json.dumps(
            {"a": str(args.a), "b": str(args.b), "rel_threshold": args.rel,
             "first_divergence": first_div, "n_diverged": n_div, "rows": rows},
            indent=2, default=lambda o: None))
        print(f"\n[compare] wrote {args.report}")

    return 0 if n_div == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
