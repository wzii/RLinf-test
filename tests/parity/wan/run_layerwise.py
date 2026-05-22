#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Wan DiT + VAE layer-by-layer activation capture for GPU<->NPU debugging.

Why this exists
---------------
``run_upstream_pipeline.py`` only captures the final video frames. When GPU and
NPU disagree, that tells you THAT they disagree but not WHERE. This script
registers forward hooks on every ``named_modules()`` entry in the DiT (WanModel)
and the VAE (WanVideoVAE), recording in execution order:

  - a cheap fp32 fingerprint (mean / std / abs-max / L2 / checksum) for EVERY
    submodule output -- keyed as ``dit/{name}@s{step}`` for the DiT (repeated
    for each of the ``num_inference_steps`` denoising steps) and ``vae/{name}``
    for the VAE decode pass;
  - the full fp32 tensor (input AND output) for a small boundary allowlist
    (patch_embedding, a few DiTBlocks, head; decoder key stages) -- only
    captured for DiT step 0 to keep the golden small.

Run on each device and diff with ``compare_layers.py``:

    python tests/parity/wan/run_layerwise.py            # -> wan_layers_<dev>.pt
    python tests/parity/compare_layers.py \\
        tests/parity/goldens/wan_layers_gpu.pt \\
        tests/parity/goldens/wan_layers_npu.pt

The FIRST layer flagged ``>>> FIRST DIVERGENCE`` (input matched, output diverged)
names the exact op class (conv / attention / layernorm / matmul) that disagrees
across hardware.

Determinism notes
-----------------
Fixed noise is installed via ``install_fixed_noise`` (same as the end-to-end Wan
parity scripts) so the diffusion noise is byte-identical across architectures.
No microbatching is used here -- ``--num-samples`` defaults to 4 so each forward
is already small; scale up if you want more statistical signal but 4 samples is
enough to expose the first-diverging op.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/

from common import (  # noqa: E402
    GOLDENS_DIR,
    WAN_CKPT_DIR,
    _first_tensor,
    collect_wan_inputs,
    device_label,
    install_fixed_noise,
    layer_fingerprint as fingerprint,
    pick_device,
    print_env_banner,
    save_golden,
    set_determinism,
)
from run_upstream_pipeline import build_pipeline, to_pil_list  # noqa: E402

# Skip full-tensor capture above this many elements (per sliced tensor).
MAX_FULL_ELEMS = 40_000_000

# Small default: 4 samples is enough to expose the first-diverging op and keeps
# each Wan forward lightweight (no microbatching needed at this scale).
DEFAULT_NUM_SAMPLES = 4


def _sparse_indices(n: int) -> list[int]:
    """first / quartiles / last, deduped."""
    if n <= 0:
        return []
    if n <= 5:
        return list(range(n))
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})


def select_dit_boundary_names(named: dict) -> list[str]:
    """Sparse but representative set of DiT module names for full-tensor capture."""
    names = list(named.keys())
    chosen: list[str] = []

    def add(n):
        if n and n in named and n not in chosen:
            chosen.append(n)

    # 3-D patch conv (very first op that processes the latent)
    add(next((n for n in names if n == "patch_embedding"), None))

    # Sparse set of DiTBlocks (whole block, not sub-layers, to keep golden small)
    blocks = sorted(
        [n for n in names if re.fullmatch(r"blocks\.\d+", n)],
        key=lambda n: int(n.split(".")[1]),
    )
    for idx in _sparse_indices(len(blocks)):
        add(blocks[idx])

    # Output head (norm + linear that maps back to latent space)
    add(next((n for n in names if n == "head"), None))

    return [n for n in chosen if n]


def select_vae_boundary_names(named: dict) -> list[str]:
    """Key decode-path stages in the VAE."""
    names = list(named.keys())
    chosen: list[str] = []

    def add(n):
        if n and n in named and n not in chosen:
            chosen.append(n)

    # Top-level encoder / decoder if they exist as direct children
    for candidate in ("encoder", "decoder", "quant_conv", "post_quant_conv"):
        add(next((n for n in names if n == candidate), None))

    # Mid-blocks and a few up/down-sample blocks (depth ≤ 2 to avoid too many)
    shallow = [n for n in names if n.count(".") <= 1 and n not in chosen]
    for idx in _sparse_indices(len(shallow)):
        add(shallow[idx])

    return [n for n in chosen if n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=WAN_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    ap.add_argument("--full-samples", type=int, default=2,
                    help="batch elements stored as full tensors for boundary layers")
    ap.add_argument("--repeats", type=int, default=2, help="self-consistency runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "wan-layerwise", "ckpt": str(args.ckpt_dir)})

    t0 = time.time()
    pipe = build_pipeline(args.ckpt_dir, device)
    print(f"[parity] pipeline built in {time.time() - t0:.1f}s")

    # Pin diffusion noise to a byte-identical artifact across x86/aarch64.
    # No microbatching: num_samples is small by design so each forward is light.
    install_fixed_noise(pipe)

    inputs = collect_wan_inputs(n=args.num_samples)
    print(f"[parity] inputs fingerprint = {inputs['fingerprint']}")

    dit_named = {n: m for n, m in pipe.dit.named_modules() if n}
    vae_named = {n: m for n, m in pipe.vae.named_modules() if n}
    dit_boundary = set(select_dit_boundary_names(dit_named))
    vae_boundary = set(select_vae_boundary_names(vae_named))
    print(f"[parity] DiT modules: {len(dit_named)}, boundary: {len(dit_boundary)}")
    for n in sorted(dit_boundary):
        print(f"           dit/{n}")
    print(f"[parity] VAE modules: {len(vae_named)}, boundary: {len(vae_boundary)}")
    for n in sorted(vae_boundary):
        print(f"           vae/{n}")

    nf = args.full_samples

    # Prepare pipeline inputs once (same for every repeat).
    cond_frames = inputs["cond_frames"]
    cond_actions = torch.from_numpy(inputs["cond_actions"]).to(device=device, dtype=torch.bfloat16)
    new_actions = torch.from_numpy(inputs["new_actions"]).to(device=device, dtype=torch.bfloat16)
    actions_tensor = torch.cat([cond_actions, new_actions], dim=1)
    input_image, input_image4 = to_pil_list(cond_frames)
    pipe_kwargs = dict(
        seed=0,
        tiled=False,
        input_image=input_image,
        input_image4=input_image4,
        action=actions_tensor,
        height=256,
        width=256,
        num_frames=13,
        num_inference_steps=5,
        cfg_scale=1.0,
        progress_bar_cmd=lambda x: x,
        batch_size=cond_frames.shape[0],
    )

    def run_once() -> dict:
        records: dict[str, dict] = {}
        order = {"i": 0}
        dit_step = {"n": -1}  # incremented to 0 before the first DiT forward
        handles = []

        # Pre-hook on the top-level DiT: fires before any submodule hook for
        # that denoising step, so dit_step["n"] is already correct when
        # submodule hooks run.
        def _dit_pre(module, inp):
            dit_step["n"] += 1

        handles.append(pipe.dit.register_forward_pre_hook(_dit_pre))

        def make_dit_hook(name):
            def hook(module, inp, out):
                s = dit_step["n"]
                key = f"dit/{name}@s{s}"
                ot = _first_tensor(out)
                if ot is None:
                    return
                rec = records.setdefault(key, {})
                if "fp_out" in rec:
                    rec["calls"] = rec.get("calls", 1) + 1
                    return
                rec["order"] = order["i"]
                order["i"] += 1
                rec["fp_out"] = fingerprint(ot)
                it = _first_tensor(inp)
                if it is not None:
                    rec["fp_in"] = fingerprint(it)
                # Full tensors only for step 0 (keeps the golden manageable).
                if s == 0 and name in dit_boundary:
                    if ot.numel() // max(ot.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_out"] = ot[:nf].detach().to("cpu", torch.float32)
                    if it is not None and it.numel() // max(it.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_in"] = it[:nf].detach().to("cpu", torch.float32)
            return hook

        def make_vae_hook(name):
            def hook(module, inp, out):
                key = f"vae/{name}"
                ot = _first_tensor(out)
                if ot is None:
                    return
                rec = records.setdefault(key, {})
                if "fp_out" in rec:
                    rec["calls"] = rec.get("calls", 1) + 1
                    return
                rec["order"] = order["i"]
                order["i"] += 1
                rec["fp_out"] = fingerprint(ot)
                it = _first_tensor(inp)
                if it is not None:
                    rec["fp_in"] = fingerprint(it)
                if name in vae_boundary:
                    if ot.numel() // max(ot.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_out"] = ot[:nf].detach().to("cpu", torch.float32)
                    if it is not None and it.numel() // max(it.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_in"] = it[:nf].detach().to("cpu", torch.float32)
            return hook

        for name, mod in dit_named.items():
            handles.append(mod.register_forward_hook(make_dit_hook(name)))
        for name, mod in vae_named.items():
            handles.append(mod.register_forward_hook(make_vae_hook(name)))

        try:
            with torch.no_grad():
                pipe(**pipe_kwargs)
        finally:
            for h in handles:
                h.remove()
        return records

    runs = []
    for i in range(args.repeats):
        set_determinism()
        t1 = time.time()
        r = run_once()
        runs.append(r)
        print(f"[parity] layerwise run {i}: {time.time() - t1:.2f}s, {len(r)} records")

    # Same-device self-consistency check.
    if args.repeats > 1:
        a, b = runs[0], runs[-1]
        drift = [
            n for n in a if n in b
            and a[n].get("fp_out", {}).get("chk") != b[n].get("fp_out", {}).get("chk")
        ]
        if drift:
            print(f"\n[parity][WARN] {len(drift)} records drift between repeats on the "
                  f"SAME device — determinism not fully pinned. First few: {drift[:5]}")
        else:
            print("\n[parity] self-consistency OK: all fingerprints bit-exact across repeats.")

    payload = {
        "schema": "wan_layers_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "full_samples": nf,
        "input_fingerprint": inputs["fingerprint"],
        "dit_boundary_layers": sorted(dit_boundary),
        "vae_boundary_layers": sorted(vae_boundary),
        "records": runs[0],
    }
    out_path = args.out or (GOLDENS_DIR / f"wan_layers_{device_label(device)}.pt")
    save_golden(payload, out_path)
    n_full = sum(1 for r in runs[0].values() if "full_out" in r)
    print(f"[parity] wrote {out_path}  ({len(runs[0])} records, {n_full} with full tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
