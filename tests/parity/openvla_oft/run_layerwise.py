#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""OpenVLA-OFT layer-by-layer activation capture for GPU<->NPU debugging.

Why this exists
---------------
``run_rlinf.py`` only captures the *endpoints* (pixel_values in, action tokens /
logprobs / actions out). When GPU and NPU disagree, that tells you THAT they
disagree but not WHERE. This script registers a forward hook on every
``named_modules()`` entry and records, in execution order:

  - a cheap fp32 fingerprint (mean / std / abs-max / L2 / sha256) for EVERY
    module output -- enough to localise the first diverging layer, and
  - the full fp32 tensor (input AND output, sliced to the first
    ``--full-samples`` batch elements) for a small dynamic *boundary* allowlist
    (patch_embed, a few vision blocks, vision norm, projector, a few LLM layers,
    final norm) -- enough for Stage-2 "input injection" (replay one layer in
    isolation with the GPU input on the NPU).

Run it on each device and diff with ``compare_layers.py``:

    python tests/parity/openvla_oft/run_layerwise.py            # -> openvla_oft_layers_<dev>.pt
    python tests/parity/compare_layers.py \
        tests/parity/goldens/openvla_oft_layers_gpu.pt \
        tests/parity/goldens/openvla_oft_layers_npu.pt

The first layer whose fingerprint flips from bit-exact to divergent (while its
input was still ~identical) is the culprit op class (conv / attention /
layernorm / silu / matmul).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))      # openvla_oft/

from common import (  # noqa: E402
    GOLDENS_DIR,
    NUM_SAMPLES,
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    device_label,
    pick_device,
    print_env_banner,
    save_golden,
    set_determinism,
)
from run_rlinf import build_cfg, build_env_obs  # noqa: E402


def _first_tensor(x):
    """Return the first torch.Tensor found in a (possibly nested) structure."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for e in x:
            t = _first_tensor(e)
            if t is not None:
                return t
    if isinstance(x, dict):
        for e in x.values():
            t = _first_tensor(e)
            if t is not None:
                return t
    return None


def fingerprint(t: torch.Tensor) -> dict:
    """Cheap, device-independent summary of a tensor.

    Computed with reductions ON the tensor's own device (no CPU copy of the
    activation -- copying every layer's hidden state to CPU to hash it is what
    made the first version unusably slow). The four scalars (mean / std /
    abs-max / L2) plus a device-side order-sensitive checksum are enough to (a)
    measure the magnitude of any GPU<->NPU disagreement and (b) flag exact
    equality across same-device repeats. Only a 5-value vector crosses to CPU.
    """
    orig_dtype = str(t.dtype)
    d = t.detach().reshape(-1)
    n = d.numel()
    if n == 0:
        return {"shape": tuple(t.shape), "dtype": orig_dtype,
                "mean": 0.0, "std": 0.0, "absmax": 0.0, "l2": 0.0, "chk": 0.0}
    f = d.float()
    # Order-sensitive checksum: dot the flattened signal against a deterministic
    # ramp so a permutation or any single-element change moves the value (a plain
    # sum would not). The ramp is the same shape on both devices, so equal chk =>
    # bit-identical layout. Cheap GPU reduction, no host copy of the activation.
    chk = torch.dot(f, torch.linspace(1.0, 2.0, n, device=f.device, dtype=torch.float32))
    vec = torch.stack([
        f.mean(),
        f.std() if n > 1 else f.new_zeros(()),
        f.abs().max(),
        f.norm(),
        chk,
    ]).double().cpu()
    return {
        "shape": tuple(t.shape),
        "dtype": orig_dtype,
        "mean": vec[0].item(),
        "std": vec[1].item(),
        "absmax": vec[2].item(),
        "l2": vec[3].item(),
        "chk": vec[4].item(),
    }


def select_boundary_names(named: dict[str, torch.nn.Module]) -> list[str]:
    """Pick a small, representative set of module names to capture in FULL.

    Discovered dynamically from the real module tree so we never hardcode a
    layer count that doesn't match this checkpoint.
    """
    names = list(named.keys())
    chosen: list[str] = []

    def add(n):
        if n and n in named and n not in chosen:
            chosen.append(n)

    # Vision: patch embed, a sparse set of blocks, the featurizer's final norm.
    add(next((n for n in names if n.endswith("patch_embed")), None))
    vis_blocks = sorted(
        [n for n in names if re.search(r"vision_backbone\..*\.blocks\.\d+$", n)],
        key=lambda n: int(n.rsplit(".", 1)[1]),
    )
    for idx in _sparse_indices(len(vis_blocks)):
        add(vis_blocks[idx])
    add(next((n for n in names if re.search(r"featurizer\.norm$", n)), None))

    # Projector (vision -> LLM dim).
    add(next((n for n in names if re.search(r"projector\.fc2$", n)), None))
    add(next((n for n in names if n.endswith("projector")), None))

    # LLM: a sparse set of whole decoder layers + the final norm.
    llm_layers = sorted(
        [n for n in names if re.search(r"language_model\..*\.layers\.\d+$", n)],
        key=lambda n: int(n.rsplit(".", 1)[1]),
    )
    for idx in _sparse_indices(len(llm_layers)):
        add(llm_layers[idx])
    add(next((n for n in names if re.search(r"language_model\..*\.model\.norm$", n)
              or re.search(r"language_model\.model\.norm$", n)), None))
    # lm_head fingerprint only (full logits are huge) -- handled via fingerprint
    # of all modules anyway; we still add it to the boundary list for clarity but
    # the writer skips full capture for very large tensors (see MAX_FULL_ELEMS).
    add(next((n for n in names if n.endswith("lm_head")), None))
    return chosen


def _sparse_indices(n: int) -> list[int]:
    """first / quartiles / last, deduped, for n items."""
    if n <= 0:
        return []
    if n <= 5:
        return list(range(n))
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})


# Skip full-tensor capture above this many elements (per sliced tensor) to keep
# the golden small; lm_head logits / huge intermediates fall back to fingerprint.
MAX_FULL_ELEMS = 40_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=OPENVLA_OFT_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--full-samples", type=int, default=4,
                    help="batch elements stored as FULL tensors for boundary layers")
    ap.add_argument("--repeats", type=int, default=2, help="self-consistency runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "rlinf-layerwise", "ckpt": str(args.model_path)})

    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    cfg = build_cfg(args.model_path)
    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"[parity] model loaded in {time.time() - t0:.1f}s")

    inputs = collect_libero_image_inputs(n=args.num_samples)
    print(f"[parity] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs, device)

    id2name = {id(m): n for n, m in model.named_modules() if n}
    named = {n: m for n, m in model.named_modules() if n}
    boundary = set(select_boundary_names(named))
    print(f"[parity] capturing FULL tensors for {len(boundary)} boundary layers:")
    for n in sorted(boundary):
        print(f"           {n}")

    nf = args.full_samples

    def run_once() -> dict:
        records: dict[str, dict] = {}
        order = {"i": 0}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                ot = _first_tensor(out)
                if ot is None:
                    return
                rec = records.setdefault(name, {})
                # If a module fires twice, keep the first; note the repeat count.
                if "fp_out" in rec:
                    rec["calls"] = rec.get("calls", 1) + 1
                    return
                rec["order"] = order["i"]
                order["i"] += 1
                rec["fp_out"] = fingerprint(ot)
                it = _first_tensor(inp)
                if it is not None:
                    rec["fp_in"] = fingerprint(it)
                if name in boundary:
                    if ot.numel() // max(ot.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_out"] = ot[:nf].detach().to("cpu", torch.float32)
                    if it is not None and it.numel() // max(it.shape[0], 1) * nf <= MAX_FULL_ELEMS:
                        rec["full_in"] = it[:nf].detach().to("cpu", torch.float32)
            return hook

        for name, mod in named.items():
            handles.append(mod.register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                actions, result = model.predict_action_batch(
                    env_obs=env_obs, do_sample=False, temperature=1.0,
                    top_k=-1, top_p=1.0, max_new_tokens=None, mode="eval",
                )
        finally:
            for h in handles:
                h.remove()
        records["__actions__"] = {"order": 10**9, "fp_out": fingerprint(actions.float())}
        return records

    runs = []
    for i in range(args.repeats):
        set_determinism()
        t1 = time.time()
        runs.append(run_once())
        print(f"[parity] layerwise run {i}: {time.time() - t1:.2f}s, "
              f"{len(runs[-1])} modules captured")

    # Same-device self-consistency on the fingerprints (sha256).
    if args.repeats > 1:
        a, b = runs[0], runs[-1]
        drift = [n for n in a if n in b
                 and a[n].get("fp_out", {}).get("chk")
                 != b[n].get("fp_out", {}).get("chk")]
        if drift:
            print(f"\n[parity][WARN] {len(drift)} modules drift between repeats on "
                  f"the SAME device — determinism not fully pinned. First few: "
                  f"{drift[:5]}")
        else:
            print("\n[parity] self-consistency OK: all module fingerprints "
                  "bit-exact across repeats.")

    payload = {
        "schema": "openvla_oft_layers_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "full_samples": nf,
        "input_fingerprint": inputs["fingerprint"],
        "boundary_layers": sorted(boundary),
        "records": runs[0],
    }
    out_path = args.out or (GOLDENS_DIR / f"openvla_oft_layers_{device_label(device)}.pt")
    save_golden(payload, out_path)
    n_full = sum(1 for r in runs[0].values() if "full_out" in r)
    print(f"[parity] wrote {out_path}  ({len(runs[0])} modules, {n_full} with full tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
