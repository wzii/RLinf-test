#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""OpenVLA-OFT input-injection parity test.

What this isolates
------------------
``compare_layers.py`` pinpointed that divergence between GPU and NPU starts in
the SigLIP vision backbone (featurizer blocks.1). To confirm that the LLM itself
is NOT a second source of error, this script takes the GPU-produced projector
output (stored as full tensors in the layerwise golden) and injects it into the
NPU model via a forward hook -- replacing the NPU's own projector output with
the GPU's bit-identical tensor.

If the resulting logprobs / action tokens match the GPU baseline:
  → the LLM forward on NPU is clean; vision backbone is the sole root cause.

If they still diverge:
  → the LLM also has hardware-specific kernel behaviour; both need attention.

The test covers exactly ``full_samples`` samples (default 4) -- the number of
samples for which the layerwise golden stored full tensors.

Injection point
---------------
``model.projector`` maps SigLIP features → LLM hidden dim (4096).
Its output ([B, 256, 4096]) is concatenated with text token embeddings
([B, 106, 4096]) to form the full LLM input ([B, 362, 4096]).
Text token embeddings come from a simple lookup table (identical on any device),
so after injection the LLM input is byte-identical to the GPU run.

Usage
-----
    python tests/parity/openvla_oft/run_inject.py \\
        --layers-golden tests/parity/goldens/openvla_oft_layers_gpu.pt \\
        --rlinf-golden  tests/parity/goldens/openvla_oft_rlinf_gpu.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # openvla_oft/

from common import (  # noqa: E402
    GOLDENS_DIR,
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    compare_tensors,
    device_label,
    format_diff_table,
    pick_device,
    print_env_banner,
    save_golden,
    set_determinism,
)
from run_rlinf import build_cfg, build_env_obs  # noqa: E402


def load_gpu_projector_out(layers_golden_path: Path) -> torch.Tensor:
    """Return the projector full_out tensor from the GPU layerwise golden.

    The tensor is fp32 on CPU, shape [full_samples, 256, 4096].
    """
    g = torch.load(layers_golden_path, map_location="cpu", weights_only=False)
    schema = g.get("schema", "")
    if not schema.endswith("_layers_v1"):
        raise ValueError(f"Expected *_layers_v1 golden, got schema={schema!r}")
    rec = g["records"].get("projector")
    if rec is None or "full_out" not in rec:
        raise KeyError(
            "Golden does not contain full_out for 'projector'. "
            "Re-run run_layerwise.py to regenerate (the projector must be in "
            "the boundary list)."
        )
    return rec["full_out"]  # [full_samples, 256, 4096], fp32 CPU


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=OPENVLA_OFT_CKPT_DIR)
    ap.add_argument(
        "--layers-golden",
        type=Path,
        default=GOLDENS_DIR / "openvla_oft_layers_gpu.pt",
        help="Layerwise golden that contains the GPU projector full_out",
    )
    ap.add_argument(
        "--rlinf-golden",
        type=Path,
        default=GOLDENS_DIR / "openvla_oft_rlinf_gpu.pt",
        help="End-to-end golden with GPU prev_logprobs / actions for comparison",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "rlinf-inject", "ckpt": str(args.model_path)})

    # ── Load GPU reference tensors ─────────────────────────────────────────────
    print(f"[inject] loading GPU layerwise golden: {args.layers_golden}")
    gpu_proj_out = load_gpu_projector_out(args.layers_golden)
    n = gpu_proj_out.shape[0]
    print(f"[inject] GPU projector out: shape={tuple(gpu_proj_out.shape)}  "
          f"(will run {n} samples)")

    print(f"[inject] loading GPU rlinf golden: {args.rlinf_golden}")
    gpu_rlinf = torch.load(args.rlinf_golden, map_location="cpu", weights_only=False)
    gpu_tensors = gpu_rlinf["tensors"]
    # Slice to the first n samples (the ones the full tensors cover).
    gpu_ref = {k: v[:n] for k, v in gpu_tensors.items() if torch.is_tensor(v)}

    # ── Build model ────────────────────────────────────────────────────────────
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    cfg = build_cfg(args.model_path)
    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"[inject] model loaded in {time.time() - t0:.1f}s")

    # Verify the projector attribute exists and has a matching output shape.
    if not hasattr(model, "projector"):
        raise AttributeError(
            "model.projector not found -- check the RLinf model wrapper."
        )

    # ── Prepare inputs (first n samples) ──────────────────────────────────────
    inputs = collect_libero_image_inputs(n=n)
    print(f"[inject] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs, device)

    # ── Register injection hook ────────────────────────────────────────────────
    # Move the GPU tensor to the NPU device once; the hook just returns it.
    proj_gpu_on_device = gpu_proj_out.to(device=device, dtype=torch.bfloat16)

    def _inject(module, inp, out):
        return proj_gpu_on_device

    handle = model.projector.register_forward_hook(_inject)
    print(f"[inject] hook installed on model.projector -- "
          f"will replace NPU output with GPU tensor")

    # ── Run forward ───────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            actions, result = model.predict_action_batch(
                env_obs=env_obs,
                do_sample=False,
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                max_new_tokens=None,
                mode="eval",
            )
    finally:
        handle.remove()

    npu_out = {
        "actions":       actions.detach().cpu(),
        "action_tokens": result["forward_inputs"]["action_tokens"].detach().cpu(),
        "prev_logprobs": result["prev_logprobs"].detach().cpu(),
        "prev_values":   result["prev_values"].detach().cpu(),
    }

    # ── Compare with GPU reference ─────────────────────────────────────────────
    print("\n=== injection result: NPU (GPU projector) vs GPU baseline ===")
    print(f"    (n={n} samples; NPU runs LLM only, vision from GPU golden)\n")
    keys = [k for k in ("prev_logprobs", "action_tokens", "actions", "prev_values")
            if k in npu_out and k in gpu_ref]
    diffs = [compare_tensors(gpu_ref[k], npu_out[k], name=k) for k in keys]
    print(format_diff_table(diffs))

    any_diff = any(not d.bit_exact for d in diffs)
    if not any_diff:
        print("\n[inject] CONCLUSION: bit-exact match -- NPU LLM is clean.")
        print("  Root cause confirmed: vision backbone only.")
    else:
        logprob_diff = next((d for d in diffs if d.name == "prev_logprobs"), None)
        if logprob_diff and logprob_diff.max_abs < 1e-2:
            print("\n[inject] CONCLUSION: logprob diff < 1e-2 after injection -- "
                  "NPU LLM is numerically close. Vision backbone is the primary cause.")
        else:
            print("\n[inject] CONCLUSION: logprob diff persists after injection -- "
                  "NPU LLM ALSO contributes divergence independently of the vision backbone.")
            print("  Next step: run compare_layers.py from language_model.model.layers.0 "
                  "with a fixed (GPU-injected) projector output.")

    # ── Save result ────────────────────────────────────────────────────────────
    out_path = args.out or (GOLDENS_DIR / f"openvla_oft_inject_{device_label(device)}.pt")
    save_golden({
        "schema": "openvla_oft_inject_v1",
        "device_label": device_label(device),
        "device": str(device),
        "n_samples": n,
        "layers_golden": str(args.layers_golden),
        "rlinf_golden": str(args.rlinf_golden),
        "tensors": npu_out,
        "diffs": [
            {"name": d.name, "bit_exact": d.bit_exact,
             "max_abs": d.max_abs, "mean_abs": d.mean_abs}
            for d in diffs
        ],
    }, out_path)
    print(f"\n[inject] wrote {out_path}")
    return 0 if not any_diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
