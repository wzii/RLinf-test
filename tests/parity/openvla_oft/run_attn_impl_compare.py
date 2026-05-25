#!/usr/bin/env python3
"""Compare different attention implementations on the same device.

Loads the model four times with different attn_implementation settings and
compares prev_logprobs / actions to quantify the numerical spread WITHIN a
single device. This answers: "how much do different attention backends
disagree on GPU itself?" -- the cross-device (GPU vs NPU) divergence should
ideally be no worse than the worst within-GPU implementation gap.

Implementations tested:
  sdpa   : F.scaled_dot_product_attention (default in HF transformers 4.40+)
  eager  : manual LlamaAttention with explicit bf16 softmax
  fp32   : our patch -- manual fp32 q@k^T and weights@v
  flash2 : Flash Attention 2 (if available)

Usage
-----
    python tests/parity/openvla_oft/run_attn_impl_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    NUM_SAMPLES,
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    compare_tensors,
    format_diff_table,
    install_fp32_attention,
    pick_device,
    print_env_banner,
    set_determinism,
)
from run_rlinf import build_cfg, build_env_obs, run_forward  # noqa: E402


def load_and_run(attn_impl: str, device, env_obs) -> dict:
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model
    cfg = build_cfg(OPENVLA_OFT_CKPT_DIR)
    cfg["attn_implementation"] = attn_impl if attn_impl != "fp32" else "sdpa"

    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"  [{attn_impl}] loaded in {time.time()-t0:.1f}s", end="", flush=True)

    # Check what class was actually used
    attn_classes = {type(m).__name__ for n, m in model.named_modules()
                    if "self_attn" in n and "." not in n.replace("language_model.model.layers.0.self_attn", "")}
    llm_cls = next((type(m).__name__ for n, m in model.named_modules()
                    if n == "language_model.model.layers.0.self_attn"), "?")
    print(f"  (LLM attn class: {llm_cls})", end="")

    # Save original forward BEFORE patching so we can restore it afterward.
    # install_fp32_attention patches the CLASS (not the instance), so without
    # restoring it the next model load would also run fp32 attention.
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
    _orig_sdpa_fwd = LlamaSdpaAttention.forward

    if attn_impl == "fp32":
        install_fp32_attention(model)

    set_determinism()
    t1 = time.time()
    with torch.no_grad():
        out = run_forward(model, env_obs)
    print(f"  forward: {time.time()-t1:.2f}s")

    # Restore the original SDPA forward so subsequent runs are not affected.
    LlamaSdpaAttention.forward = _orig_sdpa_fwd

    del model
    torch.cuda.empty_cache()
    return out


def main():
    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "attn-impl-compare"})

    inputs = collect_libero_image_inputs(n=NUM_SAMPLES)
    print(f"inputs fingerprint = {inputs['fingerprint']}\n")
    env_obs = build_env_obs(inputs, device)

    impls = ["sdpa", "eager", "fp32"]
    try:
        import flash_attn  # noqa: F401
        impls.append("flash_attention_2")
    except ImportError:
        print("flash_attn not installed, skipping flash2\n")

    results = {}
    for impl in impls:
        print(f"\n--- {impl} ---")
        try:
            results[impl] = load_and_run(impl, device, env_obs)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Print pairwise comparison table
    ref = "sdpa"
    if ref not in results:
        ref = list(results.keys())[0]

    print(f"\n{'='*70}")
    print(f"Pairwise comparison vs {ref} (prev_logprobs max_abs)")
    print(f"{'='*70}")

    for impl, out in results.items():
        if impl == ref:
            continue
        print(f"\n--- {ref} vs {impl} ---")
        keys = ["prev_logprobs", "action_tokens", "actions"]
        diffs = [compare_tensors(results[ref][k], out[k], name=k)
                 for k in keys if k in results[ref] and k in out]
        print(format_diff_table(diffs))


if __name__ == "__main__":
    main()
