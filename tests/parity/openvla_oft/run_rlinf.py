#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""OpenVLA-OFT cross-hardware parity test (RLinf in-repo impl).

What this script does
---------------------
1. Loads the RLinf adapter `OpenVLAOFTForRLActionPrediction` from
   `rlinf.models.embodiment.openvla_oft.rlinf` against the libero-spatial
   checkpoint at ``/workspace/Openvla-oft-SFT-libero-spatial-traj1/``.
2. Builds 32 deterministic inputs (image + instruction) sourced from the
   Wan checkpoint's bundled dataset .npy files, so the inputs are bit-identical
   on any machine.
3. Runs ``predict_action_batch`` twice in **deterministic decode mode**
   (do_sample=False, argmax) and verifies:
   - bit-exact self-consistency between the two runs on this device,
   - dumps a golden file containing every interesting tensor (logits / action
     tokens / discretised actions / 7D actions / value head out / per-sample
     hashes) so it can be diffed against a run on another device with
     ``tests/parity/compare.py``.

Why deterministic decode
------------------------
The production eval uses ``do_sample=True, temperature_eval=1.6``, which calls
``torch.multinomial`` — its result depends on the RNG state of the active
device. That noise hides any kernel-level disagreement between GPU and NPU.
For a parity test we want to expose those disagreements, so we replace
sampling with argmax. The token logits we save still let you measure how the
sampling distribution would diverge.

Run on each device, compare:

    python tests/parity/openvla_oft/run_rlinf.py
    # ... transfer the resulting tests/parity/goldens/openvla_oft_rlinf_<dev>.pt
    python tests/parity/compare.py \
        tests/parity/goldens/openvla_oft_rlinf_gpu.pt \
        tests/parity/goldens/openvla_oft_rlinf_npu.pt
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

from omegaconf import OmegaConf  # noqa: E402

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
    per_sample_summary,
    save_golden,
    set_determinism,
    tensor_hash,
)


def build_cfg(model_path: Path) -> OmegaConf:
    """Mirror the eval YAML for libero-spatial+openvla-oft, with the knobs we
    care about set deterministically."""
    cfg = OmegaConf.create(
        {
            "model_path": str(model_path),
            "model_type": "openvla_oft",
            "implement_version": "rlinf",
            "precision": "bf16",
            "value_type": "action_level",
            "action_dim": 7,
            "num_action_chunks": 8,
            "use_proprio": False,
            "use_film": False,
            "unnorm_key": "libero_spatial_no_noops",
            "center_crop": True,
            "max_prompt_length": 50,
            "vocab_size": 32000,
            "hidden_size": 4096,
            "add_value_head": False,
            "image_size": [224, 224],
            "is_lora": False,
            "lora_rank": 32,
            "lora_path": None,
            "num_images_in_input": 1,
            "attn_implementation": "eager",  # avoid flash-attn drift across hw
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "policy_setup": "widowx_bridge",
        }
    )
    return cfg


def build_env_obs(inputs: dict, device: torch.device) -> dict:
    """Pack the 32-sample batch into the exact shape OpenVLA-OFT's
    `predict_action_batch` expects when called with `env_obs=...`."""
    main_images = torch.from_numpy(inputs["main_images"])  # [N, H, W, 3] uint8
    wrist_images = torch.from_numpy(inputs["wrist_images"])  # [N, H, W, 3] uint8
    states = torch.from_numpy(inputs["states"]).to(torch.float32)
    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "states": states,
        "task_descriptions": list(inputs["task_descriptions"]),
    }


def run_forward(model, env_obs) -> dict:
    """Single deterministic forward through OpenVLA-OFT.

    Returns a dict of tensors and metadata we want to capture for parity.
    """
    actions, result = model.predict_action_batch(
        env_obs=env_obs,
        do_sample=False,           # ARGMAX decode → deterministic
        temperature=1.0,           # ignored when do_sample=False
        top_k=-1,
        top_p=1.0,
        max_new_tokens=None,
        mode="eval",
    )
    # `forward_inputs["action_tokens"]` is the integer token grid (B, chunk, dim).
    out = {
        "actions": actions.detach().cpu(),                              # [B, chunk, dim] float
        "action_tokens": result["forward_inputs"]["action_tokens"].detach().cpu(),
        "prev_logprobs": result["prev_logprobs"].detach().cpu(),
        "prev_values": result["prev_values"].detach().cpu(),
        "input_ids": result["forward_inputs"]["input_ids"].detach().cpu(),
        "attention_mask": result["forward_inputs"]["attention_mask"].detach().cpu(),
        "pixel_values": result["forward_inputs"]["pixel_values"].detach().to(torch.float32).cpu(),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=OPENVLA_OFT_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=2, help="self-consistency runs")
    ap.add_argument("--fp32-attention", action="store_true",
                    help="Replace F.scaled_dot_product_attention with a manual fp32 "
                         "implementation to eliminate Ascend-vs-NVIDIA kernel divergence. "
                         "Apply on BOTH devices to get cross-hardware comparable goldens.")
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    impl_tag = "rlinf-fp32attn" if args.fp32_attention else "rlinf"
    print_env_banner(device, {"impl": impl_tag, "ckpt": str(args.model_path)})

    # Lazy import so the parity dir can be imported on machines without HF.
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    cfg = build_cfg(args.model_path)

    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    print(f"[parity] model loaded in {time.time() - t0:.1f}s")

    if args.fp32_attention:
        install_fp32_attention(model)

    inputs = collect_libero_image_inputs(n=args.num_samples)
    print(f"[parity] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs, device)

    # Run repeats and prove same-device self-consistency before we even talk
    # about cross-device parity.
    runs: list[dict] = []
    for i in range(args.repeats):
        with torch.no_grad():
            t1 = time.time()
            out = run_forward(model, env_obs)
            print(f"[parity] run {i}: {time.time() - t1:.2f}s")
        runs.append(out)

    if args.repeats > 1:
        diffs = [
            compare_tensors(runs[0][k], runs[-1][k], name=k)
            for k in runs[0].keys()
        ]
        print("\n=== self-consistency (run0 vs run-1, same hardware) ===")
        print(format_diff_table(diffs))
        any_drift = any(not d.bit_exact for d in diffs)
        if any_drift:
            print("\n[parity][WARN] same-hardware runs drift — determinism flags "
                  "did not fully pin the math. Investigate before trusting "
                  "cross-hardware diffs.")

    payload = {
        "schema": "openvla_oft_rlinf_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "cfg": OmegaConf.to_container(cfg),
        "input_fingerprint": inputs["fingerprint"],
        "input_files": inputs["files"],
        "task_descriptions": inputs["task_descriptions"],
        "input_main_image_hash": tensor_hash(
            torch.from_numpy(inputs["main_images"]).to(torch.float32)
        ),
        "tensors": runs[0],
        "tensor_hashes": {k: tensor_hash(v) for k, v in runs[0].items()},
        "per_sample": [
            per_sample_summary(f"actions[{i}]", runs[0]["actions"][i])
            for i in range(runs[0]["actions"].shape[0])
        ],
    }

    dev = device_label(device)
    suffix = "_fp32attn" if args.fp32_attention else ""
    out_path = args.out or (GOLDENS_DIR / f"openvla_oft_rlinf{suffix}_{dev}.pt")
    save_golden(payload, out_path)
    print(f"[parity] wrote {out_path}")
    for k, h in payload["tensor_hashes"].items():
        print(f"  {k}: sha256={h[:16]}…  shape={list(runs[0][k].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
