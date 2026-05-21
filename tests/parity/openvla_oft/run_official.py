#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""OpenVLA-OFT parity test using the *official-flavoured* RLinf wrapper.

This drives the loader at ``rlinf.models.embodiment.openvla_oft.official``,
which mirrors the upstream OpenVLA-OFT HF code path (full
``AutoModelForVision2Seq.register`` + processor) rather than the trimmed
RLinf wrapper. Pairing this script's output with ``run_rlinf.py``'s output on
the same device tells you whether the two adapters diverge **independent**
of the GPU/NPU question.

Inputs and decoding are kept identical to ``run_rlinf.py`` so per-sample
outputs are directly comparable. See that script's docstring for the
motivation behind argmax decoding.
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
    per_sample_summary,
    pick_device,
    print_env_banner,
    save_golden,
    set_determinism,
    tensor_hash,
)


def build_cfg(model_path: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "model_type": "openvla_oft",
            "implement_version": "official",
            "precision": "bf16",
            "value_type": "action_level",
            "action_dim": 7,
            "num_action_chunks": 8,
            "proprio_dim": 8,
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
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "policy_setup": "widowx_bridge",
        }
    )


def build_env_obs(inputs: dict) -> dict:
    return {
        "main_images": torch.from_numpy(inputs["main_images"]),
        "wrist_images": torch.from_numpy(inputs["wrist_images"]),
        "states": torch.from_numpy(inputs["states"]).to(torch.float32),
        "task_descriptions": list(inputs["task_descriptions"]),
    }


def run_forward(model, env_obs) -> dict:
    actions, result = model.predict_action_batch(
        env_obs=env_obs,
        do_sample=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        max_new_tokens=None,
        mode="eval",
    )
    out = {
        "actions": actions.detach().cpu(),
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
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "official", "ckpt": str(args.model_path)})

    from rlinf.models.embodiment.openvla_oft.official import get_model

    cfg = build_cfg(args.model_path)
    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"[parity] model loaded in {time.time() - t0:.1f}s")

    inputs = collect_libero_image_inputs(n=args.num_samples)
    print(f"[parity] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs)

    runs = []
    for i in range(args.repeats):
        with torch.no_grad():
            t1 = time.time()
            out = run_forward(model, env_obs)
            print(f"[parity] run {i}: {time.time() - t1:.2f}s")
        runs.append(out)

    if args.repeats > 1:
        diffs = [compare_tensors(runs[0][k], runs[-1][k], name=k) for k in runs[0]]
        print("\n=== self-consistency ===")
        print(format_diff_table(diffs))

    payload = {
        "schema": "openvla_oft_official_v1",
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
    out_path = args.out or (
        GOLDENS_DIR / f"openvla_oft_official_{device_label(device)}.pt"
    )
    save_golden(payload, out_path)
    print(f"[parity] wrote {out_path}")
    for k, h in payload["tensor_hashes"].items():
        print(f"  {k}: sha256={h[:16]}…  shape={list(runs[0][k].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
