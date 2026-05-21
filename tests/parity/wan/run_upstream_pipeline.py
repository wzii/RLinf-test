#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Wan parity test driving DiffSynth's WanVideoPipeline directly.

What this isolates
------------------
``run_rlinf_env.py`` runs the full ``WanEnv``: image normalisation, queue
management, reward model on top. This script calls ``WanVideoPipeline``
straight, with the exact same inputs the env would have constructed. If
``run_rlinf_env.py`` shows a discrepancy and this one does not, the
discrepancy is in the RLinf env wrapper. If both show the same discrepancy,
it is in the upstream WAN forward (diffsynth code path).

Inputs are produced by ``common.collect_wan_inputs`` — deterministic, sourced
from the checkpoint's bundled kir dataset, identical across machines.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/

from PIL import Image  # noqa: E402

from common import (  # noqa: E402
    GOLDENS_DIR,
    NUM_SAMPLES,
    WAN_CKPT_DIR,
    array_hash,
    collect_wan_inputs,
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


def build_pipeline(ckpt_dir: Path, device: torch.device):
    """Same construction as ``WanEnv._build_pipeline`` but with offload_device
    set to ``device`` so we don't bounce weights through CPU between runs."""
    # Lazy import: diffsynth + groot get pulled in here, not at module load
    # (so the file is importable in any venv).
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=[
            ModelConfig(path=str(ckpt_dir / "model-00001.safetensors"),
                        offload_device=str(device)),
            ModelConfig(path=str(ckpt_dir / "Wan2.2_VAE.pth"),
                        offload_device=str(device)),
        ],
    )
    pipe.dit.to(device)
    pipe.vae.to(device)
    return pipe


def to_pil_list(cond_frames_np: np.ndarray) -> tuple[list[Image.Image], list[list[Image.Image]]]:
    """Mirror WanEnv._infer_next_chunk_frames PIL conversion.

    Inputs are uint8 [N, T_cond, H, W, 3]; outputs are
    - input_image: per-env first frame as PIL,
    - input_image4: per-env last-4 frames as PIL list."""
    input_image: list[Image.Image] = []
    input_image4: list[list[Image.Image]] = []
    for env_idx in range(cond_frames_np.shape[0]):
        frames = [Image.fromarray(cond_frames_np[env_idx, t]) for t in range(cond_frames_np.shape[1])]
        input_image.append(frames[0])
        input_image4.append(frames[-4:])
    return input_image, input_image4


def run_pipeline_once(pipe, inputs: dict, device: torch.device) -> dict:
    """One forward through WanVideoPipeline with all inputs pinned."""
    cond_frames = inputs["cond_frames"]
    cond_actions = torch.from_numpy(inputs["cond_actions"]).to(device=device, dtype=torch.bfloat16)
    new_actions = torch.from_numpy(inputs["new_actions"]).to(device=device, dtype=torch.bfloat16)
    # WanEnv applies retain_action=True: prepend cond_actions to new_actions.
    actions_tensor = torch.cat([cond_actions, new_actions], dim=1)

    input_image, input_image4 = to_pil_list(cond_frames)
    kwargs = dict(
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

    out = pipe(**kwargs)
    # ``out`` is a list-of-lists of PIL images, env-major. Convert to a tensor
    # in [-1, 1] range to match the env's internal representation.
    arrays = []
    for env_frames in out:
        env_arr = np.stack([np.asarray(im) for im in env_frames], axis=0).astype(np.float32)
        env_arr = (env_arr / 255.0) * 2.0 - 1.0
        arrays.append(env_arr)
    video = torch.from_numpy(np.stack(arrays, axis=0))  # [N, T, H, W, 3]
    video = video.permute(0, 4, 1, 2, 3).contiguous()    # [N, 3, T, H, W]
    return {"video": video.to(torch.float32).cpu()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=WAN_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "wan-upstream", "ckpt": str(args.ckpt_dir)})

    t0 = time.time()
    pipe = build_pipeline(args.ckpt_dir, device)
    print(f"[parity] pipeline built in {time.time() - t0:.1f}s")

    inputs = collect_wan_inputs(n=args.num_samples)
    print(f"[parity] inputs fingerprint = {inputs['fingerprint']}")

    runs: list[dict] = []
    for i in range(args.repeats):
        set_determinism()
        t1 = time.time()
        out = run_pipeline_once(pipe, inputs, device)
        print(f"[parity] pipeline run {i}: {time.time() - t1:.2f}s")
        runs.append(out)

    if args.repeats > 1:
        diffs = [compare_tensors(runs[0][k], runs[-1][k], name=k) for k in runs[0]]
        print("\n=== self-consistency ===")
        print(format_diff_table(diffs))

    payload = {
        "schema": "wan_upstream_pipeline_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "input_fingerprint": inputs["fingerprint"],
        "input_files": inputs["files"],
        "task_descriptions": inputs["task_descriptions"],
        "cond_frames_hash": array_hash(inputs["cond_frames"]),
        "cond_actions_hash": array_hash(inputs["cond_actions"]),
        "new_actions_hash": array_hash(inputs["new_actions"]),
        "tensors": runs[0],
        "tensor_hashes": {k: tensor_hash(v) for k, v in runs[0].items()},
        "per_sample": [
            per_sample_summary(f"video[{i}]", runs[0]["video"][i])
            for i in range(runs[0]["video"].shape[0])
        ],
    }
    out_path = args.out or (
        GOLDENS_DIR / f"wan_upstream_pipeline_{device_label(device)}.pt"
    )
    save_golden(payload, out_path)
    print(f"[parity] wrote {out_path}")
    for k, h in payload["tensor_hashes"].items():
        print(f"  {k}: sha256={h[:16]}…  shape={list(runs[0][k].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
