#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Wan-as-env parity test, driving the exact RLinf code path.

What this script does
---------------------
Builds the same ``WanEnv`` that the libero+wan training entry uses (env_type
``wan_wm``, env config ``env/wan_libero_spatial``), then deterministically
plays one ``chunk_step`` per sample with **fixed action sequences** so the
ONLY thing that varies between runs is the model math. Captures:

  - reward-model output for each chunk frame,
  - the predicted frame chunk (post-VAE decode) saved as float32 in [-1, 1],
  - the success / termination signals.

It uses ``num_envs == NUM_SAMPLES`` so a single ``chunk_step`` produces all
32 forward results at once. The point of the test is to spot kernel-level
divergence in the WAN diffusion + VAE forward pass, which is the long pole
between GPU and NPU.

Determinism notes
-----------------
WanVideoPipeline samples noise via ``torch.randn(..., generator=Generator("cpu").manual_seed(seed))``
in ``BasePipeline.generate_noise``, so the initial noise is hardware-
independent. The pipeline's DDIM / EulerDiscrete sampler is deterministic
given the inputs. The remaining drift sources are convolution / attention
kernels — exactly what we are trying to characterise here.

How to use across hardware
--------------------------
    python tests/parity/wan/run_rlinf_env.py
    # transfer the resulting .pt
    python tests/parity/compare.py \
        tests/parity/goldens/wan_rlinf_env_gpu.pt \
        tests/parity/goldens/wan_rlinf_env_npu.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (imported for the type annotations below)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/
os.environ.setdefault("EMBODIED_PATH", str(REPO / "examples/embodiment"))

from omegaconf import OmegaConf  # noqa: E402

from common import (  # noqa: E402
    GOLDENS_DIR,
    NUM_SAMPLES,
    WAN_CKPT_DIR,
    WAN_DATASET_DIR,
    array_hash,
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


def build_wan_env_cfg(ckpt_dir: Path, num_envs: int) -> OmegaConf:
    """Replicate ``env/wan_libero_spatial.yaml`` with paths resolved to the
    real checkpoint dir and ``total_num_envs`` set to NUM_SAMPLES so we get
    32 simultaneous forwards out of one chunk_step."""
    cfg = OmegaConf.create(
        {
            "env_type": "wan_wm",
            "task_suite_name": "libero_spatial",
            "wm_env_type": "libero",
            "total_num_envs": num_envs,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": 240,
            "max_episode_steps": 240,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": False,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": False,
            "specific_reset_id": None,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/parity_wan_video",
            },
            "enable_offload": False,
            "wan_wm_hf_ckpt_path": str(ckpt_dir),
            "VAE_path": str(ckpt_dir / "Wan2.2_VAE.pth"),
            "model_path": str(ckpt_dir / "model-00001.safetensors"),
            "enable_kir": True,
            "initial_image_path": str(WAN_DATASET_DIR),
            "num_inference_steps": 5,
            "chunk": 8,
            "condition_frame_length": 5,
            "image_size": [256, 256],
            "num_frames": 13,
            "reward_model": {
                "type": "ResnetRewModel",
                "from_pretrained": str(ckpt_dir / "resnet_rm.pth"),
            },
        }
    )
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=WAN_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "rlinf-wan-env", "ckpt": str(args.ckpt_dir)})

    # WanEnv builds its pipeline on cuda:0 unconditionally in its constructor.
    # We let it; the parity tests are about whether the math agrees on
    # whatever device the harness chooses at the top level.
    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    cfg = build_wan_env_cfg(args.ckpt_dir, args.num_samples)
    t0 = time.time()
    env = WanEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)
    print(f"[parity] env built in {time.time() - t0:.1f}s")

    # Pin reset state ids to [0, num_envs) so the dataset frames feeding the
    # pipeline are byte-identical to the ones in collect_wan_inputs's
    # fingerprint.
    env.reset_state_ids = torch.arange(args.num_samples, device=env.device)
    env._is_start = True

    t1 = time.time()
    obs0, _ = env.reset()
    print(f"[parity] env reset in {time.time() - t1:.1f}s; current_obs shape="
          f"{tuple(env.current_obs.shape)}")

    # Fixed action: gripper open (-1), all other dims zero. Same convention as
    # the reset_gripper_open=True libero path.
    actions = np.zeros((args.num_samples, cfg.chunk, 7), dtype=np.float32)
    actions[..., -1] = -1.0

    runs: list[dict] = []
    # Snapshot the full env state including the bits get_state() omits so each
    # repeat restarts from an identical place. ``image_queue`` and
    # ``condition_action`` are mutated by ``_infer_next_chunk_frames``; without
    # restoring them the second run sees post-mutation inputs and disagrees.
    saved_state = {
        "current_obs": env.current_obs.detach().clone(),
        "image_queue": [
            [f.detach().clone() for f in env.image_queue[i]] for i in range(env.num_envs)
        ],
        "condition_action": env.condition_action.detach().clone(),
        "task_descriptions": list(env.task_descriptions),
        "elapsed_steps": env.elapsed_steps,
        "prev_step_reward": env.prev_step_reward.detach().clone(),
        "_is_start": env._is_start,
        "generator_state": env._generator.get_state().clone(),
    }

    for i in range(args.repeats):
        env.current_obs = saved_state["current_obs"].clone().to(env.device)
        env.image_queue = [
            [f.clone() for f in saved_state["image_queue"][k]]
            for k in range(env.num_envs)
        ]
        env.condition_action = saved_state["condition_action"].clone()
        env.task_descriptions = list(saved_state["task_descriptions"])
        env.elapsed_steps = saved_state["elapsed_steps"]
        env.prev_step_reward = saved_state["prev_step_reward"].clone().to(env.device)
        env._is_start = saved_state["_is_start"]
        env._generator.set_state(saved_state["generator_state"].clone())

        set_determinism()  # repin RNG before each run
        t2 = time.time()
        new_obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
        print(f"[parity] chunk_step run {i}: {time.time() - t2:.2f}s")
        runs.append(
            {
                "rewards": rewards.detach().cpu(),
                "current_obs": env.current_obs.detach().to(torch.float32).cpu(),
                "terminations": terminations.detach().cpu(),
                "truncations": truncations.detach().cpu(),
                "main_images_after": new_obs[0]["main_images"].detach().cpu(),
            }
        )

    if args.repeats > 1:
        diffs = [compare_tensors(runs[0][k], runs[-1][k], name=k) for k in runs[0]]
        print("\n=== self-consistency ===")
        print(format_diff_table(diffs))

    payload = {
        "schema": "wan_rlinf_env_v1",
        "device_label": device_label(device),
        "device": str(device),
        "num_samples": args.num_samples,
        "cfg": OmegaConf.to_container(cfg),
        "actions_in": actions,
        "actions_in_hash": array_hash(actions),
        "tensors": runs[0],
        "tensor_hashes": {k: tensor_hash(v) for k, v in runs[0].items()},
        "per_sample": [
            per_sample_summary(f"rewards[{i}]", runs[0]["rewards"][i])
            for i in range(runs[0]["rewards"].shape[0])
        ],
    }
    out_path = args.out or (
        GOLDENS_DIR / f"wan_rlinf_env_{device_label(device)}.pt"
    )
    save_golden(payload, out_path)
    print(f"[parity] wrote {out_path}")
    for k, h in payload["tensor_hashes"].items():
        print(f"  {k}: sha256={h[:16]}…  shape={list(runs[0][k].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
