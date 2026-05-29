#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""OpenVLA-OFT cross-hardware msprobe per-API dump runner.

This is the parity-path equivalent of the eval-side msprobe dump in
``MultiStepRolloutWorker.evaluate``. The eval-side version goes through
Ray + LIBERO + MuJoCo rendering -- which means the *input* the model sees
on GPU and NPU is not byte-identical (EGL vs OSMesa renderer, x86 vs
aarch64 FPU on MuJoCo physics, etc.), so per-API stat differences pile
up before the attention path ever runs.

This script bypasses all of that:

* Inputs come from ``collect_libero_image_inputs`` (the same loader used
  by ``run_rlinf.py``); they read the Wan checkpoint's bundled ``.npy``
  files, so they are byte-identical on any host.
* No Ray, no MuJoCo, no env worker -- the model is loaded directly and
  ``predict_action_batch`` is called in a plain ``with torch.no_grad()``.
* 32 samples are split into ``--num-chunks`` chunks (default 4) of 8.
  Each chunk gets its own ``debugger.start / stop / step`` cycle, so the
  dump contains ``--num-chunks`` separate steps. This is mildly larger
  than the single-step eval dump and gives msprobe enough material to
  expose shape-dependent kernel divergences.

Usage
-----
::

    # Baseline -- whatever attention the model loads (sdpa here).
    python tests/parity/openvla_oft/run_rlinf_msprobe.py \\
        --mode no_patch \\
        --dump-path tests/parity/dumps/gpu_parity/no_patch/

    # Patched -- manual decomposed attention in bf16.
    python tests/parity/openvla_oft/run_rlinf_msprobe.py \\
        --mode bf16_patch \\
        --dump-path tests/parity/dumps/gpu_parity/bf16_patch/

    # Patched -- manual decomposed attention in fp32.
    python tests/parity/openvla_oft/run_rlinf_msprobe.py \\
        --mode fp32_patch \\
        --dump-path tests/parity/dumps/gpu_parity/fp32_patch/

Run on both GPU and NPU hosts with matching ``--mode`` and ``--num-chunks``,
then diff ``dump.json`` per step with ``msprobe compare``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/

from omegaconf import OmegaConf  # noqa: E402

from common import (  # noqa: E402
    NUM_SAMPLES,
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    device_label,
    pick_device,
    print_env_banner,
    set_determinism,
)


def build_cfg(model_path: Path, attn_implementation: str) -> OmegaConf:
    """Same shape as ``run_rlinf.py``'s build_cfg, but the attn impl is a
    knob so we can hold it identical across no_patch/bf16_patch runs."""
    return OmegaConf.create({
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
        # "sdpa" lets msprobe see F.scaled_dot_product_attention as a single
        # high-level op (and on NPU it lets torch_npu's PrivateUse1 SDPA hook
        # take over). Patched modes overwrite the SDPA forward anyway, so this
        # only matters for no_patch.
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "policy_setup": "widowx_bridge",
    })


def build_env_obs(inputs: dict) -> dict:
    """Pack the N-sample golden batch into the dict shape OpenVLA-OFT's
    ``predict_action_batch`` expects."""
    return {
        "main_images": torch.from_numpy(inputs["main_images"]),     # [N,H,W,3] u8
        "wrist_images": torch.from_numpy(inputs["wrist_images"]),   # [N,H,W,3] u8
        "states": torch.from_numpy(inputs["states"]).to(torch.float32),
        "task_descriptions": list(inputs["task_descriptions"]),
    }


def slice_env_obs(env_obs: dict, start: int, end: int) -> dict:
    """Return a view of env_obs over samples [start:end]."""
    return {
        "main_images": env_obs["main_images"][start:end],
        "wrist_images": env_obs["wrist_images"][start:end],
        "states": env_obs["states"][start:end],
        "task_descriptions": env_obs["task_descriptions"][start:end],
    }


def install_patch(mode: str) -> None:
    """Apply (or skip) the fp32_attn_loader patch in the requested compute
    dtype. The loader reads PARITY_ATTN_DTYPE / PARITY_DISABLE_FP32_ATTN at
    its ``_install`` call time -- importing it here triggers that call."""
    if mode == "no_patch":
        os.environ["PARITY_DISABLE_FP32_ATTN"] = "1"
    elif mode == "bf16_patch":
        os.environ.pop("PARITY_DISABLE_FP32_ATTN", None)
        os.environ["PARITY_ATTN_DTYPE"] = "bf16"
    elif mode == "fp32_patch":
        os.environ.pop("PARITY_DISABLE_FP32_ATTN", None)
        os.environ["PARITY_ATTN_DTYPE"] = "fp32"
    else:
        raise ValueError(f"unknown --mode {mode!r}")
    # Import the parity loader to actually fire the patch. (Re-import is
    # idempotent thanks to the loader's marker attribute.)
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/parity/
    if "fp32_attn_loader" in sys.modules:
        importlib.reload(sys.modules["fp32_attn_loader"])
    else:
        import fp32_attn_loader  # noqa: F401  -- side-effect: patches LlamaSdpaAttention


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=OPENVLA_OFT_CKPT_DIR)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                    help=f"Total samples to forward (default {NUM_SAMPLES}).")
    ap.add_argument("--num-chunks", type=int, default=4,
                    help="Number of dump steps. Samples are split evenly. "
                         "Default 4 => 4 steps of 8 samples each from the "
                         "32-sample fixture.")
    ap.add_argument("--mode", choices=["no_patch", "bf16_patch", "fp32_patch"],
                    default="no_patch",
                    help="no_patch = let model use its declared attn impl; "
                         "bf16_patch / fp32_patch = monkey-patch LlamaSdpaAttention "
                         "with a manual decomposed implementation at that compute dtype.")
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"],
                    help="HF attn impl. Only affects no_patch.")
    ap.add_argument("--dump-path", type=Path, required=True,
                    help="msprobe dump directory. Each chunk produces stepN/rankR/.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.num_samples % args.num_chunks != 0:
        raise SystemExit(
            f"--num-samples ({args.num_samples}) must be divisible by "
            f"--num-chunks ({args.num_chunks})"
        )
    chunk_size = args.num_samples // args.num_chunks

    set_determinism(seed=args.seed)
    device = pick_device()
    print_env_banner(device, {
        "script": "run_rlinf_msprobe",
        "mode": args.mode,
        "attn_impl": args.attn_implementation,
        "samples": args.num_samples,
        "chunks": args.num_chunks,
        "chunk_size": chunk_size,
        "dump_path": str(args.dump_path),
    })

    # Apply the requested patch BEFORE the model is built, so when LLaMA
    # subclasses are instantiated they pick up the patched forward.
    install_patch(args.mode)

    # Lazy import so this script can be discovered without HF / torch_npu.
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model

    cfg = build_cfg(args.model_path, args.attn_implementation)

    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    print(f"[parity-msprobe] model loaded in {time.time() - t0:.1f}s")

    inputs = collect_libero_image_inputs(n=args.num_samples)
    print(f"[parity-msprobe] inputs fingerprint = {inputs['fingerprint']}")
    env_obs = build_env_obs(inputs)

    # msprobe lives in the same process; configure here so dump_path is
    # honoured without env var plumbing.
    from msprobe.pytorch import PrecisionDebugger
    args.dump_path.mkdir(parents=True, exist_ok=True)
    debugger = PrecisionDebugger(
        task="statistics",
        dump_path=str(args.dump_path),
    )

    sample_kwargs = {
        "do_sample": False,        # argmax decode -> no multinomial RNG
        "temperature": 1.0,         # ignored when do_sample=False
        "top_k": -1,
        "top_p": 1.0,
        "max_new_tokens": None,
        "mode": "eval",
    }

    for chunk_idx in range(args.num_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size
        chunk_obs = slice_env_obs(env_obs, s, e)
        t1 = time.time()
        debugger.start(model=model)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(env_obs=chunk_obs, **sample_kwargs)
        debugger.stop()
        debugger.step()
        print(f"[parity-msprobe] chunk {chunk_idx} samples[{s}:{e}] "
              f"actions={tuple(actions.shape)} in {time.time() - t1:.2f}s")

    print(f"[parity-msprobe] done. dumps in {args.dump_path}")
    print(f"[parity-msprobe] device_label={device_label(device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
