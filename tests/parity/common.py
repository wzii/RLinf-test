# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared helpers for cross-hardware (GPU/NPU) inference parity tests.

Goal
----
Given a fixed model checkpoint + a fixed batch of 32 inputs, every run of
the same script on the same hardware must produce **bit-identical** output
tensors. Two runs across hardware should match within a documented tolerance.

These helpers handle determinism setup, golden file IO, tensor hashing, and
the bit-exact / semantic comparison logic shared by the OpenVLA-OFT and Wan
parity scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

NUM_SAMPLES = 32
SEED = 1234

REPO_ROOT = Path(__file__).resolve().parents[2]
# WAN_DATASET_DIR resolves to the in-repo sample dataset shipped with the
# parity tests (so a fresh clone has everything it needs). Override with the
# env var ``WAN_DATASET_DIR`` to point at the full RLinf-Wan-LIBERO-Spatial
# checkpoint dataset for larger sweeps.
_REPO_SAMPLE_DIR = REPO_ROOT / "tests" / "parity" / "sample_dataset"
WAN_DATASET_DIR = Path(
    os.environ.get(
        "WAN_DATASET_DIR",
        str(_REPO_SAMPLE_DIR if _REPO_SAMPLE_DIR.exists() else "/workspace/RLinf-Wan-LIBERO-Spatial/dataset"),
    )
)
WAN_CKPT_DIR = Path(os.environ.get("WAN_CKPT_DIR", "/workspace/RLinf-Wan-LIBERO-Spatial"))
OPENVLA_OFT_CKPT_DIR = Path(
    os.environ.get("OPENVLA_OFT_CKPT_DIR", "/workspace/Openvla-oft-SFT-libero-spatial-traj1")
)
GOLDENS_DIR = REPO_ROOT / "tests" / "parity" / "goldens"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_determinism(seed: int = SEED) -> None:
    """Pin every randomness source we know about.

    Why: cross-hardware parity requires identical outputs across runs of the
    same script. NPU / CUDA kernel libraries pick different algorithms based
    on environment flags, and Python / NumPy / Torch each maintain their own
    RNG.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def pick_device() -> torch.device:
    """Return the best available compute device (cuda → npu → cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Short readable tag for filenames."""
    kind = device.type
    if kind == "cuda":
        return "gpu"
    return kind


# ---------------------------------------------------------------------------
# Hashing / IO
# ---------------------------------------------------------------------------


def tensor_hash(t: torch.Tensor) -> str:
    """Stable SHA256 of a tensor's bytes after moving to CPU + float32 cast.

    Why float32: bf16 NaN payloads and subnormals can be re-encoded differently
    across kernels even when the numeric value is the same. Casting to fp32
    canonicalises the representation.
    """
    arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def array_hash(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(a)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_golden(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_golden(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class TensorDiff:
    name: str
    bit_exact: bool
    allclose_strict: bool
    allclose_bf16: bool
    max_abs: float
    mean_abs: float
    shape: tuple
    dtype_a: str
    dtype_b: str
    extra: dict = field(default_factory=dict)


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    name: str,
    rtol_strict: float = 1e-5,
    atol_strict: float = 1e-5,
    rtol_bf16: float = 1e-2,
    atol_bf16: float = 1e-2,
) -> TensorDiff:
    a32 = a.detach().to("cpu", dtype=torch.float32)
    b32 = b.detach().to("cpu", dtype=torch.float32)
    if a32.shape != b32.shape:
        return TensorDiff(
            name=name,
            bit_exact=False,
            allclose_strict=False,
            allclose_bf16=False,
            max_abs=float("nan"),
            mean_abs=float("nan"),
            shape=tuple(a32.shape),
            dtype_a=str(a.dtype),
            dtype_b=str(b.dtype),
            extra={"shape_mismatch": (tuple(a32.shape), tuple(b32.shape))},
        )
    diff = (a32 - b32).abs()
    return TensorDiff(
        name=name,
        bit_exact=tensor_hash(a) == tensor_hash(b),
        allclose_strict=bool(torch.allclose(a32, b32, rtol=rtol_strict, atol=atol_strict)),
        allclose_bf16=bool(torch.allclose(a32, b32, rtol=rtol_bf16, atol=atol_bf16)),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        shape=tuple(a32.shape),
        dtype_a=str(a.dtype),
        dtype_b=str(b.dtype),
    )


def format_diff_table(diffs: list[TensorDiff]) -> str:
    header = f"{'tensor':40s} {'bit-exact':>10s} {'allclose1e-5':>13s} {'allclose1e-2':>13s} {'max_abs':>10s} {'mean_abs':>10s} {'shape':>20s}"
    rows = [header, "-" * len(header)]
    for d in diffs:
        rows.append(
            f"{d.name:40s} {str(d.bit_exact):>10s} {str(d.allclose_strict):>13s} {str(d.allclose_bf16):>13s} "
            f"{d.max_abs:>10.3e} {d.mean_abs:>10.3e} {str(d.shape):>20s}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Input collectors (deterministic, sourced from the dataset bundled with the
# RLinf-Wan-LIBERO-Spatial checkpoint so the same 32 inputs are reproducible
# on any machine that has the dataset)
# ---------------------------------------------------------------------------


def collect_libero_image_inputs(
    n: int = NUM_SAMPLES, dataset_dir: Path = WAN_DATASET_DIR
) -> dict[str, Any]:
    """Build a deterministic batch of (image, instruction) pairs.

    Source: the kir-trajectory .npy files under the Wan checkpoint dataset
    folder. We pick the first n trajectories sorted by file name, then the
    first frame from each. This means the inputs are bit-identical no matter
    what hardware you run on.
    """
    files = sorted(p for p in dataset_dir.glob("step_*_seed_*_traj_*_kir.npy"))
    if len(files) < n:
        # Fall back to seed_*_traj_*.npy single-frame files.
        files = sorted(p for p in dataset_dir.glob("seed_*_traj_*.npy"))
    if len(files) < n:
        raise FileNotFoundError(
            f"Need at least {n} trajectory files in {dataset_dir}, found {len(files)}"
        )
    files = files[:n]

    images = np.zeros((n, 256, 256, 3), dtype=np.uint8)
    instructions: list[str] = []
    for i, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        # _kir files are arrays of frame dicts; non-kir are 1-element arrays.
        if data.dtype == object and len(data) > 0 and isinstance(data[0], dict):
            frame = data[0]
        else:
            frame = data.item()
        img = frame["image"]
        if img.shape != (256, 256, 3):
            raise ValueError(f"Unexpected image shape {img.shape} from {f}")
        images[i] = img
        instructions.append(str(frame["instruction"]))

    return {
        "files": [str(f) for f in files],
        "main_images": images,                                # [N, H, W, 3] uint8
        "task_descriptions": instructions,                    # list[str]
        "wrist_images": np.zeros_like(images),                # placeholder, OFT uses 1-image mode by default
        "states": np.zeros((n, 8), dtype=np.float32),         # placeholder
        "fingerprint": array_hash(images) + ":" + hashlib.sha256("|".join(instructions).encode()).hexdigest(),
    }


def collect_wan_inputs(
    n: int = NUM_SAMPLES,
    dataset_dir: Path = WAN_DATASET_DIR,
    condition_frame_length: int = 5,
    chunk: int = 8,
) -> dict[str, Any]:
    """Build a deterministic batch suitable for the Wan video pipeline.

    For each of the n samples we mirror what `WanEnv.reset()` does on the first
    reset using `enable_kir=True`: take the kir trajectory's first frame as
    the reference frame, the next 4 frames as the recent condition frames,
    and zero-padded actions for the prediction horizon.
    """
    files = sorted(p for p in dataset_dir.glob("step_*_seed_*_traj_*_kir.npy"))
    if len(files) < n:
        raise FileNotFoundError(
            f"Need at least {n} kir trajectory files in {dataset_dir}, found {len(files)}"
        )
    files = files[:n]

    # Per-env condition frames: [N, condition_frame_length, H, W, 3] uint8.
    cond_frames = np.zeros((n, condition_frame_length, 256, 256, 3), dtype=np.uint8)
    cond_actions = np.zeros((n, condition_frame_length, 7), dtype=np.float32)
    instructions: list[str] = []

    for i, f in enumerate(files):
        traj = np.load(f, allow_pickle=True)
        if len(traj) < condition_frame_length:
            raise ValueError(f"{f}: only {len(traj)} frames, need {condition_frame_length}")
        ref_frame = traj[0]
        instructions.append(str(ref_frame["instruction"]))
        cond_frames[i, 0] = ref_frame["image"]
        for t in range(1, condition_frame_length):
            cond_frames[i, t] = traj[t]["image"]
            cond_actions[i, t] = traj[t]["delta_action"]

    # Actions sent to chunk_step: zero deltas, gripper open (-1) as the
    # reset_gripper_open=True libero path does.
    new_actions = np.zeros((n, chunk, 7), dtype=np.float32)
    new_actions[..., -1] = -1.0

    return {
        "files": [str(f) for f in files],
        "cond_frames": cond_frames,        # [N, T_cond, H, W, 3] uint8 in [0,255]
        "cond_actions": cond_actions,      # [N, T_cond, 7] float32
        "new_actions": new_actions,        # [N, chunk, 7] float32
        "task_descriptions": instructions,
        "condition_frame_length": condition_frame_length,
        "chunk": chunk,
        "fingerprint": array_hash(cond_frames) + ":" + array_hash(cond_actions) + ":" + array_hash(new_actions),
    }


# ---------------------------------------------------------------------------
# Per-sample digest used for the human-facing per-sample report.
# ---------------------------------------------------------------------------


def per_sample_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    return {
        "name": name,
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "hash": array_hash(arr),
    }


# ---------------------------------------------------------------------------
# Entry-point banner used by the runnable scripts so logs are easy to grep.
# ---------------------------------------------------------------------------


def print_env_banner(device: torch.device, extra: dict[str, Any] | None = None) -> None:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cubla_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if extra:
        info.update(extra)
    print("[parity-env] " + json.dumps(info, default=str))
