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

# Total number of parity samples (test coverage). This is NOT the per-forward
# batch -- see WAN_MICRO_BATCH below, which splits each Wan forward into smaller
# chunks so the NPU host's peak memory stays low without dropping coverage.
NUM_SAMPLES = int(os.environ.get("PARITY_NUM_SAMPLES", "32"))
# Per-forward micro-batch for the Wan pipeline. The total NUM_SAMPLES samples are
# processed WAN_MICRO_BATCH at a time and the results concatenated, so the golden
# is identical in coverage to a single big batch but each forward only holds a
# few samples in memory. Set <=0 to disable (one forward over everything).
WAN_MICRO_BATCH = int(os.environ.get("PARITY_WAN_MICRO_BATCH", "4"))
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
# fp32 attention patch
# ---------------------------------------------------------------------------


def install_fp32_attention(model) -> None:
    """Patch every LLaMA attention module in the model to compute attention in fp32.

    Root cause: the SDPA backend calls a device FA kernel (cuDNN on NVIDIA,
    aclnn_fa on Ascend). The two kernels accumulate bfloat16 matmuls differently,
    producing ~0.1-0.5 relL2 attention output divergence that cascades through
    all LLM layers.

    Supports two transformers layouts:

    * transformers ≤ 4.45: ``LlamaSdpaAttention`` subclass with its own
      ``forward``; we replace that method with a manual fp32 version.

    * transformers ≥ 4.46: the three subclasses were merged into a single
      ``LlamaAttention``. Attention dispatch goes through the module-level
      ``eager_attention_forward`` / ``sdpa_attention_forward`` functions in
      ``transformers.models.llama.modeling_llama``. We replace
      ``sdpa_attention_forward`` with a fp32 implementation so all
      ``LlamaAttention`` instances that would use SDPA instead use fp32.

    Overhead: ~15-20 % extra memory for the fp32 intermediates; no change
    to parameter storage or linear-proj arithmetic.
    """
    import math as _math
    import torch.nn.functional as _F

    # ── transformers ≤ 4.45: LlamaSdpaAttention subclass exists ─────────────
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaSdpaAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )

        def _fp32_forward_old(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position=None,
            **kwargs,
        ):
            if output_attentions:
                return super(LlamaSdpaAttention, self).forward(
                    hidden_states=hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position,
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads,           self.head_dim).transpose(1, 2)
            key_states   = key_states  .view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            _pkv = getattr(self, "past_key_value", past_key_value)
            if _pkv is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = _pkv.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states   = repeat_kv(key_states,   self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Preserve model-specific padding-mask transformation.
            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
                D = causal_mask.shape[-1]
                last_row = causal_mask[:, :, -1, :].clone()
                causal_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)

            q = query_states.float()
            k = key_states.float()
            v = value_states.float()
            scale = 1.0 / _math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask.float()
            attn_weights = _F.softmax(attn_weights, dim=-1)
            if self.training and self.attention_dropout > 0.0:
                attn_weights = _F.dropout(attn_weights, p=self.attention_dropout)
            attn_output = torch.matmul(attn_weights, v).to(query_states.dtype)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

        LlamaSdpaAttention.forward = _fp32_forward_old
        n = sum(1 for m in model.modules() if type(m).__name__ == "LlamaSdpaAttention")
        print(f"[parity] install_fp32_attention: patched {n} LlamaSdpaAttention modules "
              f"(transformers ≤4.45)")
        return

    except ImportError:
        pass  # fall through to ≥4.46 path

    # ── transformers ≥ 4.46: unified LlamaAttention with sdpa_attention_forward ─
    try:
        from transformers.models.llama import modeling_llama as _llama_mod
        from transformers.models.llama.modeling_llama import LlamaAttention

        # Locate the module-level sdpa dispatch function.  In 4.46 it is called
        # ``sdpa_attention_forward``; in some Ascend-patched builds it may be
        # named differently.  We try sdpa first, then eager as fallback.
        _fn_name = None
        for candidate in ("sdpa_attention_forward", "eager_attention_forward"):
            if hasattr(_llama_mod, candidate):
                _fn_name = candidate
                break

        if _fn_name is None:
            print("[parity][WARN] install_fp32_attention: neither sdpa_attention_forward "
                  "nor eager_attention_forward found in modeling_llama; patch skipped")
            return

        def _fp32_attn_fn(module, query, key, value, attention_mask,
                          scaling, dropout=0.0, **kwargs):
            """Drop-in fp32 replacement for sdpa/eager_attention_forward."""
            q, k, v = query.float(), key.float(), value.float()
            attn_w = torch.matmul(q, k.transpose(-2, -1)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key.shape[-2]]
                attn_w = attn_w + causal_mask.float()
            attn_w = _F.softmax(attn_w, dim=-1)
            if module.training and getattr(module, "attention_dropout", 0.0) > 0.0:
                attn_w = _F.dropout(attn_w, p=module.attention_dropout)
            out = torch.matmul(attn_w, v).to(query.dtype)
            return out, attn_w

        setattr(_llama_mod, _fn_name, _fp32_attn_fn)
        n = sum(1 for m in model.modules() if isinstance(m, LlamaAttention))
        print(f"[parity] install_fp32_attention: replaced {_fn_name} with fp32 kernel "
              f"({n} LlamaAttention modules, transformers ≥4.46)")

    except ImportError:
        print("[parity][WARN] install_fp32_attention: LlamaAttention not found; patch skipped")


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
    # Seed NPU RNG (Ascend/torch_npu). Without this, torch.npu ops that draw
    # from the device RNG (e.g. attention softmax internal stochastic rounding)
    # give different results across calls even in eval() mode.
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except Exception:
        pass
    # Ascend CANN determinism flag: prevents the FA kernel from using
    # non-deterministic atomics. No-op on non-Ascend hosts.
    os.environ.setdefault("ACLNN_DETERMINISTIC", "1")


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
# Cross-architecture fixed noise
# ---------------------------------------------------------------------------

# Wan's ``BasePipeline.generate_noise`` draws ``torch.randn`` through a CPU
# generator on the theory that "CPU => hardware-independent". That is only half
# true: the MT19937 integer stream IS portable, but ``torch.randn``'s Box-Muller
# transform uses transcendental ops (log/cos/sqrt) whose libm / vectorised
# (sleef) results differ by 1-2 ULP between x86 (the GPU host) and aarch64 (the
# NPU host). The fp32->bf16 cast washes out most -- but not all -- of that, which
# is enough to break the "inputs are byte-identical" premise the whole parity
# test rests on. So we generate the noise ONCE (fp32, CPU, seeded), persist it to
# goldens/, and load the exact same bytes on every machine. fp32 tensors are
# byte-portable across x86/aarch64 (both little-endian IEEE-754), so the noise
# is now a fixed input artifact just like the dataset frames.
WAN_FIXED_NOISE_PATH = GOLDENS_DIR / "wan_fixed_noise.pt"


def _probe_pipe_target_device(pipe) -> torch.device:
    """Resolve the pipe's *actual* compute device from a model parameter.

    ``pipe.device`` can be a stale string (e.g. ``"cuda:0"``) even when the
    model components have been moved elsewhere -- RLinf's ``WanEnv._build_pipeline``
    hardcodes ``cuda:0`` and NPU adaptations may patch the submodules but leave
    ``pipe.device`` untouched. Reading any parameter's ``.device`` gives the
    truth without risking ``torch.cuda._lazy_init`` on a CUDA-disabled build.
    """
    for attr in ("dit", "denoising_model", "transformer", "text_encoder", "vae"):
        m = getattr(pipe, attr, None)
        if m is None or not hasattr(m, "parameters"):
            continue
        try:
            return next(iter(m.parameters())).device
        except (StopIteration, AttributeError, TypeError):
            continue
    # Fallback: trust pipe.device if the probe failed (CPU-only debug runs).
    return torch.device(getattr(pipe, "device", "cpu"))


def pin_pipe_device(pipe) -> torch.device:
    """Set ``pipe.device`` to where the model parameters actually live.

    Many diffsynth pipeline methods (``preprocess_image``, ``preprocess_video``,
    the original ``generate_noise``, ...) do ``x.to(device=self.device)``.
    RLinf's ``WanEnv._build_pipeline`` hardcodes ``pipe.device='cuda:0'``, and
    NPU adaptations typically move the model sub-modules to npu but leave the
    pipe attribute alone. On a CUDA-disabled NPU torch build, any of those
    ``.to(device='cuda:0')`` calls then triggers ``torch.cuda._lazy_init()``
    and crashes with::

        AssertionError: Torch not compiled with CUDA enabled

    Setting ``pipe.device`` to the probed real device fixes every downstream
    consumer in one shot. The proper home for this fix is RLinf's
    ``WanEnv._build_pipeline`` (or the user's NPU adapter for it), but the
    parity test must work without depending on that.

    Returns the resolved device for logging.
    """
    probed = _probe_pipe_target_device(pipe)
    current = getattr(pipe, "device", None)
    if str(current) != str(probed):
        try:
            pipe.device = probed
            print(f"[parity] pipe.device pinned: {current} -> {probed}")
        except (AttributeError, TypeError) as exc:
            print(f"[parity][WARN] could not set pipe.device "
                  f"({current} -> {probed}): {exc}")
    return probed


def install_fixed_noise(pipe, path: Path = WAN_FIXED_NOISE_PATH) -> None:
    """Monkeypatch ``pipe.generate_noise`` to return byte-identical noise across
    architectures.

    First machine to run generates and saves the noise; every later run --
    crucially the NPU run -- loads the same bytes. Keyed by ``(seed, shape)`` so
    multiple configs in one process each get their own stable noise. Transfer
    ``goldens/wan_fixed_noise.pt`` to the NPU host alongside the goldens.

    Also pins ``pipe.device`` (via :func:`pin_pipe_device`) so other diffsynth
    methods that read it (``preprocess_image``, ``preprocess_video``, ...) do
    not crash with ``cuda:0`` on an NPU build.
    """
    pin_pipe_device(pipe)
    cache: dict[tuple, torch.Tensor] = {}
    if path.exists():
        cache = torch.load(path, map_location="cpu")
        print(f"[parity] fixed noise: loaded {len(cache)} tensor(s) from {path}")

    def _generate_noise(shape, seed=None, rand_device="cpu",
                        rand_torch_dtype=torch.float32, device=None,
                        torch_dtype=None, **_):
        key = (seed, tuple(shape))
        if key not in cache:
            gen = torch.Generator("cpu")
            if seed is not None:
                gen.manual_seed(seed)
            cache[key] = torch.randn(tuple(shape), generator=gen,
                                     device="cpu", dtype=torch.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, path)
            print(f"[parity] fixed noise: generated + saved {key} -> {path}")
        else:
            print(f"[parity] fixed noise: reused {key}")
        noise = cache[key]
        target_device = device if device is not None else _probe_pipe_target_device(pipe)
        target_dtype = torch_dtype or getattr(pipe, "torch_dtype", torch.float32)
        return noise.to(dtype=target_dtype, device=target_device)

    pipe.generate_noise = _generate_noise


def install_microbatch(pipe, micro_batch: int) -> None:
    """Split each ``pipe(**kwargs)`` forward into ``micro_batch``-sized chunks.

    The total batch is processed a few samples at a time and the per-env outputs
    concatenated, so the golden covers the same samples as a single big batch but
    the NPU only ever holds ``micro_batch`` Wan forwards in memory at once.

    ``pipe(**kwargs)`` resolves ``__call__`` on the *type*, not the instance, so
    we wrap the class method (fine for a test process). Both Wan parity entry
    points call the pipeline with the same batched kwargs: ``input_image`` (list
    of length B), ``input_image4`` (B x list), ``action`` (tensor [B, ...]),
    ``batch_size`` (B); the return is an env-major list of length B.

    Note: with the fixed-noise patch keyed by (seed, shape), equal-sized chunks
    share the same noise tensor. That is fine for a kernel-parity test -- the
    conditioning differs per env and both devices run the identical split -- but
    it means the golden is not bit-comparable to an un-chunked single-batch run.
    """
    if micro_batch is None or micro_batch <= 0:
        return
    cls = type(pipe)
    orig_call = cls.__call__

    def _chunked_call(self, *args, input_image=None, input_image4=None,
                      action=None, batch_size=None, **kwargs):
        # Fall through unless this is the batched shape both parity scripts use.
        if (args or not isinstance(input_image, list) or action is None
                or batch_size is None or batch_size <= micro_batch):
            return orig_call(self, *args, input_image=input_image,
                             input_image4=input_image4, action=action,
                             batch_size=batch_size, **kwargs)
        out: list = []
        for s in range(0, batch_size, micro_batch):
            e = min(s + micro_batch, batch_size)
            out.extend(orig_call(
                self,
                input_image=input_image[s:e],
                input_image4=input_image4[s:e] if input_image4 is not None else None,
                action=action[s:e],
                batch_size=e - s,
                **kwargs,
            ))
        return out

    cls.__call__ = _chunked_call


# ---------------------------------------------------------------------------
# Layer fingerprinting (shared by openvla_oft and wan layerwise scripts)
# ---------------------------------------------------------------------------


def _first_tensor(x) -> "torch.Tensor | None":
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


def layer_fingerprint(t: torch.Tensor) -> dict:
    """Cheap, device-independent summary of a tensor.

    Computes reductions ON the tensor's own device (no CPU copy of the full
    activation). Only a 5-value vector crosses to CPU, so fingerprinting all
    ~1500 modules of a large VLA takes ~6s instead of 17+ minutes.

    The order-sensitive checksum (dot against a linspace ramp) flags any
    single-element change or permutation, so equal chk => bit-identical layout.
    """
    orig_dtype = str(t.dtype)
    d = t.detach().reshape(-1)
    n = d.numel()
    if n == 0:
        return {"shape": tuple(t.shape), "dtype": orig_dtype,
                "mean": 0.0, "std": 0.0, "absmax": 0.0, "l2": 0.0, "chk": 0.0}
    f = d.float()
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
