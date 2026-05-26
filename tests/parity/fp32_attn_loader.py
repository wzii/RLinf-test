# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Auto-loadable fp32 attention patch for OpenVLA-OFT eval on NPU.

When this module is *imported* (via a venv ``.pth`` file installed by
``eval_libero_npu_fp32.sh``), it monkey-patches the LLaMA SDPA attention
implementation in ``transformers`` to compute q/k/v in fp32. This eliminates
the ~0.1-0.5 relL2 attention drift between cuDNN FA (GPU) and aclnn_fa (NPU)
that was empirically responsible for ~80 % of GPU↔NPU output divergence in
the OpenVLA-OFT model (token agreement on 32-sample parity test jumped from
20 % to 97.6 % with the patch active).

The patch lives entirely in this file (no imports from ``tests/parity/common.py``)
so it can be loaded from any directory layout. It is identical in math to
``common.install_fp32_attention`` but written as a self-contained module that
auto-installs on first import and is idempotent on subsequent imports.

Behaviour
---------
- Supports both transformers ≤ 4.45 (``LlamaSdpaAttention`` subclass) and
  ≥ 4.46 (module-level ``sdpa_attention_forward`` function).
- Idempotent: a marker attribute on the patch target prevents double-patching.
- Set ``PARITY_DISABLE_FP32_ATTN=1`` to skip the patch (useful for A/B runs).
- Logs ``[parity] fp32 attention patch installed (PID <pid>)`` so you can
  confirm it ran in the head process AND every Ray worker.
"""

from __future__ import annotations

import math
import os
import sys

_MARKER = "__rlinf_parity_fp32_attn_installed__"
_VERBOSE = os.environ.get("PARITY_FP32_ATTN_VERBOSE", "0") == "1"


def _log(msg: str) -> None:
    # Stay quiet by default in workers; only the first install per process logs.
    print(f"[parity] {msg} (PID {os.getpid()})", file=sys.stderr, flush=True)


def _install() -> None:
    if os.environ.get("PARITY_DISABLE_FP32_ATTN") == "1":
        _log("fp32 attention patch DISABLED via PARITY_DISABLE_FP32_ATTN=1")
        return

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        # torch isn't installed yet in this process; nothing to patch.
        return

    try:
        from transformers.models.llama import modeling_llama as _llama_mod
    except ImportError:
        # transformers not available in this process (e.g. early init scripts).
        if _VERBOSE:
            _log("transformers not importable yet; skipping")
        return

    # ── Path A: transformers ≤ 4.45 — LlamaSdpaAttention subclass ────────────
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaSdpaAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )
        _has_sdpa_cls = True
    except ImportError:
        _has_sdpa_cls = False

    if _has_sdpa_cls:
        if getattr(LlamaSdpaAttention.forward, _MARKER, False):
            if _VERBOSE:
                _log("LlamaSdpaAttention already patched; skipping")
            return

        def _fp32_forward(
            self,
            hidden_states,
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
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            bsz, q_len, _ = hidden_states.size()
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            pkv = getattr(self, "past_key_value", past_key_value)
            if pkv is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = pkv.update(k, v, self.layer_idx, cache_kwargs)

            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : k.shape[-2]]
                D = causal_mask.shape[-1]
                last = causal_mask[:, :, -1, :].clone()
                causal_mask = last.unsqueeze(2).expand(-1, -1, D, -1)

            qf, kf, vf = q.float(), k.float(), v.float()
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
            if causal_mask is not None:
                attn = attn + causal_mask.float()
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attention_dropout > 0.0:
                attn = F.dropout(attn, p=self.attention_dropout)
            out = torch.matmul(attn, vf).to(q.dtype)

            out = out.transpose(1, 2).contiguous()
            out = out.view(bsz, q_len, self.hidden_size)
            out = self.o_proj(out)
            return out, None, past_key_value

        setattr(_fp32_forward, _MARKER, True)
        LlamaSdpaAttention.forward = _fp32_forward
        _log("fp32 attention patch installed (transformers ≤4.45 path)")
        return

    # ── Path B: transformers ≥ 4.46 — module-level sdpa_attention_forward ──
    target_fn_name = None
    for name in ("sdpa_attention_forward", "eager_attention_forward"):
        if hasattr(_llama_mod, name):
            target_fn_name = name
            break
    if target_fn_name is None:
        _log("neither LlamaSdpaAttention nor sdpa/eager_attention_forward found; "
             "fp32 patch SKIPPED")
        return

    existing = getattr(_llama_mod, target_fn_name)
    if getattr(existing, _MARKER, False):
        if _VERBOSE:
            _log(f"{target_fn_name} already patched; skipping")
        return

    def _fp32_eager_attn(module, query, key, value, attention_mask,
                         scaling, dropout=0.0, **kwargs):
        qf, kf, vf = query.float(), key.float(), value.float()
        attn = torch.matmul(qf, kf.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn = attn + causal_mask.float()
        attn = F.softmax(attn, dim=-1)
        if module.training and getattr(module, "attention_dropout", 0.0) > 0.0:
            attn = F.dropout(attn, p=module.attention_dropout)
        out = torch.matmul(attn, vf).to(query.dtype)
        return out, attn

    setattr(_fp32_eager_attn, _MARKER, True)
    setattr(_llama_mod, target_fn_name, _fp32_eager_attn)
    _log(f"fp32 attention patch installed (transformers ≥4.46 path, "
         f"replaced {target_fn_name})")


_install()
