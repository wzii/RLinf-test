#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""诊断 install_fp32_attention 在 NPU 上是否真正执行。

现象：在 NPU 上 fp32attn 和 bf16 结果完全相同（bit-exact），但在 GPU 上
fp32attn ≠ bf16（差异 ~0.7）。这说明 _fp32_forward 在 NPU 上可能根本没有
被执行——最可能的原因是 torch_npu 在模型加载后对 LlamaSdpaAttention.forward
做了二次 monkey-patch，覆盖了我们的修改。

本脚本做三件事：
  1. 在加载模型之前打印 LlamaSdpaAttention.forward 的函数身份（id）。
  2. 安装 fp32 patch（install_fp32_attention）后再打印。
  3. 加载模型后（get_model 可能触发 torch_npu patch）再打印。
  4. 运行一个极小的单样本 forward，用 call_counter 验证 _fp32_forward 是否被调用。
  5. 输出 forward 后 LlamaSdpaAttention.forward 的 id，确认是否被改回。

Usage
-----
    python tests/parity/openvla_oft/run_patch_verify.py
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
    OPENVLA_OFT_CKPT_DIR,
    collect_libero_image_inputs,
    pick_device,
    print_env_banner,
    set_determinism,
)
from run_rlinf import build_cfg, build_env_obs  # noqa: E402


def _fmt_fwd(fn) -> str:
    """返回函数的模块+名称+id，便于追踪是哪个版本的 forward。"""
    mod = getattr(fn, "__module__", "?")
    name = getattr(fn, "__name__", getattr(fn, "__qualname__", str(fn)))
    return f"{mod}.{name}  id={id(fn):#x}"


def main():
    set_determinism()
    device = pick_device()
    print_env_banner(device, {"impl": "patch-verify"})

    # ── Step 0: 检查 transformers 版本和 attention 类 ─────────────────────
    print("\n=== Step 0: transformers version + attention class BEFORE get_model ===")
    try:
        import transformers as _tf
        print(f"  transformers version: {_tf.__version__}")
    except Exception:
        pass

    _has_sdpa_cls = False
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention
        _has_sdpa_cls = True
        print(f"  LlamaSdpaAttention.forward: {_fmt_fwd(LlamaSdpaAttention.forward)}")
    except ImportError:
        print("  LlamaSdpaAttention NOT found (transformers ≥4.46 unified class)")

    # Check for ≥4.46 unified path
    from transformers.models.llama import modeling_llama as _llama_mod
    for _fn_name in ("sdpa_attention_forward", "eager_attention_forward"):
        _fn = getattr(_llama_mod, _fn_name, None)
        if _fn is not None:
            print(f"  {_fn_name}: {_fmt_fwd(_fn)}")

    if not _has_sdpa_cls and not any(
        hasattr(_llama_mod, n) for n in ("sdpa_attention_forward", "eager_attention_forward")
    ):
        print("  Neither LlamaSdpaAttention nor sdpa/eager_attention_forward found!")
        return 1

    # ── Step 1: 安装带计数器的 fp32 patch（加载模型之前）─────────────────────
    print("\n=== Step 1: install counted fp32 patch (before get_model) ===")
    import math as _math
    import torch.nn.functional as _F
    from transformers.models.llama import modeling_llama as _llama_mod

    call_counter = [0]

    if _has_sdpa_cls:
        # transformers ≤4.45: 替换 LlamaSdpaAttention.forward
        from transformers.models.llama.modeling_llama import (
            LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv,
        )

        def _fp32_forward_counted(self, hidden_states, attention_mask=None,
                                   position_ids=None, past_key_value=None,
                                   output_attentions=False, use_cache=False,
                                   cache_position=None, **kwargs):
            call_counter[0] += 1
            if call_counter[0] <= 2:
                print(f"  [counter] LlamaSdpaAttention _fp32_forward called! total={call_counter[0]}")
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
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
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
            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
                D = causal_mask.shape[-1]
                last_row = causal_mask[:, :, -1, :].clone()
                causal_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
            q, k, v = query_states.float(), key_states.float(), value_states.float()
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

        _orig_forward_id = id(LlamaSdpaAttention.forward)
        LlamaSdpaAttention.forward = _fp32_forward_counted
        _patched_obj = LlamaSdpaAttention
        _patched_attr = "forward"
        print(f"  Patched LlamaSdpaAttention.forward: {_orig_forward_id:#x} -> {id(_fp32_forward_counted):#x}")

    else:
        # transformers ≥4.46: 替换 modeling_llama 模块级别函数
        _fn_name = next((n for n in ("sdpa_attention_forward", "eager_attention_forward")
                         if hasattr(_llama_mod, n)), None)
        if _fn_name is None:
            print("  No patchable attention function found! Cannot install counter patch.")
            return 1
        _orig_fn = getattr(_llama_mod, _fn_name)

        def _fp32_attn_counted(module, query, key, value, attention_mask,
                               scaling, dropout=0.0, **kwargs):
            call_counter[0] += 1
            if call_counter[0] <= 2:
                print(f"  [counter] {_fn_name} fp32 called! total={call_counter[0]}")
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

        setattr(_llama_mod, _fn_name, _fp32_attn_counted)
        _patched_obj = _llama_mod
        _patched_attr = _fn_name
        print(f"  Patched {_fn_name} in modeling_llama: {id(_orig_fn):#x} -> {id(_fp32_attn_counted):#x}")

    # ── Step 2: 加载模型 ──────────────────────────────────────────────────────
    print("\n=== Step 2: get_model ===")
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model  # noqa: E402
    cfg = build_cfg(OPENVLA_OFT_CKPT_DIR)

    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("\n=== Step 3: patch status AFTER get_model ===")
    current_fn = getattr(_patched_obj, _patched_attr)
    if _has_sdpa_cls:
        expected_fn = _fp32_forward_counted
        patch_id_str = f"LlamaSdpaAttention.forward id={id(current_fn):#x}"
    else:
        expected_fn = _fp32_attn_counted
        patch_id_str = f"{_fn_name} id={id(current_fn):#x}"
    if id(current_fn) != id(expected_fn):
        print(f"  *** OVERRIDDEN after get_model! {patch_id_str} != expected id={id(expected_fn):#x}")
        print(f"  Current fn: {_fmt_fwd(current_fn)}")
    else:
        print(f"  OK: patch still in place. {patch_id_str}")

    # 检查第一个 self_attn 实例的 class 和实例级 forward
    first_attn = next(
        (m for n, m in model.named_modules()
         if n == "language_model.model.layers.0.self_attn"),
        None,
    )
    if first_attn is not None:
        inst_fwd = first_attn.__dict__.get("forward")
        if inst_fwd is not None:
            print(f"  instance-level forward override: {_fmt_fwd(inst_fwd)}")
        print(f"  type(self_attn): {type(first_attn).__name__}")
        print(f"  class.forward id: {id(type(first_attn).forward):#x}")

    # ── Step 4: 运行 forward 并检查计数器 ────────────────────────────────────
    print("\n=== Step 4: run 1-sample forward ===")
    inputs = collect_libero_image_inputs(n=1)
    env_obs = build_env_obs(inputs, device)

    call_counter[0] = 0
    print(f"  call_counter before forward: {call_counter[0]}")
    t1 = time.time()
    with torch.no_grad():
        _, result = model.predict_action_batch(
            env_obs=env_obs, do_sample=False, temperature=1.0,
            top_k=-1, top_p=1.0, max_new_tokens=None, mode="eval",
        )
    print(f"  forward done in {time.time()-t1:.2f}s")
    print(f"  call_counter after forward:  {call_counter[0]}")

    if call_counter[0] == 0:
        print("\n  *** DIAGNOSIS: fp32 patch was NEVER CALLED during forward.")
        print("  Possible causes:")
        print("  a) The model uses a different attention class not covered by the patch")
        print("  b) torch_npu/Ascend transformers override our patch after model load")
        print("  c) Model bypasses Python-level dispatch (C++ extension)")
        # Identify actual attention class(es)
        attn_classes = set()
        for n, m in model.named_modules():
            if "self_attn" in n and n.count(".") == 4:
                attn_classes.add(type(m).__name__)
        print(f"\n  Actual attention class(es): {attn_classes}")
        if first_attn is not None:
            print(f"  MRO: {[c.__name__ for c in type(first_attn).__mro__]}")
        # Show what functions are in modeling_llama now
        for fn in ("sdpa_attention_forward", "eager_attention_forward", "ALL_ATTENTION_FUNCTIONS"):
            fn_obj = getattr(_llama_mod, fn, None)
            if fn_obj is not None:
                print(f"  {fn}: {_fmt_fwd(fn_obj) if callable(fn_obj) else fn_obj}")
    else:
        print(f"\n  DIAGNOSIS: fp32 patch WAS called {call_counter[0]} times.")
        if _has_sdpa_cls:
            print("  LlamaSdpaAttention.forward is executing our fp32 code.")
        else:
            print(f"  {_fn_name} in modeling_llama is executing our fp32 code.")
        print("  If NPU fp32==bf16 still holds, Ascend's own attention == fp32 numerically.")

    # ── Step 5: 检查是否有实例级别的 _orig_forward ────────────────────────────
    print("\n=== Step 5: Check for instance-level forward overrides ===")
    overridden = []
    for n, m in model.named_modules():
        if "self_attn" in n and n.count(".") == 4:
            inst_dict_fwd = m.__dict__.get("forward")
            if inst_dict_fwd is not None:
                overridden.append((n, _fmt_fwd(inst_dict_fwd)))
    if overridden:
        print(f"  {len(overridden)} modules have instance-level forward:")
        for nm, fmted in overridden[:3]:
            print(f"    {nm}: {fmted}")
    else:
        print("  No instance-level forward overrides found")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
