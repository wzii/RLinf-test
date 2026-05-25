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

    # ── Step 0: 在任何模型加载之前检查 LlamaSdpaAttention ──────────────────
    print("\n=== Step 0: LlamaSdpaAttention.forward BEFORE get_model ===")
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention
        print(f"  forward: {_fmt_fwd(LlamaSdpaAttention.forward)}")
    except ImportError:
        print("  LlamaSdpaAttention not found in this transformers version")
        return 1

    # ── Step 1: 安装 fp32 patch（加载模型之前）───────────────────────────────
    print("\n=== Step 1: install_fp32_attention (before get_model) ===")
    call_counter = [0]      # 用列表方便闭包修改

    # 用带计数器的版本替换 _fp32_forward
    import math as _math
    import torch.nn.functional as _F
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb, repeat_kv,
    )

    def _fp32_forward_counted(
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
        call_counter[0] += 1
        if call_counter[0] <= 2:        # 只打印前两次避免刷屏
            print(f"  [counter] _fp32_forward_counted called! total={call_counter[0]}")

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

    _orig_forward_id = id(LlamaSdpaAttention.forward)
    LlamaSdpaAttention.forward = _fp32_forward_counted
    print(f"  Patched. forward id: {_orig_forward_id:#x} -> {id(LlamaSdpaAttention.forward):#x}")

    # ── Step 2: 加载模型 ──────────────────────────────────────────────────────
    print("\n=== Step 2: get_model ===")
    from rlinf.models.embodiment.openvla_oft.rlinf import get_model  # noqa: E402
    cfg = build_cfg(OPENVLA_OFT_CKPT_DIR)

    t0 = time.time()
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("\n=== Step 3: LlamaSdpaAttention.forward AFTER get_model ===")
    print(f"  forward: {_fmt_fwd(LlamaSdpaAttention.forward)}")
    fwd_after_load = id(LlamaSdpaAttention.forward)
    if fwd_after_load != id(_fp32_forward_counted):
        print(f"  *** OVERRIDDEN! Our patch id={id(_fp32_forward_counted):#x} but class now has id={fwd_after_load:#x}")
        print(f"  *** This means get_model (or torch_npu import) replaced our patch!")
    else:
        print(f"  OK: our patch is still in place (id={fwd_after_load:#x})")

    # 还检查第一个 self_attn 实例上的 forward（可能是实例级别的绑定）
    first_attn = next(
        (m for n, m in model.named_modules()
         if n == "language_model.model.layers.0.self_attn"),
        None,
    )
    if first_attn is not None:
        inst_fwd = getattr(first_attn, "forward", None)
        if inst_fwd is not None:
            print(f"  instance forward: {_fmt_fwd(inst_fwd)}")
            if id(inst_fwd) != id(_fp32_forward_counted):
                print(f"  *** INSTANCE OVERRIDE: instance forward differs from class forward!")
        print(f"  type(self_attn): {type(first_attn).__name__}  id={id(type(first_attn).forward):#x}")

    # ── Step 4: 运行 forward 并检查计数器 ────────────────────────────────────
    print("\n=== Step 4: run 1-sample forward ===")
    # 用单个样本以减少测试时间
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
        print("\n  *** DIAGNOSIS: _fp32_forward_counted was NEVER CALLED.")
        print("  *** The model is NOT using our patched LlamaSdpaAttention.forward.")
        # 打印更多诊断信息
        print("\n  Possible causes:")
        print("  a) torch_npu replaces LlamaSdpaAttention.forward after our patch")
        print("  b) The model uses a different attention class (not LlamaSdpaAttention)")
        print("  c) The model bypasses Python-level dispatch (C++ extension)")
        attn_classes = set()
        for n, m in model.named_modules():
            if "self_attn" in n and n.count(".") == 4:  # layers.X.self_attn
                attn_classes.add(type(m).__name__)
        print(f"\n  Actual attention class(es) in model: {attn_classes}")
        # 检查是否有 __torch_function__ 或其他拦截
        if first_attn is not None:
            print(f"\n  self_attn MRO: {[c.__name__ for c in type(first_attn).__mro__]}")
    else:
        print(f"\n  DIAGNOSIS: _fp32_forward_counted WAS called {call_counter[0]} times.")
        print("  The patch is executing. NPU fp32==bf16 means Ascend's matmul/softmax")
        print("  gives identical results in bf16 and fp32 precision.")

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
