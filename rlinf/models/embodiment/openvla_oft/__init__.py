# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def _install_fp32_attention(model) -> None:
    """Replace F.scaled_dot_product_attention with a manual fp32 implementation.

    Eliminates hardware-specific kernel accumulation differences (cuDNN FA on
    NVIDIA vs aclnn_fa on Ascend) by casting q/k/v to float32 before the
    matmul and softmax, then casting the output back to the original dtype.

    Supports both transformers ≤4.45 (LlamaSdpaAttention subclass) and
    transformers ≥4.46 (unified LlamaAttention with internal dispatch).
    """
    # ── transformers ≥ 4.46 ──────────────────────────────────────────────────
    # In 4.46+ the three subclasses were merged into a single LlamaAttention
    # that dispatches internally via eager_attention_forward / sdpa / flash.
    # Patch at the module level if the old subclass is absent.
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention
        _has_sdpa_cls = True
    except ImportError:
        _has_sdpa_cls = False

    if not _has_sdpa_cls:
        # 4.46+ path: patch eager_attention_forward (used by all implementations
        # when attention_type=="sdpa" is overridden) or patch LlamaAttention.forward
        # to call our fp32 kernel directly.
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            from transformers.models.llama import modeling_llama as _llama_mod

            def _fp32_eager_attn(module, query, key, value, attention_mask,
                                 scaling, dropout=0.0, **kwargs):
                q, k, v = query.float(), key.float(), value.float()
                attn_w = torch.matmul(q, k.transpose(-2, -1)) * scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, :key.shape[-2]]
                    attn_w = attn_w + causal_mask.float()
                attn_w = F.softmax(attn_w, dim=-1)
                if module.training and getattr(module, "attention_dropout", 0.0) > 0.0:
                    attn_w = F.dropout(attn_w, p=module.attention_dropout)
                out = torch.matmul(attn_w, v).to(query.dtype)
                return out, attn_w

            # Override the module-level eager_attention_forward so that all
            # calls within modeling_llama.py resolve to our fp32 version.
            if hasattr(_llama_mod, "eager_attention_forward"):
                _llama_mod.eager_attention_forward = _fp32_eager_attn
                n = sum(1 for m in model.modules()
                        if isinstance(m, LlamaAttention))
                print(f"[rlinf] fp32_attention: patched eager_attention_forward "
                      f"for {n} LlamaAttention modules (transformers ≥4.46)")
            else:
                print("[rlinf][WARN] fp32_attention: eager_attention_forward not found "
                      "in modeling_llama; patch skipped")
        except ImportError:
            print("[rlinf][WARN] fp32_attention: LlamaAttention not found; patch skipped")
        return

    # ── transformers ≤ 4.45 (has LlamaSdpaAttention subclass) ──────────────
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            repeat_kv,
        )
    except ImportError:
        print("[rlinf][WARN] fp32_attention: helper imports failed; patch skipped")
        return

    def _fp32_forward(
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

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            D = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            causal_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)

        q = query_states.float()
        k = key_states.float()
        v = value_states.float()
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.float()
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        attn_output = torch.matmul(attn_weights, v).to(query_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    LlamaSdpaAttention.forward = _fp32_forward
    n = sum(1 for m in model.modules() if type(m).__name__ == "LlamaSdpaAttention")
    print(f"[rlinf] fp32_attention: patched {n} LlamaSdpaAttention modules (transformers ≤4.45)")


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    implement_version = cfg.get("implement_version", "rlinf")
    if implement_version == "rlinf":
        from rlinf.models.embodiment.openvla_oft.rlinf import get_model
    elif implement_version == "official":
        from rlinf.models.embodiment.openvla_oft.official import get_model
    else:
        raise NotImplementedError(
            f"Unsupported model implementation version: '{implement_version}'. "
            f"Currently supported versions: ['rlinf', 'official']. "
            f"Please check ...model.version or implement the corresponding model loader."
        )

    model = get_model(cfg, torch_dtype)

    if cfg.get("fp32_attention", False):
        _install_fp32_attention(model)

    return model
