# Copyright 2026 The RLinf Authors.
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

"""Unit tests for the Wan world-model NPU patches.

The patched ``flash_attention`` / ``rope_apply`` / ``RMSNorm`` are drop-ins for
diffsynth's vendor-dispatched primitives. On a CPU/GPU CI host the NPU backends
(mindiesd / torch_npu) are absent, so these tests exercise the GPU/torch
fallback paths -- which must match the original torch math.
"""

import torch
import torch.nn.functional as F

from rlinf.envs.world_model.patch import RMSNorm, flash_attention, rope_apply


def _sdpa_reference(q, k, v, num_heads):
    bsz, seq_len, inner = q.shape
    head_dim = inner // num_heads

    def to_bnsd(t):
        return t.reshape(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(to_bnsd(q), to_bnsd(k), to_bnsd(v))
    return out.permute(0, 2, 1, 3).reshape(bsz, seq_len, inner)


def test_flash_attention_compatibility_mode_matches_sdpa():
    """compatibility_mode routes directly through SDPA; results must agree."""
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 2, 6, 4, 8
    inner = num_heads * head_dim
    q = torch.randn(bsz, seq_len, inner)
    k = torch.randn(bsz, seq_len, inner)
    v = torch.randn(bsz, seq_len, inner)

    out = flash_attention(q, k, v, num_heads=num_heads, compatibility_mode=True)
    ref = _sdpa_reference(q, k, v, num_heads)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_flash_attention_fallback_matches_sdpa():
    """On CPU (no NPU / flash-attn), the else-branch uses torch SDPA."""
    torch.manual_seed(42)
    bsz, seq_len, num_heads, head_dim = 1, 8, 2, 16
    inner = num_heads * head_dim
    q = torch.randn(bsz, seq_len, inner)
    k = torch.randn(bsz, seq_len, inner)
    v = torch.randn(bsz, seq_len, inner)

    out = flash_attention(q, k, v, num_heads=num_heads)
    ref = _sdpa_reference(q, k, v, num_heads)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_rope_apply_gpu_path_matches_reference():
    """The non-NPU rope path is the original diffsynth complex-multiply."""
    torch.manual_seed(1)
    bsz, seq_len, num_heads, head_dim = 1, 4, 2, 8
    inner = num_heads * head_dim
    x = torch.randn(bsz, seq_len, inner)

    # freqs as used by diffsynth: complex, shape (seq, 1, head_dim // 2).
    angles = torch.randn(seq_len, 1, head_dim // 2)
    freqs = torch.polar(torch.ones_like(angles), angles)

    out = rope_apply(x, freqs, num_heads)

    # Reference: identical complex-multiply.
    xr = x.reshape(bsz, seq_len, num_heads, head_dim)
    xc = torch.view_as_complex(
        xr.to(torch.float64).reshape(bsz, seq_len, num_heads, -1, 2)
    )
    ref = torch.view_as_real(xc * freqs).flatten(2).to(x.dtype)

    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_rmsnorm_gpu_path_matches_reference():
    """The non-NPU RMSNorm path is the original fp32 norm * weight."""
    torch.manual_seed(2)
    dim = 16
    norm = RMSNorm(dim, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim))

    x = torch.randn(3, 5, dim)
    out = norm(x)

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    ref = (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype) * norm.weight

    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
