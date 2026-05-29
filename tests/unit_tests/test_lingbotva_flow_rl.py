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

"""Unit tests for the LingBot-VA flow-matching RL core math.

These cover the Wan/GPU-free part of the Flow-SDE / Flow-Noise integration:
the per-step Gaussian transition, the log-prob, and the rollout/recompute
consistency that GRPO relies on (importance ratio == 1 at collection time).
"""

import importlib.util
import math
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

# ``flow_rl`` depends only on torch + stdlib, so load it directly from its file
# path rather than via ``import rlinf...`` (the package __init__ pulls in heavy
# optional deps like omegaconf). This keeps the core math testable on a bare
# torch install / CI shard.
_FLOW_RL_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "lingbotva"
    / "flow_rl.py"
)
_spec = importlib.util.spec_from_file_location("lingbotva_flow_rl", _FLOW_RL_PATH)
flow_rl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = flow_rl  # let @dataclass resolve __module__
_spec.loader.exec_module(flow_rl)

FlowRLConfig = flow_rl.FlowRLConfig
LearnableNoiseHead = flow_rl.LearnableNoiseHead
build_sigma_schedule = flow_rl.build_sigma_schedule
gaussian_entropy = flow_rl.gaussian_entropy
get_logprob_norm = flow_rl.get_logprob_norm
recompute_logprob_entropy = flow_rl.recompute_logprob_entropy
rollout_action_chain = flow_rl.rollout_action_chain
sample_mean_var_val = flow_rl.sample_mean_var_val


class _FakeScheduler:
    """Minimal DiffSynth-style FlowMatchScheduler stub (linear sigmas 1->0)."""

    num_train_timesteps = 1000

    def set_timesteps(self, n):
        self.sigmas = torch.linspace(1.0, 1.0 / n, n)
        self.timesteps = self.sigmas * self.num_train_timesteps


def _linear_velocity_fn(target_x0, target_x1):
    """Constant velocity v = x1 - x0 for a fixed (x0, x1) pair."""

    def fn(x_t, sigma):
        return target_x1 - target_x0

    return fn


def test_sigma_schedule_padded_and_decreasing():
    sigmas = build_sigma_schedule(_FakeScheduler(), 10)
    assert sigmas.numel() == 11
    assert float(sigmas[-1]) == 0.0
    assert torch.all(sigmas[:-1] - sigmas[1:] > 0)  # strictly decreasing


def test_flow_ode_reproduces_euler_step():
    # With v = x1 - x0 the deterministic step must land exactly on the
    # next-sigma interpolation x_{sigma_next} = (1-s')x0 + s' x1.
    x0 = torch.randn(4, 8)
    x1 = torch.randn(4, 8)
    sigma, sigma_next = 0.7, 0.4
    x_t = (1 - sigma) * x0 + sigma * x1
    v = x1 - x0
    cfg = FlowRLConfig(noise_method="flow_ode")
    mean, std = sample_mean_var_val(x_t, v, sigma, sigma_next, cfg, train=False)
    expected = (1 - sigma_next) * x0 + sigma_next * x1
    assert torch.allclose(mean, expected, atol=1e-5)
    assert torch.count_nonzero(std) == 0


def test_flow_sde_reduces_to_ode_as_noise_to_zero():
    x0, x1 = torch.randn(2, 5), torch.randn(2, 5)
    sigma, sigma_next = 0.6, 0.3
    x_t = (1 - sigma) * x0 + sigma * x1
    v = x1 - x0
    ode = sample_mean_var_val(
        x_t, v, sigma, sigma_next, FlowRLConfig("flow_ode"), train=False
    )[0]
    sde_mean, sde_std = sample_mean_var_val(
        x_t, v, sigma, sigma_next, FlowRLConfig("flow_sde", noise_level=0.0), train=True
    )
    assert torch.allclose(sde_mean, ode, atol=1e-5)
    assert torch.count_nonzero(sde_std) == 0


def test_flow_sde_std_matches_formula():
    x_t = torch.randn(3, 4)
    v = torch.randn(3, 4)
    sigma, sigma_next, nl = 0.8, 0.5, 0.7
    _, std = sample_mean_var_val(
        x_t, v, sigma, sigma_next, FlowRLConfig("flow_sde", noise_level=nl), train=True
    )
    delta = sigma - sigma_next
    expected = math.sqrt(delta) * nl * math.sqrt(sigma / (1 - sigma))
    assert math.isclose(float(std.flatten()[0]), expected, rel_tol=1e-5)


def test_get_logprob_norm_matches_torch_normal():
    sample = torch.randn(5, 6)
    mu = torch.randn(5, 6)
    sigma = torch.rand(5, 6) + 0.1
    ours = get_logprob_norm(sample, mu, sigma)
    ref = torch.distributions.Normal(mu, sigma).log_prob(sample)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_get_logprob_norm_zero_sigma_is_zero():
    sample = torch.randn(4)
    mu = torch.randn(4)
    sigma = torch.zeros(4)
    assert torch.count_nonzero(get_logprob_norm(sample, mu, sigma)) == 0


def test_flow_noise_entropy_positive():
    std = torch.full((3, 3), 0.5)
    ent = gaussian_entropy(std)
    expected = 0.5 * math.log(2 * math.pi * math.e * 0.25)
    assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-5)


def test_rollout_recompute_logprob_consistency_flow_sde():
    """The crux: recompute under the same policy must match rollout log-probs.

    GRPO assumes the importance ratio is 1 at collection time, i.e.
    recompute_logprob == prev_logprob before any optimiser step. With a
    deterministic velocity_fn the two paths must agree to fp precision.
    """
    torch.manual_seed(0)
    cfg = FlowRLConfig(noise_method="flow_sde", noise_level=0.7)
    sigmas = build_sigma_schedule(_FakeScheduler(), 6)
    x0, x1 = torch.randn(8, 7), torch.randn(8, 7)
    vfn = _linear_velocity_fn(x0, x1)
    gen = torch.Generator().manual_seed(123)

    out = rollout_action_chain(
        x1.clone(), sigmas, vfn, cfg, denoise_index=3, generator=gen
    )
    assert out["chains"].shape == (8, sigmas.numel(), 7)
    assert out["logprobs"].shape == (8, 7)

    re_lp, re_ent = recompute_logprob_entropy(
        out["chains"], sigmas, out["denoise_index"], vfn, cfg
    )
    assert torch.allclose(re_lp, out["logprobs"], atol=1e-5)
    assert torch.count_nonzero(re_ent) == 0  # flow_sde -> no entropy term


def test_rollout_recompute_consistency_flow_noise():
    torch.manual_seed(0)
    cfg = FlowRLConfig(noise_method="flow_noise")
    sigmas = build_sigma_schedule(_FakeScheduler(), 5)
    x0, x1 = torch.randn(4, 6), torch.randn(4, 6)
    vfn = _linear_velocity_fn(x0, x1)
    # A fixed learnable-std stand-in.
    std_fn = lambda x, s: torch.full_like(x, 0.3)  # noqa: E731
    gen = torch.Generator().manual_seed(7)

    out = rollout_action_chain(
        x1.clone(), sigmas, vfn, cfg, denoise_index=2, noise_std_fn=std_fn, generator=gen
    )
    re_lp, re_ent = recompute_logprob_entropy(
        out["chains"], sigmas, out["denoise_index"], vfn, cfg, noise_std_fn=std_fn
    )
    assert torch.allclose(re_lp, out["logprobs"], atol=1e-5)
    assert torch.all(re_ent > 0)  # flow_noise carries entropy


def test_recompute_per_sample_denoise_index():
    """Mixed micro-batch: each sample carries its own denoise index.

    Two rollout 'calls' with different indices are concatenated; recompute with
    a [B] index vector must reproduce each sample's own rollout log-prob.
    """
    torch.manual_seed(0)
    cfg = FlowRLConfig(noise_method="flow_sde", noise_level=0.7)
    sigmas = build_sigma_schedule(_FakeScheduler(), 6)

    x0a, x1a = torch.randn(3, 7), torch.randn(3, 7)
    x0b, x1b = torch.randn(2, 7), torch.randn(2, 7)
    # A shared velocity_fn over the concatenated batch (constant velocity per
    # sample): build from concatenated (x0, x1).
    x0 = torch.cat([x0a, x0b], 0)
    x1 = torch.cat([x1a, x1b], 0)
    vfn = _linear_velocity_fn(x0, x1)

    out_a = rollout_action_chain(
        x1[:3].clone(), sigmas, _linear_velocity_fn(x0a, x1a), cfg,
        denoise_index=1, generator=torch.Generator().manual_seed(11),
    )
    out_b = rollout_action_chain(
        x1[3:].clone(), sigmas, _linear_velocity_fn(x0b, x1b), cfg,
        denoise_index=4, generator=torch.Generator().manual_seed(22),
    )
    chains = torch.cat([out_a["chains"], out_b["chains"]], dim=0)
    denoise_index = torch.tensor([1, 1, 1, 4, 4])
    prev = torch.cat([out_a["logprobs"], out_b["logprobs"]], dim=0)

    re_lp, _ = recompute_logprob_entropy(chains, sigmas, denoise_index, vfn, cfg)
    assert torch.allclose(re_lp, prev, atol=1e-5)


def test_learnable_noise_head_shape_and_bounds():
    head = LearnableNoiseHead(action_dim=7, std_min=1e-3, std_max=1.0)
    x = torch.randn(4, 7, 3, 4, 1)  # [B, action_dim, F, A, 1]
    std = head(x, sigma=0.5)
    assert std.shape == x.shape
    assert torch.all(std >= 1e-3) and torch.all(std <= 1.0)
    # std is per-channel: constant across the F/A/1 axes.
    assert torch.allclose(std[:, 0], std[:, 0, :1, :1, :1].expand_as(std[:, 0]))


def test_learnable_noise_head_is_trainable():
    head = LearnableNoiseHead(action_dim=5)
    x = torch.randn(2, 5, 2, 2, 1)
    std = head(x, 0.3)
    loss = ((std - 0.4) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_joint_logprob_uses_all_steps():
    cfg = FlowRLConfig(noise_method="flow_sde", noise_level=0.5, joint_logprob=True)
    sigmas = build_sigma_schedule(_FakeScheduler(), 4)
    x0, x1 = torch.randn(2, 3), torch.randn(2, 3)
    vfn = _linear_velocity_fn(x0, x1)
    gen = torch.Generator().manual_seed(1)
    out = rollout_action_chain(x1.clone(), sigmas, vfn, cfg, generator=gen)
    re_lp, _ = recompute_logprob_entropy(
        out["chains"], sigmas, out["denoise_index"], vfn, cfg
    )
    assert torch.allclose(re_lp, out["logprobs"], atol=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
