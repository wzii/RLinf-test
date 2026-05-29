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

"""Flow-matching RL (Flow-SDE / Flow-Noise / Flow-CPS) for LingBot-VA.

This module ports RLinf's flow-matching RL machinery -- already used for the
``openpi`` (:math:`\\pi_0` / :math:`\\pi_{0.5}`) and ``lingbotvla`` action
experts (see :mod:`rlinf.models.embodiment.openpi.openpi_action_model` and
:mod:`rlinf.models.embodiment.lingbotvla.lingbotvla_action_model`) -- onto the
**action stream** of the LingBot-VA video-diffusion transformer (Wan 2.2).

Why this layer exists
---------------------
LingBot-VA generates actions by iteratively denoising an action latent with a
flow-matching ODE (``action_scheduler.step`` Euler updates in
:meth:`rlinf.models.embodiment.lingbotva.eval_adapter.native_backend.LingbotVALiberoBackend._infer_batch_impl`).
A deterministic ODE has an intractable per-action log-likelihood, so it cannot
be trained with policy-gradient RL directly. RLinf solves this two ways:

* **Flow-SDE** -- convert the denoising ODE into an equivalent SDE so each
  denoise step becomes a Gaussian transition with a closed-form log-prob.
  Exploration noise is injected by the SDE diffusion term.
* **Flow-Noise** -- keep the ODE mean but add a *learnable* per-step Gaussian
  noise head, modelling denoising as a discrete-time MDP with an exact
  log-prob.

Both make the multi-step denoiser a sequence of Gaussian transitions that GRPO
/ PPO can optimise, exactly as in the ``lingbotvla`` implementation.

Sigma / velocity convention
---------------------------
The Wan 2.2 ``FlowMatchScheduler`` (DiffSynth-derived) parameterises the
forward process as ``x_sigma = (1 - sigma) * x0 + sigma * x1`` with
``x1 ~ N(0, I)`` (pure noise at ``sigma = 1``, clean action at ``sigma = 0``)
and predicts the velocity ``v = x1 - x0 = d x_sigma / d sigma``. This is the
*same* convention RLinf's other action experts use, with the continuous time
``t`` replaced by the scheduler's ``sigma``. Concretely::

    x0_pred = x_t - sigma * v
    x1_pred = x_t + (1 - sigma) * v

and a single deterministic Euler step to the next sigma reproduces
``action_scheduler.step`` exactly::

    x_next = x0_pred * (1 - sigma_next) + x1_pred * sigma_next = x_t - v * delta

where ``delta = sigma - sigma_next``. The SDE / Noise variants only change the
``x1_pred`` weight and the std of the transition (see
:func:`sample_mean_var_val`).

The functions here are deliberately free of any Wan / GPU dependency -- they
operate on plain tensors and a ``velocity_fn`` callable -- so the core math is
unit-testable on CPU (see ``tests/unit_tests/test_lingbotva_flow_rl.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

# A velocity function maps ``(x_t, cond_time)`` -> predicted flow velocity
# ``v_t`` of the same shape as ``x_t``. ``cond_time`` is the time value handed
# to the transformer: by default the normalised ``sigma`` in ``[0, 1]``, but a
# parallel ``cond_times`` schedule can supply the scheduler's *native* timestep
# units instead (the Wan transformer expects raw scheduler timesteps, while the
# SDE math runs on ``sigma``).
VelocityFn = Callable[[torch.Tensor, float], torch.Tensor]


@dataclass
class FlowRLConfig:
    """Resolved knobs controlling the flow-matching RL sampler.

    Mirrors the fields read off the Hydra ``actor.model`` config so the action
    model can build it once and hand it around.
    """

    noise_method: str = "flow_sde"  # flow_ode | flow_sde | flow_noise | flow_cps
    noise_level: float = 0.7
    joint_logprob: bool = False  # log-prob over *all* steps (Flow-Noise style)
    ignore_last: bool = False  # never sample the final (sigma->0) step
    safe_get_logprob: bool = False  # drop the Gaussian normalisation constant
    # Numerical floor on ``1 - sigma`` so the Flow-SDE ``sqrt(sigma/(1-sigma))``
    # term stays finite at ``sigma == 1``.
    eps: float = 1e-6


class LearnableNoiseHead(nn.Module):
    """Per-channel learnable Gaussian std for Flow-Noise, conditioned on time.

    Flow-Noise (pi_RL) augments the denoising MDP with a *learnable* noise
    network. The Wan action expert does not expose its internal hidden states
    through the eval call path, so -- rather than reaching into ``wan_va``
    internals -- we condition the std on the (sinusoidally embedded) denoise
    time. This keeps the noise schedule learnable and step-dependent while
    staying decoupled from the frozen inference backend.

    ``forward(x_t, sigma)`` returns a std tensor broadcast to ``x_t``'s shape.
    The std is ``softplus``-bounded into ``[std_min, std_max]`` for stability.
    """

    def __init__(
        self,
        action_dim: int,
        time_embed_dim: int = 64,
        hidden_dim: int = 128,
        std_min: float = 1e-3,
        std_max: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_embed_dim = time_embed_dim
        self.std_min = std_min
        self.std_max = std_max
        self.net = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _time_embed(self, sigma: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding for ``sigma`` of shape ``[N]`` -> ``[N, D]``."""
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=sigma.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        ang = sigma[:, None] * freqs[None, :]  # [N, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.time_embed_dim:  # odd embed dim
            emb = torch.cat(
                [emb, emb.new_zeros(emb.shape[0], self.time_embed_dim - emb.shape[-1])],
                dim=-1,
            )
        return emb

    def forward(self, x_t: torch.Tensor, sigma) -> torch.Tensor:
        # ``sigma`` may be a float (whole batch shares a step) or a per-sample
        # ``[B]`` tensor (mixed micro-batch at the actor update).
        if torch.is_tensor(sigma) and sigma.dim() > 0:
            sig = sigma.to(device=x_t.device, dtype=torch.float32).flatten()
        else:
            val = float(sigma) if not torch.is_tensor(sigma) else float(sigma.item())
            sig = torch.full((x_t.shape[0],), val, device=x_t.device, dtype=torch.float32)
        emb = self._time_embed(sig)  # [B, D]
        raw = self.net(emb)  # [B, action_dim]
        std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(raw)
        # Action latent layout is [B, action_dim, F, A, 1]; broadcast std over
        # everything but the (batch, channel) axes.
        if x_t.dim() == 5 and x_t.shape[1] == self.action_dim:
            std = std.view(x_t.shape[0], self.action_dim, 1, 1, 1)
        else:  # generic fallback: action_dim on the last axis
            std = std.view(x_t.shape[0], *([1] * (x_t.dim() - 2)), self.action_dim)
        return std.expand_as(x_t).to(x_t.dtype)


def build_sigma_schedule(
    action_scheduler,
    num_inference_steps: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the denoising sigma schedule, padded with a trailing ``0``.

    The returned tensor has length ``num_inference_steps + 1`` and is strictly
    decreasing from (close to) ``1`` down to ``0``. Element ``i`` is the noise
    level *before* denoise step ``i``; element ``i + 1`` is the level after.

    We read ``action_scheduler.sigmas`` when available (the Wan / DiffSynth
    ``FlowMatchScheduler`` exposes it after ``set_timesteps``); otherwise we
    fall back to ``timesteps / num_train_timesteps``. The trailing ``0`` mirrors
    the ``F.pad(..., value=0)`` the eval backend applies to ``action_timesteps``.
    """
    action_scheduler.set_timesteps(num_inference_steps)
    sigmas = getattr(action_scheduler, "sigmas", None)
    if sigmas is None:
        ntt = getattr(action_scheduler, "num_train_timesteps", 1000)
        sigmas = action_scheduler.timesteps.to(torch.float32) / float(ntt)
    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype).flatten()
    if sigmas.numel() > num_inference_steps:
        sigmas = sigmas[:num_inference_steps]
    if float(sigmas[-1]) != 0.0:
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas


def _as_bcast(value, x_t: torch.Tensor) -> torch.Tensor:
    """Broadcast a float or per-sample ``[B]`` tensor to ``x_t``'s shape."""
    if torch.is_tensor(value):
        v = value.to(device=x_t.device, dtype=torch.float32)
        if v.dim() == 0:
            return v
        # [B] -> [B, 1, 1, ...] so it broadcasts over the latent axes.
        return v.view(v.shape[0], *([1] * (x_t.dim() - 1)))
    return torch.tensor(float(value), device=x_t.device, dtype=torch.float32)


def sample_mean_var_val(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    sigma,
    sigma_next,
    cfg: FlowRLConfig,
    *,
    train: bool,
    noise_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian transition of one denoise step.

    Returns ``(x_t_mean, x_t_std)`` such that the next action latent is sampled
    as ``x_next = x_t_mean + eps * x_t_std`` with ``eps ~ N(0, I)``. When
    ``train`` is ``False`` (or for ``flow_ode``) the std is zero and the update
    is the deterministic Euler step, exactly reproducing
    ``action_scheduler.step``.

    ``sigma`` / ``sigma_next`` may be python floats (whole batch shares a step,
    as at rollout) or per-sample ``[B]`` tensors (mixed micro-batches at the
    actor update, mirroring ``lingbotvla``'s per-sample ``denoise_inds`` gather).

    Args:
        x_t: current action latent.
        v_t: flow velocity predicted by the transformer at ``sigma``.
        sigma: current noise level(s) in ``[0, 1]``.
        sigma_next: noise level(s) after this step (``< sigma``).
        cfg: resolved flow-RL config.
        train: whether this step is stochastic (injects exploration noise).
        noise_std: learnable std for ``flow_noise`` (same shape as ``x_t``).
    """
    s = _as_bcast(sigma, x_t)
    s_next = _as_bcast(sigma_next, x_t)
    x0_pred = x_t - s * v_t
    x1_pred = x_t + (1.0 - s) * v_t
    delta = s - s_next

    ode_x0_w = 1.0 - s_next
    ode_x1_w = s_next

    if not train or cfg.noise_method == "flow_ode":
        x0_w, x1_w = ode_x0_w, ode_x1_w
        std = torch.zeros_like(x_t)
    elif cfg.noise_method == "flow_sde":
        # ODE -> SDE conversion (Flow-GRPO / pi_RL). sigma_i is the SDE
        # diffusion coefficient; the extra ``x1_pred`` drift keeps the marginal
        # consistent with the ODE.
        denom = torch.clamp(1.0 - s, min=cfg.eps)
        s_clamp = torch.clamp(s, min=cfg.eps)
        sigma_i = cfg.noise_level * torch.sqrt(torch.clamp(s, min=0.0) / denom)
        x0_w = ode_x0_w
        x1_w = ode_x1_w - (sigma_i**2) * delta / (2.0 * s_clamp)
        std = torch.sqrt(torch.clamp(delta, min=0.0)) * sigma_i
        std = std.expand_as(x_t)
    elif cfg.noise_method == "flow_cps":
        # Coefficients-preserving sampling (arXiv:2509.05952): rotate the noise
        # weight by ``pi * noise_level / 2`` instead of rescaling it.
        cos_term = math.cos(math.pi * cfg.noise_level / 2.0)
        sin_term = math.sin(math.pi * cfg.noise_level / 2.0)
        x0_w = ode_x0_w
        x1_w = ode_x1_w * cos_term
        std = (ode_x1_w * sin_term).expand_as(x_t)
    elif cfg.noise_method == "flow_noise":
        # Learnable Gaussian noise head; ODE mean is preserved.
        if noise_std is None:
            raise ValueError("flow_noise requires a learnable `noise_std`.")
        x0_w, x1_w = ode_x0_w, ode_x1_w
        std = noise_std
    else:
        raise ValueError(f"Invalid noise_method: {cfg.noise_method!r}")

    x_t_mean = x0_pred * x0_w + x1_pred * x1_w
    return x_t_mean, std


def get_logprob_norm(
    sample: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    safe: bool = False,
) -> torch.Tensor:
    """Elementwise Gaussian log-density ``log N(sample; mu, sigma)``.

    Deterministic steps (``sigma == 0``) contribute ``0`` so they drop out of
    the policy gradient. Ported verbatim from the ``lingbotvla`` /
    ``openpi`` implementations for byte-for-byte consistency.
    """
    sample = sample.to(torch.float32)
    mu = mu.to(torch.float32)
    sigma = sigma.to(torch.float32)

    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

    if safe:
        log_prob = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
    else:
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
    return torch.where(mask, torch.zeros_like(log_prob), log_prob)


def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
    """Differential entropy of ``N(., sigma)`` (0 where ``sigma == 0``)."""
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
    return torch.where(mask, torch.zeros_like(entropy), entropy)


def pick_denoise_index(
    num_steps: int,
    generator: torch.Generator | None = None,
) -> int:
    """Choose which denoise step carries the (single) stochastic transition.

    Flow-GRPO's efficiency trick: only one denoise step per trajectory is
    sampled stochastically (the rest are deterministic ODE steps), and only
    that step's log-prob enters the policy gradient. ``joint_logprob`` mode
    overrides this and uses every step.
    """
    high = num_steps - 1
    if generator is not None:
        return int(torch.randint(0, high, (1,), generator=generator).item())
    return int(torch.randint(0, high, (1,)).item())


@torch.no_grad()
def rollout_action_chain(
    init_noise: torch.Tensor,
    sigmas: torch.Tensor,
    velocity_fn: VelocityFn,
    cfg: FlowRLConfig,
    *,
    denoise_index: int | None = None,
    noise_std_fn: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
    cond_times: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Run the action denoising chain, returning the data RL needs.

    This is the rollout-time sampler. It denoises ``init_noise`` (the
    ``sigma = 1`` action latent) down to a clean action, injecting exploration
    noise at the selected step(s). Everything required to *recompute* the
    log-probs under gradient later (the full ``chains`` and the
    ``denoise_index``) is returned.

    Args:
        init_noise: ``[B, ...]`` Gaussian action latent at ``sigma = sigmas[0]``.
        sigmas: schedule from :func:`build_sigma_schedule` (len ``S + 1``).
        velocity_fn: maps ``(x_t, sigma)`` -> predicted velocity.
        cfg: resolved flow-RL config.
        denoise_index: which step is stochastic. ``None`` -> sample one.
        noise_std_fn: for ``flow_noise``, maps ``(x_t, sigma)`` -> learnable std.
        generator: optional RNG for reproducibility.

    Returns:
        dict with ``actions`` (clean ``x_0``), ``chains`` ``[B, S+1, ...]``,
        ``logprobs`` ``[B, ...]`` (summed over chosen steps), and
        ``denoise_index``.
    """
    num_steps = sigmas.numel() - 1
    joint = cfg.joint_logprob
    if denoise_index is None and not joint:
        hi = num_steps - 1 if cfg.ignore_last else num_steps
        denoise_index = pick_denoise_index(hi + 1, generator)

    x_t = init_noise
    chains = [x_t]
    step_logprobs: list[torch.Tensor] = []

    if joint:
        # In joint mode the initial draw from N(0, I) also counts.
        step_logprobs.append(
            get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t), cfg.safe_get_logprob
            )
        )

    for i in range(num_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        cond_time = float(cond_times[i]) if cond_times is not None else sigma
        train_step = joint or (i == denoise_index)

        v_t = velocity_fn(x_t, cond_time)
        noise_std = None
        if cfg.noise_method == "flow_noise" and train_step:
            if noise_std_fn is None:
                raise ValueError("flow_noise rollout requires noise_std_fn.")
            noise_std = noise_std_fn(x_t, cond_time)

        mean, std = sample_mean_var_val(
            x_t, v_t, sigma, sigma_next, cfg, train=train_step, noise_std=noise_std
        )
        if train_step and std.abs().sum() > 0:
            eps = torch.empty_like(x_t).normal_(generator=generator)
            x_t = mean + eps * std
        else:
            x_t = mean
        if joint or train_step:
            step_logprobs.append(get_logprob_norm(x_t, mean, std, cfg.safe_get_logprob))
        chains.append(x_t)

    chains_t = torch.stack(chains, dim=1)
    if joint:
        logprobs = torch.stack(step_logprobs, dim=1).mean(dim=1)
    else:
        logprobs = step_logprobs[0]

    return {
        "actions": x_t,
        "chains": chains_t,
        "logprobs": logprobs,
        "denoise_index": int(-1 if denoise_index is None else denoise_index),
    }


def recompute_logprob_entropy(
    chains: torch.Tensor,
    sigmas: torch.Tensor,
    denoise_index: torch.Tensor | int,
    velocity_fn: VelocityFn,
    cfg: FlowRLConfig,
    *,
    noise_std_fn: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
    cond_times: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute step log-probs / entropy under the *current* policy (with grad).

    Used by the actor update: given the ``chains`` collected at rollout and the
    ``denoise_index`` that was sampled, re-evaluate the velocity (and the
    learnable noise std for ``flow_noise``) so the Gaussian log-prob of the
    realised transition flows gradients into the policy. Mirrors
    ``lingbotvla``'s :meth:`get_log_prob_value`.

    ``denoise_index`` may be a scalar (all samples share it) or a ``[B]`` tensor.
    Returns ``(logprobs, entropy)`` matching the per-step layout of the rollout.
    """
    bsize = chains.shape[0]
    num_steps = sigmas.numel() - 1
    device = chains.device

    if cfg.joint_logprob:
        # Every step (shared across the batch). Accumulate the per-step log-prob.
        logprobs: list[torch.Tensor] = [
            get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
                cfg.safe_get_logprob,
            )
        ]
        entropies: list[torch.Tensor] = [
            gaussian_entropy(torch.ones_like(chains[:, 0]))
        ]
        for i in range(num_steps):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])
            cond_time = float(cond_times[i]) if cond_times is not None else sigma
            x_pre = chains[:, i]
            x_next = chains[:, i + 1]
            v_t = velocity_fn(x_pre, cond_time)
            noise_std = (
                noise_std_fn(x_pre, cond_time)
                if cfg.noise_method == "flow_noise"
                else None
            )
            mean, std = sample_mean_var_val(
                x_pre, v_t, sigma, sigma_next, cfg, train=True, noise_std=noise_std
            )
            logprobs.append(get_logprob_norm(x_next, mean, std, cfg.safe_get_logprob))
            entropies.append(
                gaussian_entropy(std)
                if cfg.noise_method == "flow_noise"
                else torch.zeros_like(x_next)
            )
        return (
            torch.stack(logprobs, dim=1).mean(dim=1),
            torch.stack(entropies, dim=1).mean(dim=1),
        )

    # --- single stochastic step, possibly per-sample (mixed micro-batch) ---
    if torch.is_tensor(denoise_index):
        idx = denoise_index.to(device).long().flatten()
        if idx.numel() == 1:
            idx = idx.expand(bsize)
    else:
        idx = torch.full((bsize,), int(denoise_index), device=device, dtype=torch.long)

    arange = torch.arange(bsize, device=device)
    x_pre = chains[arange, idx]
    x_next = chains[arange, idx + 1]
    sigma = sigmas.to(device)[idx]  # [B]
    sigma_next = sigmas.to(device)[idx + 1]  # [B]
    if cond_times is not None:
        cond_time = cond_times.to(device)[idx]  # [B]
    else:
        cond_time = sigma

    v_t = velocity_fn(x_pre, cond_time)
    noise_std = (
        noise_std_fn(x_pre, cond_time) if cfg.noise_method == "flow_noise" else None
    )
    mean, std = sample_mean_var_val(
        x_pre, v_t, sigma, sigma_next, cfg, train=True, noise_std=noise_std
    )
    logprob = get_logprob_norm(x_next, mean, std, cfg.safe_get_logprob)
    entropy = (
        gaussian_entropy(std)
        if cfg.noise_method == "flow_noise"
        else torch.zeros_like(x_next)
    )
    return logprob, entropy
