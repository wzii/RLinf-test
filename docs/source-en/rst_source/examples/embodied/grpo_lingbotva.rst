LingBot-VA Online RL (Flow-SDE / Flow-Noise GRPO)
=================================================

This document describes online RL fine-tuning of **LingBot-VA** on the
Libero-Object benchmark using RLinf's flow-matching RL algorithms
(**Flow-SDE** and **Flow-Noise**), built on top of the LingBot-VA SFT + eval
integration.

It reuses the same flow-RL machinery RLinf already ships for the ``openpi``
(:math:`\pi_0` / :math:`\pi_{0.5}`) and ``lingbotvla`` action experts, adapted
to the **action stream** of the LingBot-VA Wan 2.2 video-diffusion transformer.


Motivation
----------

LingBot-VA produces actions by iteratively denoising an action latent with a
flow-matching ODE (the ``action_scheduler.step`` Euler updates inside the eval
backend). A deterministic ODE has an *intractable* per-action log-likelihood,
so it cannot be optimised with policy-gradient RL directly. RLinf resolves this
in two ways (see ``π_RL``, arXiv:2510.25889, and Flow-GRPO, arXiv:2505.05470):

- **Flow-SDE** — convert the denoising ODE into an equivalent SDE. Each denoise
  step becomes a Gaussian transition with a closed-form log-prob; the SDE
  diffusion term provides exploration noise. No new parameters.

- **Flow-Noise** — keep the ODE mean but add a small *learnable* Gaussian noise
  network, modelling denoising as a discrete-time MDP with an exact log-prob.

Both turn the multi-step denoiser into a sequence of Gaussian transitions that
GRPO optimises, exactly as for ``lingbotvla``.


How it maps onto LingBot-VA
---------------------------

The Wan ``FlowMatchScheduler`` parameterises the forward process as
``x_sigma = (1 - sigma) * x0 + sigma * x1`` with ``x1 ~ N(0, I)`` and predicts
the velocity ``v = x1 - x0``. This is the same convention RLinf's other action
experts use, with continuous time ``t`` replaced by the scheduler's ``sigma``.
The per-step Gaussian transition (``rlinf/models/embodiment/lingbotva/flow_rl.py``)
is therefore identical math::

    x0_pred = x_t - sigma * v
    x1_pred = x_t + (1 - sigma) * v

    # Flow-SDE
    sigma_i  = noise_level * sqrt(sigma / (1 - sigma))
    mean     = x0_pred * (1 - sigma_next)
             + x1_pred * (sigma_next - sigma_i**2 * delta / (2 * sigma))
    std      = sqrt(delta) * sigma_i

    # Flow-Noise
    mean     = x0_pred * (1 - sigma_next) + x1_pred * sigma_next
    std      = noise_head(sigma)          # learnable

Following Flow-GRPO, only **one** denoise step per rollout is sampled
stochastically (the rest are deterministic ODE steps) and only that step's
log-prob enters the policy gradient; ``joint_logprob: True`` switches to the
all-steps variant.

Two execution paths
~~~~~~~~~~~~~~~~~~~~~

Unlike ``lingbotvla`` (a single ``nn.Module``), LingBot-VA has two disjoint
transformer paths that the weight syncer keeps in lockstep:

- **Rollout worker** (``lingbotva.training_mode=False``) — samples actions via
  the eval backend (``VA_Server``). ``predict_action_batch(mode="train")`` runs
  the deterministic video denoise to warm the KV cache, then the stochastic
  action denoise, returning real ``prev_logprobs`` plus the ``chains`` and
  conditioning needed for recompute.
- **Actor worker** (``lingbotva.training_mode=True``) — exposes the transformer
  to FSDP. ``default_forward`` re-evaluates the velocity (and learnable noise
  std) for the saved ``chains`` under gradient and returns the GRPO
  ``logprobs`` / ``values`` / ``entropy``.


Configuration
-------------

Recommended config:

- ``examples/embodiment/config/libero_object_grpo_lingbotva.yaml``

Key knobs (under ``actor.model``):

.. code:: yaml

   noise_method: flow_sde   # flow_sde | flow_noise | flow_cps
   noise_level: 0.7         # SDE diffusion scale (or CPS rotation angle)
   joint_logprob: False     # single-step (default) vs all-step log-prob
   add_value_head: False    # GRPO is critic-free
   lingbotva:
     training_mode: True               # actor: FSDP transformer
     action_num_inference_steps: 10    # action denoise steps

Switch to Flow-Noise with a single override::

   bash examples/embodiment/run_embodiment.sh libero_object_grpo_lingbotva \
       actor.model.noise_method=flow_noise rollout.model.noise_method=flow_noise \
       algorithm.entropy_bonus=0.01

The rollout model mirrors the actor's ``noise_method`` / ``noise_level`` so the
collected ``prev_logprobs`` and the recomputed log-probs are consistent (GRPO
assumes an importance ratio of 1 at collection time).


Verification status
-------------------

- **Flow-RL core math** — unit-tested in
  ``tests/unit_tests/test_lingbotva_flow_rl.py`` (Flow-SDE / Flow-Noise /
  Flow-CPS / Flow-ODE transitions, Gaussian log-prob vs ``torch.distributions``,
  the ODE limit as ``noise_level → 0``, the learnable noise head, and — most
  importantly — rollout/recompute log-prob consistency). These run on a bare
  CPU ``torch`` install.
- **Wan-transformer plumbing** — the rollout/recompute transformer calls and
  the actor-side KV-cache warm-up mirror the eval backend's deterministic loop
  but require a GPU, the real ``wan_va`` package, and an SFT checkpoint to
  validate end-to-end. The single ``wan_va``-coupled recompute step is flagged
  with a ``note`` in ``LingbotVAActionModel._warm_video_cache``.
