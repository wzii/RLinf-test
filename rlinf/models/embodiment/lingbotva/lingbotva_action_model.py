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

"""LingBot-VA action model adapter for RLinf Libero eval and SFT training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lingbotva import flow_rl as _flow_rl
from rlinf.models.embodiment.lingbotva._utils import (
    extend_import_path,
    load_transformer_state_dict,
)
from rlinf.models.embodiment.lingbotva.eval_adapter.history_buffer import (
    LingbotVAEpisodeState,
)
from rlinf.models.embodiment.lingbotva.eval_adapter.native_backend import (
    LingbotVALiberoBackend,
)
from rlinf.models.embodiment.lingbotva.eval_adapter.observation_adapter import (
    LingbotVALiberoObservationAdapter,
)


class LingbotVAActionModel(nn.Module, BasePolicy):
    """LingBot-VA adapter for the Libero suite.

    Two operating modes:

    * **Eval (default)** — :meth:`predict_action_batch` runs one diffusion-based
      inference per call, wrapping ``wan_va.wan_va_server.VA_Server`` in-process
      via :class:`LingbotVALiberoBackend`. The transformer lives inside the
      backend and is invisible to FSDP.

    * **Training** (``cfg.lingbotva.training_mode=True``) —
      :meth:`sft_forward` runs one flow-matching diffusion step on the
      pre-extracted latent + action targets from the dataloader. The
      transformer is exposed as a direct ``nn.Module`` child so that FSDP can
      wrap it and the SFT worker can drive the optimizer. VAE and text encoder
      are not loaded (the dataset provides cached latents and the empty UMT5
      embedding).
    """

    # FSDP wrap policy reads `_no_split_modules` from the top-level model.
    # `WanTransformer3DModel` already declares `["WanTransformerBlock"]`
    # internally; we re-declare here for the outer wrapper.
    _no_split_modules = [
        "WanTransformerBlock",
        "WanAttention",
        "WanTransformer3DModel",
    ]

    def __init__(self, cfg: Any, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = cfg
        self.torch_dtype = torch_dtype
        self.action_dim = int(getattr(cfg, "action_dim", 7))
        self.action_per_frame = int(getattr(cfg.lingbotva, "action_per_frame", 4))
        # Number of action steps to actually execute per inference. Defaults to
        # ``(frame_chunk_size - 1) * action_per_frame`` (12 for Libero), but
        # callers can lower it to force shorter chunks with more frequent
        # replanning.
        self.exec_steps_per_chunk = int(
            getattr(cfg, "num_action_chunks", 3 * self.action_per_frame)
        )
        # When ``False`` we always restart from ``frame_st_id=0`` and treat each
        # chunk as the model's first chunk. When ``True`` we maintain
        # per-environment KV-cache history (chunked obs + prior action) and
        # replay it on each call so the model keeps long-horizon context.
        self.enable_kv_cache_replay = bool(
            getattr(cfg.lingbotva, "enable_kv_cache_replay", False)
        )
        self.training_mode = bool(
            getattr(cfg.lingbotva, "training_mode", False)
        )

        self._backend: LingbotVALiberoBackend | None = None
        self._episode_states: dict[int, LingbotVAEpisodeState] = {}
        self._ac_applied = False

        # ---- Flow-matching RL (Flow-SDE / Flow-Noise) ----
        # Resolved from cfg so the same model object can serve eval, SFT, and
        # online RL. ``flow_ode`` (the default for non-RL paths) keeps
        # predict_action_batch fully deterministic.
        self.flow_rl_cfg = _flow_rl.FlowRLConfig(
            noise_method=str(getattr(cfg, "noise_method", "flow_ode")),
            noise_level=float(getattr(cfg, "noise_level", 0.7)),
            joint_logprob=bool(getattr(cfg, "joint_logprob", False)),
            ignore_last=bool(getattr(cfg, "ignore_last", False)),
            safe_get_logprob=bool(getattr(cfg, "safe_get_logprob", False)),
        )
        self.noise_method = self.flow_rl_cfg.noise_method
        self.add_value_head = bool(getattr(cfg, "add_value_head", False))

        # Flow-Noise learnable noise network. Created unconditionally for
        # flow_noise so it exists on both the rollout and actor workers and is
        # picked up by the weight syncer.
        self.noise_head: nn.Module | None = None
        if self.noise_method == "flow_noise":
            self.noise_head = _flow_rl.LearnableNoiseHead(action_dim=self.action_dim)

        # Optional critic head on the (mean-pooled) clean action latent.
        self.value_head: nn.Module | None = None
        if self.add_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(self.action_dim, 128), nn.SiLU(), nn.Linear(128, 1)
            )

        if self.training_mode:
            self._load_transformer_for_training()

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"LingBot-VA does not support forward_type={forward_type}."
        )

    # ``default_forward`` is defined near the end of the class -- it dispatches
    # the flow-matching RL actor update (recompute log-prob / value / entropy).

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def _load_transformer_for_training(self) -> None:
        """Load only the transformer + flow-matching schedulers (no VAE / TE)."""
        repo_path = Path(self.config.lingbotva.repo_path)
        extend_import_path(repo_path)
        from wan_va.modules.utils import load_transformer

        transformer_path = os.path.join(self.config.model_path, "transformer")
        # The reference recipe loads fp32 then immediately casts to bf16 in
        # `shard_model` (`model.to(dtype).cuda()`). We collapse that into a
        # single bf16 load to keep peak host memory low and to avoid a 5 B
        # fp32 model ever landing on the GPU.
        # `attn_mode` defaults to "flex" to match the reference run; on
        # torch < 2.9 the flex_attention + inductor combination hits a
        # known codegen bug, so callers can override to "torch" (SDPA).
        attn_mode = getattr(self.config.lingbotva, "attn_mode", "flex")
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=self.torch_dtype,
            torch_device="cpu",
            attn_mode=attn_mode,
        )
        # `from_pretrained(torch_dtype=...)` casts checkpoint tensors but
        # leaves freshly-initialised parameters (e.g. `scale_shift_table`)
        # in their default fp32. FSDP1's flat-param refuses to flatten a
        # module with mixed dtypes, so force the whole module to bf16.
        self.transformer.to(dtype=self.torch_dtype)

        # Optional SFT-checkpoint override (rare for first-stage SFT; useful
        # when resuming or fine-tuning further).
        override_path = getattr(
            self.config.lingbotva, "transformer_state_dict_path", None
        )
        if override_path:
            load_transformer_state_dict(self.transformer, override_path)

        self._build_train_schedulers()
        # Cache values needed by the loss / noise machinery.
        self._patch_size = (1, 2, 2)  # va_shared_cfg.patch_size
        self._snr_shift = 5.0  # va_libero_cfg.snr_shift
        self._action_snr_shift = 0.05  # va_libero_cfg.action_snr_shift

    def _build_train_schedulers(self) -> None:
        from wan_va.utils import FlowMatchScheduler

        self._train_scheduler_latent = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True
        )
        self._train_scheduler_latent.set_timesteps(1000, training=True)
        self._train_scheduler_action = FlowMatchScheduler(
            shift=0.05, sigma_min=0.0, extra_one_step=True
        )
        self._train_scheduler_action.set_timesteps(1000, training=True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        if self._ac_applied:
            return
        if not self.training_mode:
            raise RuntimeError(
                "gradient_checkpointing_enable is only valid in training_mode."
            )
        from wan_va.distributed.fsdp import apply_ac

        apply_ac(self.transformer)
        self._ac_applied = True

    @torch.no_grad()
    def _add_noise(
        self,
        latent: torch.Tensor,
        train_scheduler,
        action_mask: torch.Tensor | None = None,
        action_mode: bool = False,
        noisy_cond_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Sample flow-matching noise + timesteps. Mirrored from wan_va/train.py:_add_noise."""
        from wan_va.utils import get_mesh_id, sample_timestep_id

        B, _C, FrameDim, _H, _W = latent.shape
        device = latent.device

        timestep_ids = sample_timestep_id(
            batch_size=FrameDim,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self._patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=FrameDim,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(
                device=device
            )
            latent = train_scheduler.add_noise(
                latent, noise, cond_timesteps, t_dim=2
            )
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict: dict) -> dict:
        """Build the transformer input dict. Mirrored from wan_va/train.py:_prepare_input_dict."""
        latent_dict = self._add_noise(
            latent=batch_dict["latents"],
            train_scheduler=self._train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self._train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        latent_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": torch.randint(1, 5, (1,)).item(),
            "window_size": torch.randint(4, 65, (1,)).item(),
        }

    def _compute_loss(self, input_dict: dict, pred) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-frame flow-matching MSE on latent + action streams.

        Mirrored from wan_va/train.py:compute_loss, but without the
        ``/ gradient_accumulation_steps`` division (RLinf's FSDPSftWorker
        handles grad-accumulation by dividing the loss internally).
        """
        from einops import rearrange
        from wan_va.utils import data_seq_to_patch

        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self._patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
        latent_loss_weight = self._train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)
        action_loss_weight = self._train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (
            latent_loss_per_frame / (latent_mask_per_frame + 1e-6)
        ).mean()

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1)
        action_mask = input_dict["action_dict"]["actions_mask"].float().permute(
            0, 2, 3, 4, 1
        )
        action_loss = action_loss.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (
            action_loss_per_frame / (action_mask_per_frame + 1e-6)
        ).mean()

        return latent_loss, action_loss

    def sft_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        if not self.training_mode:
            raise RuntimeError(
                "sft_forward called but cfg.lingbotva.training_mode is False."
            )

        device = next(self.transformer.parameters()).device
        batch = {
            k: v.to(device) for k, v in data.items() if torch.is_tensor(v)
        }
        input_dict = self._prepare_input_dict(batch)
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self._compute_loss(input_dict, output)
        return {
            "loss": latent_loss + action_loss,
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Eval path (unchanged from the milestone-1 integration)
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> LingbotVALiberoBackend:
        if self.training_mode:
            raise RuntimeError(
                "Eval backend not available in training_mode; call sft_forward instead."
            )
        if self._backend is None:
            self._backend = LingbotVALiberoBackend(self.config, self.torch_dtype)
        return self._backend

    def _get_state(self, env_idx: int) -> LingbotVAEpisodeState:
        if env_idx not in self._episode_states:
            self._episode_states[env_idx] = LingbotVAEpisodeState()
        return self._episode_states[env_idx]

    @staticmethod
    def _get_prompt(env_obs: dict[str, Any], env_idx: int) -> str:
        prompts = env_obs.get("task_descriptions")
        if prompts is None:
            raise ValueError(
                "LingBot-VA requires task_descriptions in env observations."
            )
        return str(prompts[env_idx])

    @staticmethod
    def _select_executable_actions(
        raw_action: np.ndarray, first_chunk: bool
    ) -> np.ndarray:
        """Convert raw model output to a flat sequence of env actions.

        Args:
            raw_action: array of shape
                ``(action_dim, frame_chunk_size, action_per_frame)``.
            first_chunk: if True the leading frame is skipped to match the
                LingBot-VA Libero client behaviour.

        Returns:
            Array of shape ``(num_actions, action_dim)``.
        """
        start_idx = 1 if first_chunk else 0
        selected = raw_action[:, start_idx:, :]
        return np.transpose(selected, (1, 2, 0)).reshape(-1, raw_action.shape[0])

    def reset_episode(self, env_idx: int, prompt: str | None = None) -> None:
        """Drop cached state for the given env so the next call is a first chunk."""
        self._get_state(env_idx).reset(prompt or "")

    def record_chunk_observations(
        self,
        env_idx: int,
        chunk_obs_list: list[dict[str, Any]],
        prev_model_action: np.ndarray,
    ) -> None:
        """Push the previous chunk's observations into the KV-cache history.

        ``chunk_obs_list`` is the per-step raw libero observation dicts (with
        ``agentview_image`` and ``robot0_eye_in_hand_image`` keys). We pick the
        configured key frames (every ``action_per_frame`` steps) and pair them
        with the raw model action that produced them so the backend can replay
        them into the transformer's KV cache on the next inference.
        """
        if not self.enable_kv_cache_replay:
            return
        state = self._get_state(env_idx)
        if state.prompt is None:
            return
        # The LingBot-VA clients sample key frames every ``action_per_frame // 4``
        # env steps (so they record 4 key frames per video frame predicted by
        # the model). For Libero that period is 1 (every step), for RoboTwin
        # 4. Fall back to 1 if the integer division would round to 0.
        period = max(1, self.action_per_frame // 4)
        key_frames: list[dict[str, Any]] = []
        for step_idx, raw_obs in enumerate(chunk_obs_list):
            if (step_idx + 1) % period != 0:
                continue
            key_frames.append(
                LingbotVALiberoObservationAdapter.format_raw_step_observation(
                    raw_obs=raw_obs,
                    prompt=state.prompt,
                )
            )
        if not key_frames:
            return
        state.kv_cache_history.append(
            (key_frames, np.asarray(prev_model_action, dtype=np.float32).copy())
        )

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **kwargs: Any
    ):
        if mode in ("train", "rollout"):
            return self._predict_action_batch_rl(env_obs, **kwargs)
        if mode != "eval":
            raise NotImplementedError(
                "LingBot-VA Libero adapter supports eval / train (RL) modes only."
            )
        if self.training_mode:
            raise RuntimeError(
                "predict_action_batch is unavailable while training_mode=True; "
                "the eval backend is not initialised."
            )
        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA requires batched states in env observations.")
        batch_size = states_tensor.shape[0]

        backend = self._ensure_backend()

        prompts: list[str] = []
        obs_batch: list[dict[str, Any]] = []
        kv_cache_histories: list[list[tuple[list[dict[str, Any]], np.ndarray]]] = []
        first_chunk_flags: list[bool] = []

        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            state = self._get_state(env_idx)
            if state.prompt != prompt:
                state.reset(prompt)
            obs = LingbotVALiberoObservationAdapter.format_observation(
                env_obs, env_idx, prompt
            )
            if state.first_obs is None:
                state.first_obs = obs
            prompts.append(prompt)
            # When KV-cache replay is enabled we reuse the cached first_obs to
            # keep the obs grid consistent with the replayed history.
            obs_batch.append(state.first_obs if self.enable_kv_cache_replay else obs)
            kv_cache_histories.append(list(state.kv_cache_history))
            first_chunk_flags.append(state.first_chunk)

        replay_groups_match = all(
            len(history) == len(kv_cache_histories[0]) for history in kv_cache_histories
        )
        if (
            self.enable_kv_cache_replay
            and replay_groups_match
            and any(len(h) > 0 for h in kv_cache_histories)
        ):
            raw_actions = backend.infer_batch(
                obs_batch, prompts, kv_cache_histories=kv_cache_histories
            )
            current_first_chunk = False
        else:
            raw_actions = backend.infer_batch(obs_batch, prompts)
            current_first_chunk = any(first_chunk_flags)

        chunks: list[torch.Tensor] = []
        for env_idx, raw_action in enumerate(raw_actions):
            state = self._get_state(env_idx)
            # Without KV-cache replay every call rebuilds context from the
            # current observation, so the model is effectively starting a
            # fresh "first chunk" each time and we must drop the leading
            # placeholder frame. Once we are using KV replay we follow the
            # LingBot-VA client semantics: skip the frame only on the very
            # first inference per episode.
            is_first_chunk_for_selection = (
                state.first_chunk if self.enable_kv_cache_replay else True
            )
            env_actions = self._select_executable_actions(
                raw_action, first_chunk=is_first_chunk_for_selection
            )
            limit = min(self.exec_steps_per_chunk, env_actions.shape[0])
            chunks.append(torch.from_numpy(env_actions[:limit]))
            # Track this chunk's raw model action so callers can hand it back
            # via ``record_chunk_observations`` for KV replay on the next call.
            state.prev_model_action = raw_action.astype(np.float32)
            state.last_action_per_frame = raw_action.shape[2]
            state.first_chunk = False

        # In multi-env mode envs can be in different first-chunk states; we
        # truncate to the shortest chunk so we can stack into a single tensor.
        common_len = min(c.shape[0] for c in chunks)
        chunks = [c[:common_len] for c in chunks]

        action_tensor = torch.stack(chunks, dim=0).to(dtype=torch.float32)
        zeros = torch.zeros(action_tensor.shape[:2], dtype=torch.float32)
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {"action": action_tensor},
        }
        return action_tensor, result

    # ------------------------------------------------------------------
    # Flow-matching RL: rollout (Flow-SDE / Flow-Noise)
    # ------------------------------------------------------------------

    def _noise_std_fn(self):
        """Bound ``self.noise_head`` into the ``(x_t, sigma) -> std`` callable."""
        if self.noise_head is None:
            return None
        return lambda x_t, sigma: self.noise_head(x_t, sigma)

    @staticmethod
    def _latent_logprob_to_steps(
        lp_latent: torch.Tensor, num_exec: int, first_chunk: bool
    ) -> torch.Tensor:
        """Map a per-element latent log-prob to per-executed-step log-probs.

        ``lp_latent`` is ``[B, action_dim, F, A, 1]`` (the Gaussian log-density
        of the sampled action latent). Executable actions are the latent's
        ``(F, A)`` grid flattened with the leading frame optionally dropped (see
        :meth:`_select_executable_actions`). Summing over ``action_dim`` yields
        the joint log-prob of each executed action step, shape ``[B, num_exec]``,
        so it lines up element-wise with the executed action tensor used by GRPO.
        """
        lp = lp_latent.squeeze(-1).sum(dim=1)  # [B, F, A] (sum over action_dim)
        start = 1 if first_chunk else 0
        lp = lp[:, start:, :]
        lp = lp.reshape(lp.shape[0], -1)  # [B, (F-start)*A]
        return lp[:, :num_exec]

    def _predict_action_batch_rl(self, env_obs: dict[str, Any], **_: Any):
        """Online-RL rollout: stochastic flow denoising with log-probs.

        Mirrors the eval path's I/O contract but returns *real* per-step
        log-probs (and values) plus the ``forward_inputs`` the actor needs to
        recompute them under gradient. Runs on the rollout worker, where the
        eval backend (the frozen inference transformer) is available.
        """
        if self.training_mode:
            raise RuntimeError(
                "RL rollout uses the eval backend; do not call it with "
                "training_mode=True (that path is the actor update)."
            )
        if self.noise_method == "flow_ode":
            raise ValueError(
                "predict_action_batch(mode='train') needs a stochastic "
                "noise_method (flow_sde / flow_noise / flow_cps), got flow_ode."
            )

        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA RL rollout requires batched states.")
        batch_size = states_tensor.shape[0]
        backend = self._ensure_backend()

        prompts: list[str] = []
        obs_batch: list[dict[str, Any]] = []
        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            state = self._get_state(env_idx)
            if state.prompt != prompt:
                state.reset(prompt)
            obs = LingbotVALiberoObservationAdapter.format_observation(
                env_obs, env_idx, prompt
            )
            prompts.append(prompt)
            obs_batch.append(obs)

        rollout = backend.rl_rollout(
            obs_batch,
            prompts,
            self.flow_rl_cfg,
            noise_std_fn=self._noise_std_fn(),
        )

        # Build the executed action tensor (env-facing), exactly as eval does.
        chunks: list[torch.Tensor] = []
        for env_idx, raw_action in enumerate(rollout["actions"]):
            env_actions = self._select_executable_actions(raw_action, first_chunk=True)
            limit = min(self.exec_steps_per_chunk, env_actions.shape[0])
            chunks.append(torch.from_numpy(env_actions[:limit]))
            self._get_state(env_idx).first_chunk = False
        common_len = min(c.shape[0] for c in chunks)
        chunks = [c[:common_len] for c in chunks]
        action_tensor = torch.stack(chunks, dim=0).to(dtype=torch.float32)

        prev_logprobs = self._latent_logprob_to_steps(
            rollout["logprobs"], common_len, first_chunk=True
        )
        if self.value_head is not None:
            prev_values = self._value_from_latent(rollout["raw_actions"]).expand(
                -1, common_len
            )
        else:
            prev_values = torch.zeros_like(prev_logprobs)

        # All forward_inputs values must be batch-leading tensors -- the rollout
        # worker splits every entry with ``torch.split(value, sizes, dim=0)``.
        # ``cond`` is already flattened into ``cond_*`` batch-leading tensors by
        # the backend; per-sample scalars are broadcast to ``[B]``.
        B = action_tensor.shape[0]
        forward_inputs = {
            "chains": rollout["chains"],
            "denoise_index": torch.full((B,), int(rollout["denoise_index"])),
            "exec_len": torch.full((B,), int(common_len)),
            "first_chunk": torch.ones(B, dtype=torch.bool),
        }
        forward_inputs.update(rollout["cond"])
        result = {
            "prev_logprobs": prev_logprobs.to(torch.float32),
            "prev_values": prev_values.to(torch.float32),
            "forward_inputs": forward_inputs,
        }
        return action_tensor, result

    # ------------------------------------------------------------------
    # Flow-matching RL: actor update (recompute log-probs under gradient)
    # ------------------------------------------------------------------

    def _value_from_latent(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Critic value from the mean-pooled clean action latent -> ``[B, 1]``."""
        device = next(self.value_head.parameters()).device
        pooled = (
            raw_actions.to(device).squeeze(-1).mean(dim=(2, 3))
        )  # [B, action_dim]
        return self.value_head(pooled.to(self.torch_dtype)).to(torch.float32)

    @staticmethod
    def _unpack_meta(meta_row: torch.Tensor) -> dict[str, int]:
        keys = [
            "frame_st_id",
            "use_cfg",
            "attn_window",
            "frame_chunk_size",
            "action_per_frame",
            "latent_token_per_chunk",
        ]
        return {k: int(meta_row[i].item()) for i, k in enumerate(keys)}

    def _warm_video_cache(self, cond: dict[str, Any], meta: dict[str, int], device) -> None:
        """Repopulate ``self.transformer``'s KV cache with the rollout's video
        context so the action-stream recompute sees the same conditioning.

        .. note::
            This is the single ``wan_va``-coupled step of the recompute path and
            must be validated on GPU with the real package. It mirrors the
            (deterministic) final video forward of
            :meth:`...native_backend.LingbotVALiberoBackend.rl_rollout`.
        """
        tfm = self.transformer
        cache_name = "rlinf_rl_actor"
        bsz = cond["cond_video_latents"].shape[0]
        cfg_bsz = bsz * (2 if meta["use_cfg"] else 1)
        try:
            tfm.clear_cache(cache_name)
        except Exception:
            pass
        tfm.create_empty_cache(
            cache_name,
            meta["attn_window"],
            meta["latent_token_per_chunk"],
            meta["frame_chunk_size"] * meta["action_per_frame"],
            dtype=self.torch_dtype,
            device=device,
            batch_size=cfg_bsz,
        )
        latents = cond["cond_video_latents"].to(device).to(self.torch_dtype)
        grid = cond["cond_video_grid"].to(device)
        text = cond["cond_text_emb"].to(device).to(self.torch_dtype)
        timesteps = torch.zeros(
            (bsz, latents.shape[2]), device=device, dtype=torch.float32
        )
        if meta["use_cfg"]:
            latents = latents.repeat(2, 1, 1, 1, 1)
            grid = grid.repeat(2, 1, 1)
            text = text.repeat(2, 1, 1)
            timesteps = timesteps.repeat(2, 1)
        video_in = {
            "noisy_latents": latents,
            "timesteps": timesteps,
            "grid_id": grid,
            "text_emb": text,
        }
        tfm(video_in, update_cache=1, cache_name=cache_name, action_mode=False)
        self._rl_cache_name = cache_name

    def _recompute_action_velocity(
        self, x_t: torch.Tensor, raw_t, cond: dict[str, Any],
        meta: dict[str, int], device,
    ) -> torch.Tensor:
        """Velocity of the action stream under ``self.transformer`` (with grad).

        ``raw_t`` is the scheduler-native timestep: a float (shared step) or a
        per-sample ``[B]`` tensor (mixed micro-batch).
        """
        bsz = x_t.shape[0]
        n_tok = meta["frame_chunk_size"] * meta["action_per_frame"]
        noisy = x_t.to(self.torch_dtype)
        grid = cond["cond_action_grid"].to(device)
        text = cond["cond_text_emb"].to(self.torch_dtype).to(device)
        if torch.is_tensor(raw_t) and raw_t.dim() > 0:
            timesteps = raw_t.to(device, torch.float32).flatten()[:, None].expand(bsz, n_tok)
        else:
            val = float(raw_t) if not torch.is_tensor(raw_t) else float(raw_t.item())
            timesteps = torch.full((bsz, n_tok), val, device=device, dtype=torch.float32)
        if meta["use_cfg"]:
            noisy = noisy.repeat(2, 1, 1, 1, 1)
            grid = grid.repeat(2, 1, 1)
            text = text.repeat(2, 1, 1)
            timesteps = timesteps.repeat(2, 1)
        action_in = {
            "noisy_latents": noisy,
            "timesteps": timesteps,
            "grid_id": grid,
            "text_emb": text,
        }
        out = self.transformer(
            action_in,
            update_cache=0,
            cache_name=getattr(self, "_rl_cache_name", "rlinf_rl_actor"),
            action_mode=True,
        )
        # Reshape [B*, (F*A), C] -> [B, action_dim, F, A, 1] + CFG combine.
        out = (
            out.unflatten(1, (meta["frame_chunk_size"], meta["action_per_frame"]))
            .permute(0, 3, 1, 2)
            .unsqueeze(-1)
        )
        out = out[:bsz]
        return out.to(torch.float32)

    def default_forward(self, forward_inputs=None, **kwargs):  # type: ignore[override]
        """Actor update: recompute log-prob / value / entropy for GRPO.

        When called with RL ``forward_inputs`` (containing ``chains``) this runs
        the flow-RL recompute; otherwise it falls back to the original
        not-supported behaviour.
        """
        if forward_inputs is None or "chains" not in forward_inputs:
            raise NotImplementedError(
                "LingBot-VA default_forward needs RL forward_inputs with 'chains'."
            )
        device = next(self.transformer.parameters()).device
        chains = forward_inputs["chains"].to(device)
        denoise_index = forward_inputs["denoise_index"]
        first_chunk = bool(forward_inputs["first_chunk"].flatten()[0].item())
        exec_len = int(forward_inputs["exec_len"].flatten()[0].item())
        meta = self._unpack_meta(forward_inputs["cond_meta"][0])
        # Per-sample schedules are identical within a micro-batch; take row 0.
        sigmas = forward_inputs["cond_sigmas"][0].to(device)
        raw_timesteps = forward_inputs["cond_raw_timesteps"][0].to(device)
        cond = {
            k: v for k, v in forward_inputs.items() if k.startswith("cond_")
        }

        # Warm the KV cache once with the (fixed) video context, then recompute.
        self._warm_video_cache(cond, meta, device)

        logprobs, entropy = _flow_rl.recompute_logprob_entropy(
            chains,
            sigmas,
            denoise_index,
            velocity_fn=lambda x, t: self._recompute_action_velocity(
                x, t, cond, meta, device
            ),
            cfg=self.flow_rl_cfg,
            noise_std_fn=self._noise_std_fn(),
            cond_times=raw_timesteps,
        )

        lp = self._latent_logprob_to_steps(logprobs, exec_len, first_chunk)
        ent = self._latent_logprob_to_steps(entropy, exec_len, first_chunk)
        if self.value_head is not None:
            x0 = chains[:, -1]
            values = self._value_from_latent(x0).expand(-1, exec_len)
        else:
            values = torch.zeros_like(lp)

        return {
            "logprobs": lp.to(torch.float32),
            "values": values.to(torch.float32),
            "entropy": ent.to(torch.float32),
        }
