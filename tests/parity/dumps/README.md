# Parity msprobe dumps

Per-API statistics dumps produced by [MindStudio Probe](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe)
(`pip install mindstudio-probe`, version 26.0.0). One step per worker = one
call to `MultiStepRolloutWorker.predict()` on a deterministic, single-env
LIBERO-Spatial eval.

## Capture conditions

Reproducible via `tests/parity/eval_libero_npu_fp32.sh` with the
`tests/parity/configs/libero_spatial_openvlaoft_eval.yaml` config, which forces:

- `eval_rollout_epoch=1`, `env.eval.max_steps_per_rollout_epoch=8`
  (= `num_action_chunks` ⇒ `n_eval_chunk_steps=1`),
  `rollout.pipeline_stage_num=1`, `env.eval.total_num_envs=1`
  → exactly one `predict()` per process → one msprobe `step0` directory
- `temperature_eval=0` ⇒ `do_sample=False` inside
  `huggingface_worker.setup_sample_params` ⇒ argmax decode, no `multinomial`
- `seed_all(seed=1234, mode=True)` at module import of
  `huggingface_worker.py` ⇒ `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required by cuBLAS determinism)
- `env.eval.use_fixed_reset_state_ids=True` + `use_ordered_reset_state_ids=True`
  ⇒ env reset state determined by task config
- `enable_cuda_graph=False`, `enable_torch_compile=False`
- `MUJOCO_GL=egl` on this GPU host (this `gpu/` capture). For NPU hosts use
  `osmesa`; **the rendered image bytes differ between EGL and OSMesa, so a
  byte-level GPU↔NPU comparison requires both hosts to use the same renderer
  (or to bypass the env entirely via `tests/parity/openvla_oft/run_rlinf.py`
  with a golden input).**

## Layout

```
dumps/
└── gpu/                                       # NVIDIA A100 80GB SXM4
    ├── no_patch/                              # PARITY_DISABLE_FP32_ATTN=1
    │   └── step0/rank0/
    │       ├── construct.json                 # module/op tree
    │       ├── dump.json                      # per-API stats (mean/max/min/L2/…)
    │       └── stack.json                     # Python call stack per op
    └── bf16_patch/                            # PARITY_ATTN_DTYPE=bf16
        └── step0/rank0/                       # same files; the SDPA path is
                                               # decomposed by `fp32_attn_loader`
                                               # into matmul→softmax→matmul,
                                               # all in bf16 -- so this dump
                                               # captures the *manual* attention
                                               # ops one-by-one rather than a
                                               # single fused `scaled_dot_product_attention`.
```

`construct.json` describes the module call tree; `dump.json` contains
per-op input/output statistics; `stack.json` records the originating Python
frames. `msprobe compare` consumes the pair of `dump.json`s.

## Notes

- `eval/num_trajectories: 0` in the log is **expected**: with
  `max_steps_per_rollout_epoch=8` only one chunk action runs before the
  rollout epoch ends, so no LIBERO episode finishes and no `success_once`
  is recorded. This is the price of restricting to a single deterministic
  dump step; switching `max_steps_per_rollout_epoch` back to 512 would
  produce 64 dump steps per epoch *and* real `success_once`.
- The two GPU runs above used the same EGL renderer, same checkpoint, same
  seed; differences between them isolate the **attention math** (cuDNN
  flash/efficient vs. manual decomposed bf16).
