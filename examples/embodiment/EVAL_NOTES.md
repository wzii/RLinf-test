# LIBERO eval — known issues and workarounds

This file documents the runtime issues encountered when running
`eval_embodiment.sh` (or `eval_embodied_agent.py`) for LIBERO Spatial with the
OpenVLA-OFT model inside a containerised single-GPU environment, and the fixes
that were applied.

---

## 1. MuJoCo crash — too many OS threads (cgroup `pids.max`)

**Symptom**

```
RuntimeError: Caught an unknown exception!
```
inside `mujoco.MjModel.from_xml_string` during env initialisation when running
more than ~10 concurrent LIBERO env workers.

**Root cause**

Each env subprocess inherits the default `OMP_NUM_THREADS`, which MuJoCo sets
to `nproc` (often 64–127). With 50 concurrent env workers that exceeds the
container's `pids.max` limit (e.g. 3328), exhausting the thread-stack virtual
address space.

**Fix — `rlinf/envs/libero/venv.py`**

The `_worker()` function now unconditionally sets `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` at the very top of each spawned subprocess, before any
MuJoCo import. Override with `RLINF_ENV_OMP_THREADS` if you need more threads
per worker.

Additionally, `ReconfigureSubprocEnv` now spawns workers in batches (default
10 per batch, 2 s between batches) to limit simultaneous resource pressure.
Override with `RLINF_ENV_SPAWN_BATCH` and `RLINF_ENV_SPAWN_DELAY`.

---

## 2. NCCL watchdog crash — `Resource temporarily unavailable` (EAGAIN)

**Symptom**

```
[PG ID 3 PG GUID cg-Env:0-EnvGroup:0nccl_send_0 Rank 0]
Process group watchdog thread terminated with exception:
Resource temporarily unavailable
```
followed by SIGABRT, occurring shortly after env workers are created.

**Root cause**

`MultiChannelProcessGroup` always creates NCCL process groups (in addition to
GLOO) whenever GPUs are available, even though LIBERO observations are CPU
tensors and never use NCCL. The NCCL rendezvous between `Env:0` and
`EnvGroup:0` (both running on the same `CUDA_VISIBLE_DEVICES=0`) hits EAGAIN
during the socket handshake. By default, NCCL's watchdog thread then aborts the
entire process.

**Fix — launch environment variable**

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_ASYNC_ERROR_HANDLING=0   # legacy alias
```

This disables the watchdog abort. The NCCL group creation still succeeds; the
idle NCCL groups do not affect correctness because actual data (CPU tensors)
flows through GLOO.

---

## 3. Eval config — 1 env × N epochs instead of N envs × 1 epoch

To work within these constraints the eval configs use a single env worker and
iterate over more epochs rather than parallelising across multiple envs.  The
configs in `config/libero_spatial_openvlaoft_eval.yaml` and
`config/libero_spatial_openvlaoft_fp32attn_eval.yaml` use:

```yaml
env:
  eval:
    total_num_envs: 1
algorithm:
  eval_rollout_epoch: 50   # 50 epochs × 1 env = all 50 LIBERO-Spatial task/state combos
```

This covers the full evaluation suite at the cost of sequential evaluation
(~20 min on a single A100).

---

## 4. Launch command

```bash
cd examples/embodiment
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard \
HYDRA_FULL_ERROR=1 \
EMBODIED_PATH=$PWD \
PYTHONPATH=/workspace/RLinf:$PYTHONPATH \
TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
NCCL_ASYNC_ERROR_HANDLING=0 \
/workspace/RLinf/openvlaoft_libero_venv/bin/python eval_embodied_agent.py \
  --config-path config/ \
  --config-name libero_spatial_openvlaoft_eval \
  runner.logger.log_path=/workspace/RLinf/logs/eval_run
```

Replace `--config-name` with `libero_spatial_openvlaoft_fp32attn_eval` for the
fp32-attention variant.

---

## 5. Baseline eval results

Checkpoint: `Openvla-oft-SFT-libero-spatial-traj1`  
Dataset: LIBERO Spatial (50 task/state combos, `use_ordered_reset_state_ids=True`)

| Config | `success_once` | `success_at_end` | trajectories |
|---|---|---|---|
| bf16 attention (default) | **66 %** | 40 % | 50 |
| fp32 attention patch | **68 %** | 30 % | 50 |

The ±2 pp difference in `success_once` is within the ±7 pp statistical noise
floor for 50 trajectories. The fp32 patch has no significant effect on GPU;
see `tests/parity/README.md` for its purpose on NPU.
