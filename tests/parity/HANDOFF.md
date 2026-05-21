# Parity handoff: GPU → NPU

The GPU side is done. To complete the comparison, pull these from this server
to your NPU host, then run the same scripts there and diff.

## What to download from this server

| Purpose | Path | Size | Notes |
| --- | --- | --- | --- |
| Test scripts | `RLinf/tests/parity/` (the whole dir) | ~30 KB | Everything except the goldens. |
| GPU goldens | `RLinf/tests/parity/goldens/*.pt` | ~705 MB total | One per (model × impl). |
| OpenVLA-OFT checkpoint | `/workspace/Openvla-oft-SFT-libero-spatial-traj1/` | ~28 GB | Skip if the NPU host already has it. |
| Wan checkpoint + dataset | `/workspace/RLinf-Wan-LIBERO-Spatial/` | ~10 GB | Skip if already on NPU. |

The four GPU goldens currently on disk:

```
openvla_oft_rlinf_gpu.pt        37M
openvla_oft_official_gpu.pt     37M
wan_upstream_pipeline_gpu.pt    313M
wan_rlinf_env_gpu.pt            319M
```

All four were verified **bit-exact across two repeats on the same GPU**, so any
diff against the NPU run is real hardware/kernel drift, not flakiness.

## How to use it on the NPU host

1. Lay the files out like this (paths are read by `common.py`, change them
   there if your NPU host stores the checkpoints elsewhere):

   ```
   <repo>/RLinf/                                # the repo with tests/parity/ inside
   /workspace/Openvla-oft-SFT-libero-spatial-traj1/
   /workspace/RLinf-Wan-LIBERO-Spatial/
   <repo>/RLinf/tests/parity/goldens/openvla_oft_rlinf_gpu.pt
   <repo>/RLinf/tests/parity/goldens/openvla_oft_official_gpu.pt
   <repo>/RLinf/tests/parity/goldens/wan_upstream_pipeline_gpu.pt
   <repo>/RLinf/tests/parity/goldens/wan_rlinf_env_gpu.pt
   ```

2. Generate the NPU goldens with the same scripts. `pick_device()` in
   `common.py` will pick `npu:0` automatically when `torch_npu` is installed:

   ```bash
   cd <repo>/RLinf
   export PYTHONPATH=$PWD:$PYTHONPATH
   export EMBODIED_PATH=$PWD/examples/embodiment

   # use whichever NPU-side python venvs you have
   OPENVLA_PY=/path/to/openvlaoft_libero_venv/bin/python \
   WAN_PY=/path/to/openvlaoft_wan_venv/bin/python \
       bash tests/parity/run_all.sh
   ```

   This drops `*_npu.pt` next to the GPU files.

3. Diff each pair:

   ```bash
   for n in openvla_oft_rlinf openvla_oft_official wan_rlinf_env wan_upstream_pipeline; do
       echo "=== $n ==="
       python tests/parity/compare.py \
           tests/parity/goldens/${n}_gpu.pt \
           tests/parity/goldens/${n}_npu.pt \
           --report tests/parity/goldens/${n}_diff.json
   done
   ```

   Each pair prints per-tensor bit-exact / allclose flags and downstream
   semantic metrics (chunk-action L1, token-agreement %, reward L1, video
   PSNR). The exit code is `0` for bit-exact / strict allclose, `1` for
   bf16-tolerant, `2` for larger drift.

## Reading the result

| What you see | What it means |
| --- | --- |
| Both OpenVLA-OFT goldens diverge, both Wan match | Your NPU eval-zero bug is in the VLA forward (kernel-level). Start with `pixel_values` and `prev_logprobs` deltas. |
| Both Wan goldens diverge, both OpenVLA match | The training-side problem is on the Wan side (DiT/VAE). |
| Only `wan_rlinf_env` diverges, `wan_upstream_pipeline` matches | The bug is in the RLinf env wrapper, not Wan upstream. |
| Only `openvla_oft_rlinf` diverges, `openvla_oft_official` matches | The bug is in RLinf's `predict_action_batch` adapter. |
| Everything diverges | Most likely a global bf16 matmul / attention kernel difference. Try `attn_implementation: eager` (already set) plus NPU op blacklist. |

## Sanity reminders

- The scripts pin determinism (`torch.use_deterministic_algorithms`, cudnn
  deterministic, `do_sample=False` for OFT, `seed=0` `rand_device="cpu"` for
  Wan). On GPU each script's two repeats agree bit-exactly — your NPU runs
  must also pass self-consistency before cross-device diffs mean anything.
- Inputs are sourced from `RLinf-Wan-LIBERO-Spatial/dataset/`. The fingerprint
  is printed at the top of every run; if it doesn't match between GPU and NPU
  the inputs themselves diverged and the comparison is invalid.
- Full description of the testing approach is in
  `tests/parity/README.md`.
