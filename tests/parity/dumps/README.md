# Parity msprobe dumps

Per-API statistics dumps produced by [MindStudio Probe](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe)
(`pip install mindstudio-probe`, version 26.0.0).

## How these were captured

Via `tests/parity/openvla_oft/run_rlinf_msprobe.py` — the parity-path
runner that loads OpenVLA-OFT directly and forwards a **fixed, golden,
byte-identical input batch** (the same 32 samples used by
`run_rlinf.py`, loaded from the Wan checkpoint's bundled `.npy` files).

**No LIBERO env, no MuJoCo, no Ray, no rendering.** That removes the
biggest cross-architecture noise sources (EGL vs OSMesa, x86 vs aarch64
FPU on physics) before they can pollute downstream per-API statistics.

Each run splits the 32 samples into **4 chunks of 8** and wraps each
chunk in its own `debugger.start / stop / step`, so the dump contains
**4 msprobe steps** per run. Four steps with the same shape but different
data is the right granularity for catching shape-dependent kernel
divergences (more than one step's worth of evidence, not so many that
the dump becomes unwieldy).

## Determinism knobs

- `do_sample=False`, `temperature=1.0` (ignored) ⇒ argmax decode, no
  `multinomial` RNG inside `predict_action_batch`
- `set_determinism(seed=1234)` pins torch / NumPy / random / cudnn /
  CUBLAS_WORKSPACE_CONFIG / ACLNN_DETERMINISTIC
- `attn_implementation="sdpa"` (knob: `--attn-implementation`) so the
  no-patch run goes through `F.scaled_dot_product_attention`, which is
  what NPU's `torch_npu` PrivateUse1 SDPA registration also intercepts
- bf16/fp32 patches use `tests/parity/fp32_attn_loader.py` with
  `PARITY_ATTN_DTYPE=bf16` (or `fp32`) — replaces `LlamaSdpaAttention.forward`
  with a manual `matmul → softmax → matmul` decomposition at the chosen
  compute dtype

## Layout

```
dumps/
└── gpu/                                       # NVIDIA A100 80GB SXM4
    ├── no_patch/                              # PARITY_DISABLE_FP32_ATTN=1
    │   ├── step0/proc<PID>/{construct,dump,stack}.json   # samples 0-7
    │   ├── step1/proc<PID>/{construct,dump,stack}.json   # samples 8-15
    │   ├── step2/proc<PID>/{construct,dump,stack}.json   # samples 16-23
    │   └── step3/proc<PID>/{construct,dump,stack}.json   # samples 24-31
    └── bf16_patch/                            # PARITY_ATTN_DTYPE=bf16
        ├── step0/proc<PID>/{construct,dump,stack}.json
        ├── step1/proc<PID>/{construct,dump,stack}.json
        ├── step2/proc<PID>/{construct,dump,stack}.json
        └── step3/proc<PID>/{construct,dump,stack}.json
```

`construct.json` = module/op call tree; `dump.json` = per-API
input/output statistics (mean/max/min/L2/norm/…); `stack.json` = the
originating Python frames. Compare two runs with `msprobe compare` (one
pair per matching `stepN`).

The `proc<PID>` rank-equivalent directory is msprobe's per-process
namespace; for single-process scripts there's exactly one. (Worker-mode
dumps use `rank<N>` instead.)

## What's *not* here (yet)

- **NPU dumps.** Run the same script on the NPU host:

      python tests/parity/openvla_oft/run_rlinf_msprobe.py \
          --mode no_patch \
          --dump-path tests/parity/dumps/npu/no_patch
      python tests/parity/openvla_oft/run_rlinf_msprobe.py \
          --mode bf16_patch \
          --dump-path tests/parity/dumps/npu/bf16_patch

  Then `msprobe compare tests/parity/dumps/gpu/no_patch/step0/proc*/dump.json
  tests/parity/dumps/npu/no_patch/step0/proc*/dump.json` (per step).

- **`fp32_patch` dumps.** The runner supports `--mode fp32_patch`; we
  haven't shipped a default capture for it yet — add when needed.

## Reproducing

```bash
# from the repo root, using the openvlaoft_libero_venv venv
PYTHONPATH=. /workspace/RLinf/openvlaoft_libero_venv/bin/python \
    tests/parity/openvla_oft/run_rlinf_msprobe.py \
    --mode no_patch \
    --dump-path tests/parity/dumps/gpu/no_patch

PYTHONPATH=. /workspace/RLinf/openvlaoft_libero_venv/bin/python \
    tests/parity/openvla_oft/run_rlinf_msprobe.py \
    --mode bf16_patch \
    --dump-path tests/parity/dumps/gpu/bf16_patch
```
