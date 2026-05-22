# Cross-hardware (GPU ↔ NPU) parity tests

These tests answer the question:

> When I run the same model on the same inputs on GPU vs NPU, do I get the same outputs?

If the answer is "no, and the diff is large", that explains why a metric that
is healthy on GPU (e.g. libero+openvla-oft eval at 0.5) collapses to 0 on
NPU. If the diff is small but downstream metrics still collapse, the issue
is somewhere else (env, scheduler, weight sync, ...). Either way, the test
gives a clean signal.

## What is in here

```
tests/parity/
├── common.py                       # determinism setup + 32 input collectors + diff helpers
├── openvla_oft/
│   ├── run_rlinf.py                # OpenVLA-OFT via rlinf.models.embodiment.openvla_oft.rlinf
│   └── run_official.py             # OpenVLA-OFT via rlinf.models.embodiment.openvla_oft.official
├── wan/
│   ├── run_rlinf_env.py            # Wan-as-env through WanEnv.chunk_step
│   └── run_upstream_pipeline.py    # Wan through DiffSynth WanVideoPipeline directly
├── compare.py                      # diff two goldens, bit-exact + semantic metrics
└── run_all.sh                      # produces all four goldens on the current machine
```

Each script:
1. pins every randomness source (NumPy / Python / torch / cudnn / cuBLAS workspace),
2. constructs **32 deterministic inputs** sourced from
   `/workspace/RLinf-Wan-LIBERO-Spatial/dataset/` so the inputs are
   byte-identical no matter what hardware you are on,
3. runs the forward in a **deterministic decode mode**:
   - OpenVLA-OFT: `do_sample=False` → argmax (the production eval samples
     with temperature 1.6, which hides kernel drift behind RNG noise — bad
     for a parity test),
   - Wan: the diffusion noise is pinned to a **fixed artifact**
     (`goldens/wan_fixed_noise.pt`) loaded byte-for-byte on every machine.
     `rand_device="cpu"` alone is NOT enough: `torch.randn`'s Box-Muller
     transform uses transcendental / SIMD ops whose results differ by 1–2 ULP
     between x86 (GPU host) and aarch64 (NPU host), which would silently desync
     the inputs. See `common.install_fixed_noise`. Each Wan forward is also split
     into `WAN_MICRO_BATCH` (default 4) chunks so the NPU's peak memory stays low
     without dropping coverage (`common.install_microbatch`),
4. runs twice and checks **self-consistency** before saving,
5. dumps a golden file under `tests/parity/goldens/` containing every
   tensor of interest plus SHA256s.

## How to use

### Step 1 — generate goldens on the current (GPU) machine

```
bash tests/parity/run_all.sh
```

This produces:

```
tests/parity/goldens/
├── openvla_oft_rlinf_gpu.pt
├── openvla_oft_official_gpu.pt
├── wan_rlinf_env_gpu.pt
└── wan_upstream_pipeline_gpu.pt
```

### Step 2 — run the same scripts on NPU

Same command, on the NPU host. The harness picks the active device
automatically (`pick_device()` checks CUDA, then NPU, then CPU). You will end
up with `*_npu.pt` files alongside the GPU goldens.

### Step 3 — compare

```
for n in openvla_oft_rlinf openvla_oft_official wan_rlinf_env wan_upstream_pipeline; do
    echo "=== ${n} ==="
    python tests/parity/compare.py \
        tests/parity/goldens/${n}_gpu.pt \
        tests/parity/goldens/${n}_npu.pt \
        --report tests/parity/goldens/${n}_diff.json
done
```

For each pair the comparator prints:

- a per-tensor table with `bit-exact` / `allclose@1e-5` / `allclose@1e-2`
  flags, `max_abs`, `mean_abs`,
- semantic downstream metrics:
  - OpenVLA-OFT: chunk-action L1, cosine, action-token agreement rate,
  - Wan-env: reward L1, termination agreement, image diff,
  - Wan-upstream: video L1, max, PSNR.

## Reading the result

| What you see                                                | What it means                                                                                                                                                              |
| ---                                                         | ---                                                                                                                                                                        |
| Both wan goldens diff GPU↔NPU but openvla goldens match     | Investigate the Wan pipeline — DiT / VAE kernels under NPU. Start with `wan_upstream_pipeline` (rules out the env wrapper) and look at intermediate latents / sampler step. |
| Both openvla goldens diff but wan goldens match             | Investigate OpenVLA-OFT — start with `openvla_oft_rlinf` (the path your eval uses). Compare action-token agreement to localise drift (vision vs LLM).                       |
| `_rlinf_` differs but `_official_` matches (per model)      | The bug is in the RLinf adapter (`predict_action_batch`), not the upstream model — likely something added for RL like the value head or the multimodal embedding builder.   |
| Both differ                                                 | Likely a global kernel-level thing (bf16 matmul, attention impl). Try `attn_implementation: eager` (already set), `torch.use_deterministic_algorithms`, NPU op blacklist.   |
| Within-device self-consistency fails (first run vs second)  | Determinism flags didn't fully pin the math on this device. Investigate this FIRST — cross-device numbers are meaningless until the same device is self-consistent.        |

## Caveats

- The OpenVLA-OFT inputs do not run the libero env. They use frames pulled
  from the Wan checkpoint's `dataset/` directory because (a) those files are
  bundled with the model and identical across machines, and (b) we want the
  test to run without MuJoCo / libero / EGL. The image preprocessing inside
  the model (resize to 224, normalise) still runs on the chosen device, so
  this is still a real device-side forward.
- For Wan, the env wrapper builds the pipeline on `cuda:0` unconditionally
  in `WanEnv._build_pipeline`. On the NPU side this needs the NPU patch
  that should already exist in your adaptation (the same patch that lets
  training reach the `chunk_step` call). If `run_rlinf_env.py` cannot
  construct the env on NPU but `run_upstream_pipeline.py` can, that is
  itself a useful signal.
- The strict-tolerance check is `rtol=1e-5, atol=1e-5`. Cross-hardware bf16
  forwards will basically never meet this; the bf16-friendly check is
  `rtol=1e-2, atol=1e-2`. The exit code distinguishes the two so CI can
  enforce whichever it wants.
