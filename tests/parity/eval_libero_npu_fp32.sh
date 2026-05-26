#!/bin/bash
# Run LIBERO eval on NPU with fp32 attention monkey-patched into transformers.
#
# Strategy
# --------
# Writes a single-line .pth file into the eval venv's site-packages.  The .pth
# triggers fp32_attn_loader.py at Python startup -- that includes the head
# process AND every Ray worker, since they share the same venv site-packages.
# The patch replaces LlamaSdpaAttention.forward (transformers ≤4.45) or
# sdpa_attention_forward (≥4.46) with a manual fp32 implementation, which
# eliminates ~80 % of GPU↔NPU divergence as measured by the parity test
# (token agreement 20 % -> 97.6 %).
#
# Patch/unpatch pattern: the .pth is installed before launch and removed on
# exit (even on crash or Ctrl-C, via shell trap).  No RLinf source files are
# modified -- only one file added to the venv, transiently.
#
# Usage
# -----
#   bash tests/parity/eval_libero_npu_fp32.sh [CONFIG_NAME] [-- eval_args...]
#
# CONFIG_NAME defaults to "libero_spatial_openvlaoft_eval".  Any args after
# `--` are forwarded to eval_embodied_agent.py (e.g. Hydra overrides).
#
# Environment variables
# ---------------------
#   PARITY_VENV       Path to the eval venv (auto-detected by following the
#                     `python` symlink resolved by `which python`, or set
#                     explicitly).
#   PARITY_PYTHON     Explicit python interpreter to use (overrides PARITY_VENV).
#   PARITY_DISABLE_FP32_ATTN=1   Skip the patch (sanity check / A-B baseline).
#   PARITY_FP32_ATTN_VERBOSE=1   Log every worker's patch state.
#   ROBOT_PLATFORM, LIBERO_TYPE, MUJOCO_GL, ...   Same as eval_embodiment.sh.
#
# Exit code = the eval entry's exit code.

set -euo pipefail

# ── locate the parity dir and repo ──────────────────────────────────────────
PARITY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PARITY_DIR/../.." && pwd)"
LOADER_PATH="$PARITY_DIR/fp32_attn_loader.py"

if [ ! -f "$LOADER_PATH" ]; then
    echo "[parity] ERROR: $LOADER_PATH not found" >&2
    exit 2
fi

# ── resolve the eval python + its site-packages ─────────────────────────────
if [ -n "${PARITY_PYTHON:-}" ]; then
    EVAL_PY="$PARITY_PYTHON"
elif [ -n "${PARITY_VENV:-}" ]; then
    EVAL_PY="$PARITY_VENV/bin/python"
elif [ -x "$REPO/openvlaoft_libero_venv/bin/python" ]; then
    EVAL_PY="$REPO/openvlaoft_libero_venv/bin/python"
else
    EVAL_PY="$(command -v python3 || command -v python)"
fi

if [ ! -x "$EVAL_PY" ]; then
    echo "[parity] ERROR: python interpreter not executable: $EVAL_PY" >&2
    echo "         set PARITY_PYTHON or PARITY_VENV explicitly" >&2
    exit 2
fi

# Resolve the site-packages dir of THAT python -- works for any venv layout.
SITE_PACKAGES="$("$EVAL_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH_FILE="$SITE_PACKAGES/_rlinf_parity_fp32_attn.pth"

# ── install the .pth (one-line: prepend parity to sys.path, import loader) ──
install_pth() {
    cat > "$PTH_FILE" <<EOF
import sys; sys.path.insert(0, '$PARITY_DIR'); import fp32_attn_loader
EOF
    echo "[parity] installed fp32 attention hook: $PTH_FILE"
}

remove_pth() {
    if [ -f "$PTH_FILE" ]; then
        rm -f "$PTH_FILE"
        echo "[parity] removed fp32 attention hook: $PTH_FILE"
    fi
}

trap 'rc=$?; remove_pth; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

install_pth

# ── eval-time env vars (NPU-equivalents of the GPU eval shell) ──────────────
export EMBODIED_PATH="$REPO/examples/embodiment"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# OMP / MUJOCO / NCCL workarounds documented in tests/parity/HANDOFF.md.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"  # legacy alias

# LIBERO-specific
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"

# ── parse args ──────────────────────────────────────────────────────────────
CONFIG_NAME="${1:-libero_spatial_openvlaoft_eval}"
shift || true
EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA_ARGS=("$@")
fi

# Where the eval script writes its log.
LOG_DIR="${LOG_DIR:-$REPO/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}-fp32attn}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval.log"
echo "[parity] config       = $CONFIG_NAME"
echo "[parity] interpreter  = $EVAL_PY"
echo "[parity] site-packages= $SITE_PACKAGES"
echo "[parity] log dir      = $LOG_DIR"
echo "[parity] launching eval_embodied_agent.py ..."

# ── sanity probe: confirm the patch loads in the head python before launch ──
"$EVAL_PY" - <<'PY' 2>&1 | sed 's/^/[parity][probe] /'
import os
# The .pth fires at site init; if we got here, the loader already ran.
# Verify the marker is present on the target.
try:
    from transformers.models.llama import modeling_llama as m
    targets = []
    if hasattr(m, "LlamaSdpaAttention"):
        targets.append(("LlamaSdpaAttention.forward",
                        getattr(m.LlamaSdpaAttention.forward, "__rlinf_parity_fp32_attn_installed__", False)))
    for n in ("sdpa_attention_forward", "eager_attention_forward"):
        if hasattr(m, n):
            targets.append((n, getattr(getattr(m, n), "__rlinf_parity_fp32_attn_installed__", False)))
    for name, marked in targets:
        print(f"{name}: patched={marked}")
    if not any(m_ for _, m_ in targets):
        print("WARNING: no patch marker detected — fp32 attention is NOT active")
except Exception as e:
    print(f"probe failed: {e}")
PY

# ── launch the eval -- the .pth fires in every Ray worker automatically ─────
"$EVAL_PY" "$REPO/examples/embodiment/eval_embodied_agent.py" \
    --config-path "$REPO/examples/embodiment/config/" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$LOG_DIR" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
