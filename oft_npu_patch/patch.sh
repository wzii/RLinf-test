#!/bin/bash
# Temporary OpenVLA-OFT NPU attention patch -- install / uninstall helper.
#
# Standalone, NOT part of RLinf. Run this on the NPU host BEFORE launching the
# OpenVLA-OFT eval/train so the decomposed-attention patch is live in the head
# process and in every Ray worker (it installs a .pth into the active venv).
#
#   bash patch.sh                # install (default)
#   bash patch.sh --uninstall    # remove
#   bash patch.sh --status       # show state
#
# Use a specific interpreter with PYTHON=/path/to/venv/bin/python bash patch.sh.
# Set OFT_NPU_ATTN_DTYPE=fp32 for maximum GPU<->NPU agreement (default bf16).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$(command -v python3 || command -v python)}"

if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "[oft-npu-attn] ERROR: python interpreter not found: $PY" >&2
    echo "               set PYTHON=/path/to/venv/bin/python" >&2
    exit 2
fi

if [ "$#" -eq 0 ]; then
    set -- --install
fi

exec "$PY" "$DIR/patch.py" "$@"
