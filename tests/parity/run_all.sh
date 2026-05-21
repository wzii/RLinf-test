#!/usr/bin/env bash
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Generate the four parity goldens on the current machine.
#
# Usage:
#   bash tests/parity/run_all.sh                                   # auto-detect venv
#   OPENVLA_PY=.../bin/python WAN_PY=.../bin/python bash tests/parity/run_all.sh
#
# Run the same script unchanged on NPU, then transfer the goldens to one
# host and run:
#
#   for n in openvla_oft_rlinf openvla_oft_official wan_rlinf_env wan_upstream_pipeline; do
#       python tests/parity/compare.py \
#           tests/parity/goldens/${n}_gpu.pt tests/parity/goldens/${n}_npu.pt
#   done

set -eu -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

# Default to the venvs the repo ships if the user didn't pin one.
OPENVLA_PY="${OPENVLA_PY:-${REPO}/openvlaoft_libero_venv/bin/python}"
WAN_PY="${WAN_PY:-${REPO}/openvlaoft_wan_venv/bin/python}"

# DiffSynth / DreamZero / WanEnv need these paths on PYTHONPATH.
export EMBODIED_PATH="${REPO}/examples/embodiment"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "[parity] repo=${REPO}"
echo "[parity] openvla-oft python: ${OPENVLA_PY}"
echo "[parity] wan          python: ${WAN_PY}"

mkdir -p "${HERE}/goldens"

run() {
    local tag="$1"; shift
    local py="$1"; shift
    echo
    echo "=========================================================="
    echo "[parity] ${tag}"
    echo "=========================================================="
    "${py}" "$@"
}

run "openvla-oft (RLinf impl)"     "${OPENVLA_PY}" "${HERE}/openvla_oft/run_rlinf.py"     "$@"
run "openvla-oft (official impl)"  "${OPENVLA_PY}" "${HERE}/openvla_oft/run_official.py"  "$@"
run "wan (RLinf env wrapper)"      "${WAN_PY}"     "${HERE}/wan/run_rlinf_env.py"          "$@"
run "wan (upstream WanVideoPipeline)" "${WAN_PY}"  "${HERE}/wan/run_upstream_pipeline.py"   "$@"

echo
echo "[parity] All goldens written under ${HERE}/goldens/"
ls -lh "${HERE}/goldens/"
