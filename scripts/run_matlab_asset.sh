#!/usr/bin/env bash
set -euo pipefail

MATLAB_BIN="${MATLAB_BIN:-/home/yz/MATLAB/R2025b/bin/matlab}"
ASSET_NAME="${1:?usage: $0 <asset-name> [output-json]}"
OUT_JSON="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${OUT_JSON}" ]]; then
  OUT_JSON_ABS="$(python - <<'PY' "${REPO_ROOT}" "${OUT_JSON}"
import os, sys
repo_root = sys.argv[1]
path = sys.argv[2]
print(os.path.abspath(path if os.path.isabs(path) else os.path.join(repo_root, path)))
PY
)"
  MATLAB_CMD="addpath('${REPO_ROOT}/sim_matlab/scripts'); run_repro_asset('${ASSET_NAME}', '${OUT_JSON_ABS}');"
else
  MATLAB_CMD="addpath('${REPO_ROOT}/sim_matlab/scripts'); run_repro_asset('${ASSET_NAME}');"
fi

"${MATLAB_BIN}" -batch "${MATLAB_CMD}"

if [[ -n "${OUT_JSON:-}" ]]; then
  python - <<'PY' "${OUT_JSON_ABS}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") != "PASS":
    sys.exit(1)
PY
fi
