#!/usr/bin/env bash
# Example: run SayCan trainswap from LeRobot repo root on WSL.
#   cp experiments/so101_trainswap/env/wsl.env.example experiments/so101_trainswap/env/wsl.env
#   edit wsl.env, then:
#   source experiments/so101_trainswap/env/wsl.env   # or: set -a && source ... && set +a
#   bash experiments/so101_trainswap/configs/saycan_wsl.example.sh

set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [[ -f experiments/so101_trainswap/env/wsl.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source experiments/so101_trainswap/env/wsl.env
  set +a
fi

: "${PI_BASE_URL:?Set PI_BASE_URL (see env/wsl.env.example)}"
: "${REALSENSE_SERIAL:?Set REALSENSE_SERIAL}"

uv run lerobot-saycan-trainswap \
  --pi_base_url="${PI_BASE_URL}" \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --calibration_path=experiments/so101_trainswap/artifacts/so101_tag_calibration.json \
  --realsense_serial="${REALSENSE_SERIAL}" \
  --llm.base_url="${LM_BASE_URL:-http://127.0.0.1:1234/v1}" \
  --llm.model="${LM_MODEL:-local-model}" \
  --clip.device="${CLIP_DEVICE:-cpu}"
