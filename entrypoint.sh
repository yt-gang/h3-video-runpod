#!/usr/bin/env bash
set -euo pipefail

PYTHON="${COMFY_PYTHON:-python3}"
COMFY_DIR="${COMFY_DIR:-/ComfyUI}"
MODEL_ROOT="${COMFY_MODEL_BASE:-/runpod-volume/models}"
WAIT_SECONDS="${COMFY_WAIT_SECONDS:-300}"

echo "h3-worker: build=${BUILD_VERSION:-unknown}"

if [ "${H3_BUILD_TEST:-false}" != "true" ]; then
  if [ ! -d /runpod-volume ]; then
    echo "h3-worker: /runpod-volume is required" >&2
    exit 1
  fi
  export COMFY_MODEL_BASE="${MODEL_ROOT}"
  if ! "${PYTHON}" -c 'from worker_contract import models_ready; raise SystemExit(0 if models_ready() else 1)'; then
    echo "h3-worker: required model files are missing or incomplete; run scripts/provision-volume.sh" >&2
    exit 1
  fi
else
  echo "h3-worker: build-test mode, skipping model validation"
fi

# ComfyUI performs dynamic model offloading. Reserving 1 GiB makes the 24 GiB
# 4090 path more stable; operators can override all flags through COMFY_EXTRA_ARGS.
COMFY_ARGS="${COMFY_EXTRA_ARGS:---disable-auto-launch --disable-metadata --reserve-vram 1}"
echo "h3-worker: starting ComfyUI ${COMFY_ARGS}"
# shellcheck disable=SC2086
"${PYTHON}" "${COMFY_DIR}/main.py" --listen 127.0.0.1 ${COMFY_ARGS} &
COMFY_PID=$!

for ((elapsed=0; elapsed<WAIT_SECONDS; elapsed+=2)); do
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "h3-worker: ComfyUI exited during startup" >&2
    wait "${COMFY_PID}"
  fi
  if curl -fsS http://127.0.0.1:8188/ >/dev/null; then
    echo "h3-worker: ComfyUI ready after ${elapsed}s"
    exec "${PYTHON}" /opt/h3/handler.py
  fi
  sleep 2
done

echo "h3-worker: ComfyUI did not become ready in ${WAIT_SECONDS}s" >&2
exit 1
