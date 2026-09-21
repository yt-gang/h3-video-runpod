#!/bin/bash
# Download the v1 I2V stack onto the Japan network volume.
# Run this ON a pod in AP-JP-1 with the volume mounted at /runpod-volume.
# Never run this on the laptop. Never wget these files into the Docker image.
set -euo pipefail

ROOT="${VOLUME_ROOT:-/runpod-volume}"
mkdir -p \
  "${ROOT}/models/diffusion_models" \
  "${ROOT}/models/text_encoders" \
  "${ROOT}/models/vae" \
  "${ROOT}/models/loras" \
  "${ROOT}/models/embeddings" \
  "${ROOT}/loras" \
  "${ROOT}/inputs" \
  "${ROOT}/outputs"

python3 -m pip install --no-cache-dir -U --break-system-packages "huggingface_hub[hf_transfer]" hf_transfer >/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Downloading Comfy-Org MiniMax-H3 I2V files into ${ROOT}/models ..."
hf download Comfy-Org/MiniMax-H3 \
  --include "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --include "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  --include "vae/minimax_h3_video_vae_fp16.safetensors" \
  --include "vae/minimax_h3_audio_vae_fp32.safetensors" \
  --include "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" \
  --local-dir "${ROOT}/models"

# Public Comfy template still (not a character LoRA) so I2V jobs have a path input.
if [ ! -f "${ROOT}/inputs/start.png" ]; then
  echo "Fetching public I2V test still into ${ROOT}/inputs/start.png"
  wget -q -O "${ROOT}/inputs/start.png" \
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/input/transparent_rgb_gaming_mouse.png"
fi

echo "Volume layout:"
find "${ROOT}/models" "${ROOT}/inputs" -type f | sort
du -sh "${ROOT}/models"/* 2>/dev/null || true
echo "DONE"
