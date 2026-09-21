#!/usr/bin/env bash
# Build and publish a minimal private Hugging Face repository for RunPod model caching.
# Run only in an H3-licensed territory, for example on an AP-JP-1 temporary Pod.
set -euo pipefail

SOURCE_MODEL_ROOT="${SOURCE_MODEL_ROOT:-/workspace/models}"
STAGING_ROOT="${STAGING_ROOT:-/workspace/minimax-h3-fl2va-runpod-cache}"
HF_CACHE_REPO="${HF_CACHE_REPO:?set HF_CACHE_REPO to your private org/name repository}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"

FILES=(
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
)

mkdir -p "${STAGING_ROOT}"
for relative in "${FILES[@]}"; do
  source_path="${SOURCE_MODEL_ROOT}/${relative}"
  target_path="${STAGING_ROOT}/${relative}"
  if [ ! -s "${source_path}" ]; then
    echo "missing required source file: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if ! ln -f "${source_path}" "${target_path}" 2>/dev/null; then
    cp --reflink=auto "${source_path}" "${target_path}"
  fi
done

curl -fsSL \
  https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE \
  -o "${STAGING_ROOT}/LICENSE"

printf '%s\n' \
  'MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.' \
  > "${STAGING_ROOT}/NOTICE"

cat > "${STAGING_ROOT}/README.md" <<'EOF'
---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
private: true
---

# MiniMax H3 FL2VA RunPod cache

Private, minimal RunPod cached-model package for the `yt-gang/h3-video-runpod`
worker. It contains only the pruned INT8 FL2VA diffusion model, NVFP4 text
encoder, FP16 video VAE and official four-step Turbo LoRA.

Use and distribution are restricted to the Applicable Territory defined by the
included MiniMax H3 Community License Agreement. Do not deploy this repository
in the European Union, United Kingdom, Republic of Korea or United States.
EOF

python3 -m pip install --no-cache-dir -U --break-system-packages \
  "huggingface_hub[hf_transfer]" hf_transfer >/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1

hf repo create "${HF_CACHE_REPO}" --repo-type model --private --exist-ok
hf upload-large-folder "${HF_CACHE_REPO}" "${STAGING_ROOT}" --repo-type model

echo "Published private cached-model repository: https://huggingface.co/${HF_CACHE_REPO}"
