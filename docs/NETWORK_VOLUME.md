# Network volume layout — MiniMax H3

Weights stay on the RunPod **network volume**, never in the Docker image.
v1 is pinned to **AP-JP-1** (MiniMax H3 Community License). Do not use an
EU / US / UK / KR volume. Serverless mounts at `/runpod-volume`. A GPU pod
in the same DC can mount the same volume at `/workspace` or `/runpod-volume`.

This volume is **100 GB**. The I2V sibling holds the INT8 FL2VA stack (~41 GB).
The Ref2VA sibling adds `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (~21 GB)
next to it. Do not download BF16. Do not replace the FL2VA file.

H3 LoRAs, if any, go in `/loras/` (same folder convention as the Wan booth,
different volume).

## Folder tree

```
/runpod-volume/
  models/
    diffusion_models/     # FL2VA (+ Ref2VA later). Not interchangeable.
    text_encoders/        # Qwen3-VL MiniMax pack
    vae/                  # video VAE + audio VAE
    loras/                # optional second home for H3 LoRAs
    embeddings/           # optional minimaxh3_* style embeddings
    audio_encoders/       # unused by the official I2V graph
  loras/                  # SAME folder the Wan booth already uses
  inputs/                 # stills / ref packs the client can address by path
  outputs/                # optional workspace; production output uses presigned PUT
```

## v1 I2V / FL2VA — download these five files

Use Comfy-Org **pruned INT8 ConvRot** with this repository's CUDA 12.8 / PyTorch
2.8 image. The Docker build verifies both Ada `sm_89` and Blackwell `sm_120`
kernels before publishing.

| Role | Exact filename | Size | Hugging Face |
|---|---|---|---|
| FL2VA diffusion | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GB | [text_encoders/…](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GB | [vae/…](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.58 GB | [vae/…](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |

**I2V stack total ≈ 39.6 GB.** Do not download this onto the Windows workstation.

On-volume destinations:

```
/runpod-volume/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
/runpod-volume/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
/runpod-volume/models/vae/minimax_h3_video_vae_fp16.safetensors
/runpod-volume/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

Example (run **on the pod / volume**, not locally):

```bash
hf download Comfy-Org/MiniMax-H3 \
  --include "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --include "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  --include "vae/minimax_h3_video_vae_fp16.safetensors" \
  --include "vae/minimax_h3_audio_vae_fp32.safetensors" \
  --local-dir /runpod-volume/models
```

If `hf download` writes into `diffusion_models/` under that local-dir, you already have the right layout.

## Turbo LoRA (volume-only)

The official four-step Turbo LoRA is required by `models.json`, downloaded by the
provisioning script and enabled by default. Override `H3_TURBO_LORA` or pass an
explicit `loras` list only when intentionally selecting another sampler profile.

| File | Typical steps |
|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 4 |

Drop into `/runpod-volume/loras/` (Wan convention) or `/runpod-volume/models/loras/`.

## Ref2VA (sibling worker)

FL2VA and Ref2VA are **different checkpoints**. Do not point this I2V graph at a `ref2va_*` file.

The sibling image `podbooth-minimax-h3-ref2va` downloads onto this same volume:

- `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (21 GB INT8)

Projected used with both DiTs: ~62 GB of 100 GB. Turbo LoRAs are family-specific
(I2V `fl2v` vs Ref2V `ref2v`). Do not mix them.

## 24 GB vs 32 GB

| GPU | What v1 will advertise |
|---|---|
| 24 GB (4090) | Supported default: 704×1248, 5 s, four-step Turbo, audio off. Tight; model offload is expected. |
| 32 GB (5090) | Preferred SKU: same default with more of the pipeline resident. |

Longer clips and native audio materially increase runtime and VRAM pressure. Do not
advertise 24 GB as a Ref2VA or 15-second 768p target.
