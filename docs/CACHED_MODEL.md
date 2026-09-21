# RunPod Cached Model deployment

This path replaces repeated reads from a normal Network Volume with RunPod's
host-local Hugging Face model cache. Keep the repository private.

## Territory and redistribution

MiniMax H3 excludes the European Union, United Kingdom, Republic of Korea and
United States. Create, upload, cache and run this package only in an Applicable
Territory such as Japan. The publishing script includes the required LICENSE and
NOTICE files. It intentionally creates a private repository.

The existing `EU-RO-1` endpoint is not an eligible deployment target. Create a
new `AP-JP-1` endpoint or migrate the endpoint before testing H3 cached weights.

## 1. Create the private cache repository

Start a temporary Pod in `AP-JP-1` with at least 100 GB of writable disk. Download
the approved four-file stack, then publish it:

```bash
git clone https://github.com/yt-gang/h3-video-runpod.git
cd h3-video-runpod

VOLUME_ROOT=/workspace bash scripts/provision-volume.sh

export HF_TOKEN='YOUR_WRITE_TOKEN'
export HF_CACHE_REPO='YOUR_HF_NAMESPACE/minimax-h3-fl2va-runpod-cache'
SOURCE_MODEL_ROOT=/workspace/models bash scripts/publish-cached-model.sh
```

Do not paste `HF_TOKEN` into logs or commit it. Revoke the write token after the
upload if it was created only for this operation. Keep a separate read token for
RunPod.

The private repository must contain:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
LICENSE
NOTICE
README.md
```

## 2. Configure RunPod

Create or edit an H3 endpoint in `AP-JP-1`:

1. Deploy the current `yt-gang/h3-video-runpod` `main` build.
2. Set **Model** to the private Hugging Face repository URL.
3. Give RunPod a Hugging Face read token for that private repository.
4. Set `H3_CACHED_MODEL_REPO=YOUR_HF_NAMESPACE/minimax-h3-fl2va-runpod-cache`.
5. Set `H3_REQUIRE_CACHED_MODEL=true` so a missing cache fails closed instead of
   silently measuring the old Network Volume.
6. Keep FlashBoot enabled, one GPU per worker, RTX 5090 first and RTX 4090 second.
7. Use `Max workers=1` for the initial benchmark.

The worker resolves RunPod's cache at:

```text
/runpod-volume/huggingface-cache/hub/models--ORG--NAME/snapshots/REVISION/
```

It validates every required file against `models.json` before starting ComfyUI.

## 3. Verify deployment

Inspect worker logs. A cache-backed boot must include:

```text
h3-worker: using RunPod cached model snapshot
```

It must not say `using network-volume models`.

Submit the health input:

```json
{"input":{"health_check":true}}
```

Expected output includes `"ready": true`. Then submit one real cold I2V request,
wait for completion and record RunPod `delayTime`, `executionTime`, worker GPU and
the output SHA-256. Submit two more requests immediately and compare against the
network-volume baseline:

| Metric | Current baseline |
|---|---:|
| Cold delay | 42.425 s |
| Cold generation | 217.869 s |
| Hot generation | 92.135 s average |

The output must remain H.264, 704×1248, 24 fps, 124 frames and about 5.17 seconds.
Visually inspect multiple frames; a completed status alone is not sufficient.

## Rollback

Unset `H3_CACHED_MODEL_REPO` and `H3_REQUIRE_CACHED_MODEL`, remove the Model field,
and redeploy with an eligible-region Network Volume. The worker then falls back to
`COMFY_MODEL_BASE`, normally `/runpod-volume/models`.
