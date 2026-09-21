# MiniMax H3 Turbo for RunPod Serverless

Production-oriented MiniMax Hailuo H3 **image-to-video / first-last-frame** worker for:

- NVIDIA GeForce RTX 4090 — 24 GB, Ada `sm_89`
- NVIDIA GeForce RTX 5090 — 32 GB, Blackwell `sm_120`

The repository can be imported directly into RunPod. The Docker build checks that
the installed PyTorch wheel contains kernels for both GPU architectures and CI builds
the real `linux/amd64` image on every change.

## Why this variant starts faster

- The ~42 GB H3 stack is kept on a RunPod Network Volume rather than baked into
  every container image.
- ComfyUI is pinned to `v0.36.0`; Manager, Jupyter and unrelated custom nodes are omitted.
- The default path uses the pruned INT8 ConvRot FL2VA model, NVFP4 text encoder,
  704×1248 vertical canvas, 5 seconds, no native audio and the official four-step
  Turbo LoRA.
- ComfyUI dynamically offloads models and reserves 1 GiB VRAM, which keeps the
  24 GB path usable. The 5090 keeps more of the pipeline resident and is the preferred SKU.
- Presigned HTTPS PUT output avoids returning a large base64 MP4 through RunPod.

This optimizes container pull/startup. The first inference after a cold start still
has to read the model from the Network Volume; keep `active workers = 1` when latency
is more important than scale-to-zero cost.

For RunPod's host-local Hugging Face cache path, private repository publishing,
territory requirements and a measured cold/hot verification procedure, see
[`docs/CACHED_MODEL.md`](docs/CACHED_MODEL.md).

## Import into RunPod

1. Import `https://github.com/yt-gang/h3-video-runpod` as a Serverless GitHub worker.
2. Attach a Network Volume provisioned by `scripts/provision-volume.sh`.
3. Allow `NVIDIA GeForce RTX 4090` and `NVIDIA GeForce RTX 5090`, one GPU, CUDA 12.8.
4. Use container disk 40 GB and execution timeout 1200–1800 seconds.
5. Enable FlashBoot after the first successful endpoint test.

The repository includes `.runpod/hub.json` and `.runpod/tests.json`; no Dockerfile
path override is needed.

### Provision the Network Volume

Run this once on a temporary GPU pod with the volume mounted at `/runpod-volume`:

```bash
git clone https://github.com/yt-gang/h3-video-runpod.git
cd h3-video-runpod
bash scripts/provision-volume.sh
```

Required files are validated before the worker accepts jobs. No model download occurs
during container startup.

## Request contract

The start image is required. Use exactly one of `image_url`, `image_path` or
`image_base64`. `end_image_*` is optional.

```json
{
  "input": {
    "prompt": "The subject turns toward camera, subtle handheld motion.",
    "image_url": "https://example.invalid/presigned-input",
    "width": 704,
    "height": 1248,
    "duration": 5,
    "steps": 4,
    "seed": 42,
    "disable_audio": true,
    "output": {
      "upload_url": "https://example.invalid/presigned-output",
      "object_key": "projects/project-id/generations/version-id/output.mp4",
      "content_type": "video/mp4",
      "required_headers": {
        "Content-Type": "video/mp4"
      }
    }
  }
}
```

With `output`, success returns object metadata and SHA-256. Without it, the legacy
response is `{ "video": "<base64 mp4>" }`.

Health probe:

```json
{"input":{"health_check":true}}
```

## Local checks

```bash
python -m pip install pytest requests
pytest -q
bash -n entrypoint.sh scripts/provision-volume.sh scripts/publish-cached-model.sh
docker build --platform linux/amd64 -t h3-video-runpod:test .
```

The Docker build is intentionally model-free. A successful build proves the runtime,
ComfyUI version and Ada/Blackwell CUDA wheel contract; an inference smoke test requires
the real Network Volume and GPU.

## Sources

This worker reuses proven pieces from:

- `bra-khet/podbooth-minimax-h3` — native H3 graph and lightweight volume layout.
- `vincezh2000/minimax-h3-comfyui-serverless` — CUDA architecture build guard and H3 runtime findings.
- `runpod-workers/worker-comfyui` and `yt-gang/qwen-img-2512` — RunPod contract and presigned object-storage pattern.
- Comfy-Org MiniMax H3 workflow and model packages.

MiniMax H3 weights use the MiniMax H3 Community License and have territory and
commercial-use restrictions. The endpoint operator is responsible for selecting an
eligible RunPod region and complying with that license.
