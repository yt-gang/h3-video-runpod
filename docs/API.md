# Job contract — PodBooth MiniMax H3 v1

Worker input is the RunPod `{"input": {…}}` body. Production jobs should upload
the MP4 with a presigned HTTPS PUT URL; base64 remains available for compatibility.

This image implements **I2V / FL2VA** only. A reference pack or `mode: "r2v"` returns an error pointing at the sibling Ref2VA worker (`podbooth-minimax-h3-ref2va`). The split keeps each image lightweight.

## I2V / first–last

```json
{
  "input": {
    "mode": "i2v",
    "prompt": "The subject from Picture 1 turns toward camera. Soft room tone.",
    "negative_prompt": "",
    "image_path": "/runpod-volume/inputs/start.png",
    "end_image_path": "/runpod-volume/inputs/end.png",
    "width": 704,
    "height": 1248,
    "duration": 5,
    "fps": 24,
    "steps": 4,
    "seed": 42,
    "cfg": null,
    "sampler": null,
    "disable_audio": false,
    "output": {
      "upload_url": "https://storage.example.invalid/presigned-put-url",
      "object_key": "projects/project-id/generations/version-id/output.mp4",
      "content_type": "video/mp4",
      "required_headers": {"Content-Type": "video/mp4"}
    }
  }
}
```

### Image slots

Use **one** of `_path` / `_url` / `_base64` per slot.

| Slot | Keys | Required |
|---|---|---|
| Start frame | `image_path`, `image_url`, `image_base64` | **yes** |
| End frame | `end_image_path`, `end_image_url`, `end_image_base64` | no |

- `_path` is a path the **worker** can read (`/runpod-volume/…`). Prefer this when the file is already on the volume.
- `_url` is downloaded on the worker.
- `_base64` is decoded to a temp file. Use this for local-machine uploads. Do not send huge stills this way if a volume path exists.
- There is **no** default `example_image.png`. Missing start image is an error.

End frame present → handler wires `MiniMaxH3ImageToVideo.last_frame`. Same graph, not a second Wan-style workflow file.

### Generation knobs

| Key | Default | Notes |
|---|---|---|
| `prompt` | required | Motion + audio. See `docs/PROMPTING.md`. |
| `negative_prompt` | `""` | H3 I2V has no native negative. Non-empty values are appended as `Avoid: …`. |
| `width` / `height` | `704` / `1248` | Snapped to a **multiple of 32**. This portrait default stays below 0.9 MP for the 24 GB path. |
| `duration` | `5` | Seconds. Converted to frame `length` on the 17k+5 grid at 24 fps: 5→124, 10→243, 15→362. Allowed operator set is `{5,10,15}`. |
| `fps` | `24` | Used only to convert duration. CreateVideo is baked at 24. |
| `steps` | `4` | Paired with the required official four-step Turbo LoRA. |
| `seed` | `42` | Integer. |
| `sampler` | `res_multistep` | Patched onto `KSamplerSelect` when set. |
| `cfg` | ignored | Official graph uses `BasicGuider` (no CFG). Unknown keys are ignored. |
| `disable_audio` | `false` | Native audio is enabled by default and requires the audio VAE. Set true to save VRAM and generation time. |
| `loras` | default four-step Turbo | Flat list `{name, strength}`. Not Wan high/low pairs. Files must exist under `/runpod-volume/loras/` or `/runpod-volume/models/loras/`. |

Optional overrides if the volume uses different filenames: `unet_name`, `clip_name`, `video_vae_name`, `audio_vae_name`.

### Timeouts

H3 15 s 768p on a 48 GB card can run several minutes. Client default wait is **2700 s**. Set the RunPod endpoint execution timeout to **1200–1800 s**. Handler does not kill the worker early.

## Output

Recommended success response after presigned upload:

```json
{
  "object_key": "projects/project-id/generations/version-id/output.mp4",
  "content_type": "video/mp4",
  "size_bytes": 1234567,
  "sha256": "..."
}
```

Legacy success response when `output` is omitted:

```json
{ "video": "<base64 mp4>" }
```

Failure:

```json
{ "error": "H3 v1 requires a start image (image_path | image_url | image_base64)." }
```

The handler reads Comfy history from `videos` **and** `gifs` (SaveVideo vs older VHS keys).

## R2V (sibling worker)

Lives in `bra-khet/podbooth-minimax-h3-ref2va`, not this image. Shape kept here so clients do not invent a new one:

```json
{
  "input": {
    "mode": "r2v",
    "prompt": "subject_definitions: …",
    "reference_images": [{"path": "/runpod-volume/inputs/face.png"}],
    "reference_videos": [],
    "reference_audios": [],
    "duration": 10,
    "fps": 24,
    "steps": 16,
    "seed": 42,
    "loras": []
  }
}
```

Caps on the sibling: 9 images / 3 videos / 3 audio / 12 mixed. This I2V worker returns an error instead of running FL2VA.
