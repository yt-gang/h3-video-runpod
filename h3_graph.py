"""MiniMax H3 workflow helpers.

Find nodes by class_type (and optional title), never by hard-coded numeric IDs.
Convert operator-facing duration/size into the values MiniMaxH3ImageToVideo wants.
"""

from __future__ import annotations

import copy
from typing import Any

H3_FPS = 24
H3_LENGTH_MOD = 17
H3_LENGTH_OFFSET = 5  # trained grid is 17k + 5
H3_SIZE_MULTIPLE = 32
H3_MIN_DIM = 32
ALLOWED_DURATIONS = (5, 10, 15)
DEFAULT_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 1248
DEFAULT_DURATION = 5
DEFAULT_STEPS = 4
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"


def find_nodes(prompt: dict[str, Any], class_type: str, title: str | None = None) -> list[str]:
    """Return node ids whose class_type matches, optionally filtered by _meta.title."""
    hits: list[str] = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != class_type:
            continue
        if title is not None:
            meta_title = (node.get("_meta") or {}).get("title")
            if meta_title != title:
                continue
        hits.append(str(node_id))
    return hits


def find_one(prompt: dict[str, Any], class_type: str, title: str | None = None) -> str:
    hits = find_nodes(prompt, class_type, title=title)
    if not hits:
        label = class_type if title is None else f"{class_type} ({title})"
        raise KeyError(f"No node with class_type {label!r} in workflow")
    return hits[0]


def duration_to_length(duration_seconds: float, fps: int = H3_FPS) -> int:
    """Convert seconds to H3 frame count: snap up onto the 17k+5 grid at 24 fps.

    Official template expression:
        max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    5s → 124, 10s → 243, 15s → 362.
    """
    duration = max(1.0, min(float(duration_seconds), 15.0))
    raw = max(5, int(round(duration * fps)))
    add = (H3_LENGTH_OFFSET - (raw % H3_LENGTH_MOD)) % H3_LENGTH_MOD
    return raw + add


def snap_dim(value: Any, multiple: int = H3_SIZE_MULTIPLE, minimum: int = H3_MIN_DIM) -> int:
    """H3 canvas is a multiple of 32 (not Wan's multiple of 16)."""
    numeric = float(value)
    snapped = int(round(numeric / multiple) * multiple)
    return max(minimum, snapped)


def _set_input(prompt: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    prompt[node_id].setdefault("inputs", {})[key] = value


def apply_i2v_job(
    prompt: dict[str, Any],
    *,
    first_image_path: str,
    end_image_path: str | None = None,
    text_prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    sampler: str | None = None,
    unet_name: str | None = None,
    clip_name: str | None = None,
    video_vae_name: str | None = None,
    audio_vae_name: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    disable_audio: bool = False,
) -> dict[str, Any]:
    """Return a patched copy of the I2V API graph for one job."""
    graph = copy.deepcopy(prompt)

    first_id = find_one(graph, "LoadImage", title="First Frame")
    _set_input(graph, first_id, "image", first_image_path)

    i2v_id = find_one(graph, "MiniMaxH3ImageToVideo")
    _set_input(graph, i2v_id, "prompt", text_prompt)
    _set_input(graph, i2v_id, "width", width)
    _set_input(graph, i2v_id, "height", height)
    _set_input(graph, i2v_id, "length", length)
    _set_input(graph, i2v_id, "first_frame", [first_id, 0])

    if end_image_path:
        last_id = "last_image"
        graph[last_id] = {
            "inputs": {"image": end_image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Last Frame"},
        }
        _set_input(graph, i2v_id, "last_frame", [last_id, 0])
    else:
        graph[i2v_id]["inputs"].pop("last_frame", None)

    noise_id = find_one(graph, "RandomNoise")
    _set_input(graph, noise_id, "noise_seed", int(seed))

    sched_id = find_one(graph, "BasicScheduler")
    _set_input(graph, sched_id, "steps", int(steps))

    if sampler:
        sampler_id = find_one(graph, "KSamplerSelect")
        _set_input(graph, sampler_id, "sampler_name", sampler)

    if unet_name:
        _set_input(graph, find_one(graph, "UNETLoader"), "unet_name", unet_name)
    if clip_name:
        _set_input(graph, find_one(graph, "CLIPLoader"), "clip_name", clip_name)

    if video_vae_name:
        _set_input(graph, find_one(graph, "VAELoader", title="Video VAE"), "vae_name", video_vae_name)
    if audio_vae_name:
        _set_input(graph, find_one(graph, "VAELoader", title="Audio VAE"), "vae_name", audio_vae_name)

    apply_loras(graph, loras or [])

    if disable_audio:
        create_id = find_one(graph, "CreateVideo")
        graph[create_id]["inputs"].pop("audio", None)
        audio_decode_ids = find_nodes(graph, "VAEDecodeAudio")
        audio_vae_id = find_one(graph, "VAELoader", title="Audio VAE")
        for node_id in audio_decode_ids:
            graph.pop(node_id, None)
        graph.pop(audio_vae_id, None)

    return graph


def apply_loras(prompt: dict[str, Any], loras: list[dict[str, Any]]) -> None:
    """Chain LoraLoaderModelOnly nodes after UNETLoader. Flat list, not Wan high/low pairs."""
    cleaned: list[tuple[str, float]] = []
    for item in loras:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("lora") or item.get("lora_name")
        if not name or str(name).lower() in {"none", "undefined"}:
            continue
        strength = float(item.get("strength", item.get("strength_model", 1.0)))
        cleaned.append((str(name), strength))
    if not cleaned:
        return

    unet_id = find_one(prompt, "UNETLoader")
    prev = unet_id
    injected: list[str] = []
    for index, (name, strength) in enumerate(cleaned, start=1):
        node_id = f"lora_{index}"
        prompt[node_id] = {
            "inputs": {
                "model": [prev, 0],
                "lora_name": name,
                "strength_model": strength,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"LoRA {index}"},
        }
        injected.append(node_id)
        prev = node_id

    last_lora = injected[-1]
    for node_id, node in prompt.items():
        if node_id in injected:
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) >= 1 and str(value[0]) == str(unet_id):
                inputs[key] = [last_lora, 0]


def has_reference_pack(job_input: dict[str, Any]) -> bool:
    if job_input.get("mode") in {"r2v", "ref2v", "ref2va"}:
        return True
    for key in ("references", "reference_images", "reference_videos", "reference_audios"):
        value = job_input.get(key)
        if value:
            return True
    return False


def has_start_image(job_input: dict[str, Any]) -> bool:
    return any(key in job_input for key in ("image_path", "image_url", "image_base64"))


def has_end_image(job_input: dict[str, Any]) -> bool:
    return any(key in job_input for key in ("end_image_path", "end_image_url", "end_image_base64"))
