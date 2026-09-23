"""RunPod Serverless handler for MiniMax H3 FL2VA (I2V + optional last frame).

Keeps the Wan booth I/O spine (path/url/base64 → Comfy websocket → presigned PUT
or legacy base64 MP4)
and rewrites workflow patching to look up nodes by class_type.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

import runpod
import websocket

from h3_graph import (
    DEFAULT_DURATION,
    DEFAULT_HEIGHT,
    DEFAULT_SAMPLER,
    DEFAULT_STEPS,
    DEFAULT_TURBO_LORA,
    DEFAULT_WIDTH,
    apply_i2v_job,
    duration_to_length,
    find_nodes,
    find_one,
    has_end_image,
    has_reference_pack,
    has_start_image,
    should_disable_audio,
    snap_dim,
)
from worker_contract import ContractError, models_ready, upload_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())
WORKFLOW_I2V = os.getenv("H3_I2V_WORKFLOW", "/workflows/h3_i2v_api.json")
# BUG FIX: LoadImage combo 400
# Fix: LoadImage.image is a combo of files in Comfy's input folder. An absolute
# /runpod-volume path is rejected by POST /prompt with HTTP 400.
COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", "/ComfyUI/input")
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_CANVAS_PIXELS = 768 * 1344


def video_has_audio(path: str) -> bool:
    """Return whether ffprobe finds at least one audio stream."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def process_input(input_data: str, temp_dir: str, output_filename: str, input_type: str) -> str:
    """Resolve path / url / base64 to a LoadImage basename inside Comfy's input folder."""
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    dest_name = f"{uuid.uuid4().hex}_{os.path.basename(output_filename)}"
    dest_path = os.path.join(COMFY_INPUT_DIR, dest_name)
    if input_type == "path":
        logger.info("Path input: %s", input_data)
        if not os.path.isfile(input_data):
            raise Exception(f"Input file not found: {input_data}")
        shutil.copy2(input_data, dest_path)
        logger.info("Staged %s -> %s", input_data, dest_path)
        return dest_name
    if input_type == "url":
        logger.info("Staging HTTPS image input")
        download_file_from_url(input_data, dest_path)
        return dest_name
    if input_type == "base64":
        logger.info("Base64 input")
        save_base64_to_file(input_data, COMFY_INPUT_DIR, dest_name)
        return dest_name
    raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url: str, output_path: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise Exception("image URL must use HTTPS")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "catline-h3-worker/1"})
        with urllib.request.urlopen(request, timeout=60) as response, open(output_path, "wb") as target:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_INPUT_BYTES:
                raise Exception("image URL exceeds 32 MiB")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_INPUT_BYTES:
                    raise Exception("image URL exceeds 32 MiB")
                target.write(chunk)
        logger.info("Downloaded image from %s%s", parsed.netloc, parsed.path)
        return output_path
    except Exception as exc:
        raise Exception(f"URL download failed: {exc}") from exc


def save_base64_to_file(base64_data: str, temp_dir: str, output_filename: str) -> str:
    payload = base64_data
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        decoded = base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        raise Exception(f"Base64 decoding failed: {exc}") from exc
    if len(decoded) > MAX_INPUT_BYTES:
        raise Exception("base64 image exceeds 32 MiB")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
    with open(file_path, "wb") as handle:
        handle.write(decoded)
    logger.info("Saved Base64 input to %s", file_path)
    return file_path


def resolve_slot(job_input: dict[str, Any], task_id: str, prefix: str, filename: str) -> str | None:
    path_key = f"{prefix}_path"
    url_key = f"{prefix}_url"
    b64_key = f"{prefix}_base64"
    if path_key in job_input:
        return process_input(job_input[path_key], task_id, filename, "path")
    if url_key in job_input:
        return process_input(job_input[url_key], task_id, filename, "url")
    if b64_key in job_input:
        return process_input(job_input[b64_key], task_id, filename, "base64")
    return None


def queue_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{server_address}:8188/prompt"
    logger.info("Queueing prompt to %s", url)
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(url, data=payload)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as exc:
        # BUG FIX: Comfy /prompt 400 was opaque
        # Fix: include the response body so combo/schema errors are diagnosable
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Comfy /prompt HTTP %s: %s", exc.code, body)
        raise Exception(f"Comfy /prompt HTTP {exc.code}: {body}") from exc


def get_history(prompt_id: str) -> dict[str, Any]:
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def _collect_media_paths(node_output: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for bucket in ("videos", "gifs", "images", "audio"):
        items = node_output.get(bucket) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fullpath = item.get("fullpath")
            if fullpath and os.path.isfile(fullpath):
                paths.append(fullpath)
                continue
            filename = item.get("filename")
            subfolder = item.get("subfolder") or ""
            folder_type = item.get("type") or "output"
            if not filename:
                continue
            candidate = os.path.join("/ComfyUI", folder_type, subfolder, filename)
            if os.path.isfile(candidate):
                paths.append(candidate)
    return paths


def wait_for_videos(ws: websocket.WebSocket, prompt: dict[str, Any]) -> list[str]:
    queued = queue_prompt(prompt)
    prompt_id = queued["prompt_id"]
    while True:
        raw = ws.recv()
        if not isinstance(raw, str):
            continue
        message = json.loads(raw)
        if message.get("type") != "executing":
            continue
        data = message.get("data") or {}
        if data.get("node") is None and data.get("prompt_id") == prompt_id:
            break

    history = get_history(prompt_id)[prompt_id]
    found: list[str] = []
    for node_output in (history.get("outputs") or {}).values():
        found.extend(_collect_media_paths(node_output))

    videos = [path for path in found if path.lower().endswith(VIDEO_EXTS)]
    return videos or found


def load_workflow(workflow_path: str) -> dict[str, Any]:
    with open(workflow_path, encoding="utf-8") as handle:
        return json.load(handle)


def _compose_prompt(job_input: dict[str, Any]) -> str:
    text = (job_input.get("prompt") or "").strip()
    if not text:
        raise Exception("prompt is required")
    negative = (job_input.get("negative_prompt") or "").strip()
    # MiniMaxH3ImageToVideo has no native negative slot. Fold it in if the operator sent one.
    if negative:
        text = f"{text}\n\nAvoid: {negative}"
    return text


def _wait_http(timeout_s: int = 180) -> None:
    http_url = f"http://{server_address}:8188/"
    for attempt in range(timeout_s):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("HTTP connection successful (attempt %s)", attempt + 1)
            return
        except Exception as exc:
            logger.warning("HTTP connection failed (%s/%s): %s", attempt + 1, timeout_s, exc)
            time.sleep(1)
    raise Exception("Cannot connect to ComfyUI server. Please check if the server is running.")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        return {"error": "input must be an object"}
    logger.info("Received job keys: %s", sorted(job_input.keys()))
    task_id = f"task_{uuid.uuid4()}"
    staged_inputs: list[str] = []
    generated_paths: list[str] = []

    try:
        if job_input.get("health_check"):
            audio_vae = os.path.join(
                os.environ.get("COMFY_MODEL_BASE", "/runpod-volume/models"),
                "vae",
                "minimax_h3_audio_vae_fp32.safetensors",
            )
            return {
                "ready": models_ready(),
                "model": "minimax-h3-fl2va-turbo",
                "build_version": os.environ.get("BUILD_VERSION", "unknown"),
                "native_audio_default": True,
                "audio_vae_ready": os.path.isfile(audio_vae),
                "gpu_targets": ["RTX 4090 (sm_89)", "RTX 5090 (sm_120)"],
            }
        if has_reference_pack(job_input):
            return {
                "error": "H3 v1 is I2V/FL2VA only. Ref2VA (reference_images / mode=r2v) is not wired yet."
            }
        if not has_start_image(job_input):
            return {"error": "H3 v1 requires a start image (image_path | image_url | image_base64)."}

        image_path = resolve_slot(job_input, task_id, "image", "input_image.png")
        end_image_path = resolve_slot(job_input, task_id, "end_image", "end_image.png") if has_end_image(job_input) else None
        staged_inputs.extend(path for path in (image_path, end_image_path) if path)

        duration = job_input.get("duration", DEFAULT_DURATION)
        fps = int(job_input.get("fps", 24))
        if "length" in job_input and "duration" not in job_input:
            length = int(job_input["length"])
        else:
            length = duration_to_length(duration, fps=fps)

        width = snap_dim(job_input.get("width", DEFAULT_WIDTH))
        height = snap_dim(job_input.get("height", DEFAULT_HEIGHT))
        if width * height > MAX_CANVAS_PIXELS:
            raise ContractError("width * height exceeds the 768x1344 H3 limit")
        steps = int(job_input.get("steps", DEFAULT_STEPS))
        if not 1 <= steps <= 40:
            raise ContractError("steps must be between 1 and 40")
        seed = int(job_input.get("seed", 42))
        sampler = job_input.get("sampler") or DEFAULT_SAMPLER
        loras = job_input.get("loras")
        if loras is None:
            turbo_lora = os.environ.get("H3_TURBO_LORA", DEFAULT_TURBO_LORA).strip()
            loras = [{"name": turbo_lora, "strength": 1.0}] if turbo_lora else []
        if not isinstance(loras, list):
            raise ContractError("loras must be an array")
        disable_audio = should_disable_audio(job_input)
        text_prompt = _compose_prompt(job_input)

        template = load_workflow(WORKFLOW_I2V)
        prompt = apply_i2v_job(
            template,
            first_image_path=image_path,
            end_image_path=end_image_path,
            text_prompt=text_prompt,
            width=width,
            height=height,
            length=length,
            seed=seed,
            steps=steps,
            sampler=sampler,
            unet_name=job_input.get("unet_name"),
            clip_name=job_input.get("clip_name"),
            video_vae_name=job_input.get("video_vae_name"),
            audio_vae_name=job_input.get("audio_vae_name"),
            loras=loras,
            disable_audio=disable_audio,
        )
        logger.info(
            "Patched FL2VA graph: %sx%s length=%s steps=%s seed=%s end_frame=%s loras=%s "
            "disable_audio=%s audio_decode=%s audio_output=%s",
            width,
            height,
            length,
            steps,
            seed,
            bool(end_image_path),
            len(loras),
            disable_audio,
            bool(find_nodes(prompt, "VAEDecodeAudio")),
            "audio" in prompt[find_one(prompt, "CreateVideo")].get("inputs", {}),
        )

        _wait_http()
        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        max_attempts = 36
        for attempt in range(max_attempts):
            try:
                ws.connect(ws_url)
                logger.info("WebSocket connection successful (attempt %s)", attempt + 1)
                break
            except Exception as exc:
                logger.warning("WebSocket connection failed (%s/%s): %s", attempt + 1, max_attempts, exc)
                if attempt == max_attempts - 1:
                    raise Exception("WebSocket connection timeout (3 minutes)") from exc
                time.sleep(5)

        started_at = time.monotonic()
        videos = wait_for_videos(ws, prompt)
        ws.close()
        if not videos:
            return {"error": "No video could be found."}
        generated_paths.extend(videos)

        has_audio = video_has_audio(videos[0])
        logger.info("Generated media validation: audio_stream=%s", has_audio)
        if not disable_audio and not has_audio:
            raise RuntimeError("Generated video has no audio stream while native audio is enabled")

        output_target = job_input.get("output")
        if output_target is not None:
            result = upload_video(videos[0], output_target)
            result["generation_seconds"] = round(time.monotonic() - started_at, 3)
            return result

        with open(videos[0], "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return {
            "video": encoded,
            "content_type": "video/mp4",
            "generation_seconds": round(time.monotonic() - started_at, 3),
        }
    except Exception as exc:
        logger.exception("Handler failed")
        return {"error": str(exc)}
    finally:
        for name in staged_inputs:
            try:
                os.remove(os.path.join(COMFY_INPUT_DIR, name))
            except OSError:
                pass
        for path in generated_paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
