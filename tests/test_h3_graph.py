from __future__ import annotations

import json
from pathlib import Path

from h3_graph import (
    DEFAULT_DISABLE_AUDIO,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_TURBO_LORA,
    DEFAULT_WIDTH,
    apply_i2v_job,
    apply_loras,
    duration_to_length,
    find_nodes,
    find_one,
    has_reference_pack,
    has_start_image,
    should_disable_audio,
    snap_dim,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = json.loads((ROOT / "workflows" / "h3_i2v_api.json").read_text(encoding="utf-8"))


def test_duration_grid():
    assert duration_to_length(5) == 124
    assert duration_to_length(10) == 243
    assert duration_to_length(15) == 362
    assert duration_to_length(5) % 17 == 5
    assert duration_to_length(10) % 17 == 5
    assert duration_to_length(15) % 17 == 5


def test_4090_optimized_defaults():
    assert (DEFAULT_WIDTH, DEFAULT_HEIGHT) == (704, 1248)
    assert DEFAULT_WIDTH * DEFAULT_HEIGHT < 768 * 1344
    assert DEFAULT_STEPS == 4
    assert DEFAULT_DISABLE_AUDIO is False
    assert "turbo_4step" in DEFAULT_TURBO_LORA


def test_native_audio_is_enabled_by_default():
    assert should_disable_audio({}) is False
    assert should_disable_audio({"disable_audio": False}) is False
    assert should_disable_audio({"disable_audio": True}) is True

    patched = apply_i2v_job(
        WORKFLOW,
        first_image_path="first.png",
        text_prompt="ambient sound",
        width=704,
        height=1248,
        length=124,
        seed=42,
        steps=4,
    )
    assert find_nodes(patched, "VAEDecodeAudio")
    assert find_nodes(patched, "VAELoader", title="Audio VAE")
    assert patched[find_one(patched, "CreateVideo")]["inputs"]["audio"]


def test_snap_dim_is_multiple_of_32():
    assert snap_dim(768) == 768
    assert snap_dim(770) == 768
    assert snap_dim(780) == 768
    assert snap_dim(790) == 800
    assert snap_dim(16) == 32


def test_workflow_is_native_h3_not_wan():
    class_types = {node["class_type"] for node in WORKFLOW.values()}
    assert "MiniMaxH3ImageToVideo" in class_types
    assert "SaveVideo" in class_types
    assert not any(name.startswith("Wan") for name in class_types)
    assert "WanVideoModelLoader" not in class_types
    noise = WORKFLOW[find_one(WORKFLOW, "RandomNoise")]["inputs"]
    assert "control_after_generate" not in noise
    save = WORKFLOW[find_one(WORKFLOW, "SaveVideo")]["inputs"]
    assert save["format"] == "auto"
    assert save["codec"] == "auto"


def test_find_nodes_by_class_and_title():
    assert find_one(WORKFLOW, "MiniMaxH3ImageToVideo") == "104"
    assert find_one(WORKFLOW, "LoadImage", title="First Frame") == "114"
    vaes = find_nodes(WORKFLOW, "VAELoader")
    assert len(vaes) == 2
    assert find_one(WORKFLOW, "VAELoader", title="Audio VAE") == "24"


def test_patch_end_frame_and_loras():
    patched = apply_i2v_job(
        WORKFLOW,
        first_image_path="/runpod-volume/inputs/start.png",
        end_image_path="/runpod-volume/inputs/end.png",
        text_prompt="walk forward",
        width=768,
        height=1152,
        length=124,
        seed=7,
        steps=8,
        loras=[{"name": "turbo.safetensors", "strength": 0.9}],
        disable_audio=True,
    )
    i2v = patched["104"]["inputs"]
    assert i2v["first_frame"] == ["114", 0]
    assert i2v["last_frame"] == ["last_image", 0]
    assert patched["last_image"]["inputs"]["image"] == "/runpod-volume/inputs/end.png"
    assert patched["lora_1"]["class_type"] == "LoraLoaderModelOnly"
    assert patched["lora_1"]["inputs"]["lora_name"] == "turbo.safetensors"
    assert patched["16"]["inputs"]["model"] == ["lora_1", 0]
    assert patched["9"]["inputs"]["model"] == ["lora_1", 0]
    assert "audio" not in patched["91"]["inputs"]
    assert not find_nodes(patched, "VAEDecodeAudio")
    assert not find_nodes(patched, "VAELoader", title="Audio VAE")


def test_loras_do_not_rewire_themselves():
    graph = json.loads(json.dumps(WORKFLOW))
    apply_loras(
        graph,
        [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 0.5},
        ],
    )
    assert graph["lora_1"]["inputs"]["model"] == ["6", 0]
    assert graph["lora_2"]["inputs"]["model"] == ["lora_1", 0]
    assert graph["16"]["inputs"]["model"] == ["lora_2", 0]


def test_mode_detection():
    assert has_start_image({"image_path": "/x.png"})
    assert not has_start_image({"prompt": "x"})
    assert has_reference_pack({"reference_images": [{}]})
    assert has_reference_pack({"mode": "r2v"})
    assert not has_reference_pack({"image_path": "/x.png"})
