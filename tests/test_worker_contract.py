from __future__ import annotations

import hashlib
import json

import pytest

from worker_contract import (
    ContractError,
    models_ready,
    resolve_model_base,
    upload_video,
    validate_output_target,
)


class Response:
    def raise_for_status(self):
        return None


def _target():
    return {
        "upload_url": "https://r2.example.test/signed?secret=hidden",
        "object_key": "projects/project/generations/version/output.mp4",
        "content_type": "video/mp4",
        "required_headers": {"Content-Type": "video/mp4"},
    }


def test_output_target_rejects_unsafe_or_wrong_types():
    assert validate_output_target(_target()).object_key.endswith(".mp4")
    invalid = _target()
    invalid["upload_url"] = "http://example.test/output"
    with pytest.raises(ContractError, match="HTTPS"):
        validate_output_target(invalid)


def test_upload_streams_video_and_returns_integrity(tmp_path):
    payload = b"fake-mp4-payload"
    video = tmp_path / "video.mp4"
    video.write_bytes(payload)
    captured = {}

    def put(url, *, data, headers, timeout):
        captured.update(url=url, body=data.read(), headers=headers, timeout=timeout)
        return Response()

    result = upload_video(str(video), _target(), request_put=put)
    assert captured["body"] == payload
    assert captured["headers"] == {"Content-Type": "video/mp4"}
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_models_ready_checks_required_files(monkeypatch, tmp_path):
    manifest = {
        "files": [
            {"target": "diffusion_models/model.safetensors", "min_bytes": 4},
            {"target": "vae/audio.safetensors", "min_bytes": 4},
        ]
    }
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model = tmp_path / "models" / "diffusion_models" / "model.safetensors"
    model.parent.mkdir(parents=True)
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("COMFY_MODEL_BASE", str(tmp_path / "models"))
    assert not models_ready()
    model.write_bytes(b"ready")
    assert not models_ready()
    audio = tmp_path / "models" / "vae" / "audio.safetensors"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"ready")
    assert models_ready()


def test_resolve_model_base_prefers_valid_cached_snapshot(monkeypatch, tmp_path):
    manifest = {"files": [{"target": "diffusion_models/model.safetensors", "min_bytes": 4}]}
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    repository = tmp_path / "hub" / "models--yt-gang--h3-cache"
    snapshot = repository / "snapshots" / "abc123"
    model = snapshot / "diffusion_models" / "model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"ready")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("COMFY_MODEL_BASE", str(tmp_path / "fallback"))
    monkeypatch.setenv("H3_CACHED_MODEL_REPO", "yt-gang/h3-cache")
    monkeypatch.setenv("H3_CACHE_ROOT", str(tmp_path / "hub"))
    assert resolve_model_base() == snapshot


def test_resolve_model_base_can_require_cache(monkeypatch, tmp_path):
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("H3_CACHED_MODEL_REPO", "yt-gang/missing")
    monkeypatch.setenv("H3_CACHE_ROOT", str(tmp_path / "hub"))
    monkeypatch.setenv("H3_REQUIRE_CACHED_MODEL", "true")
    with pytest.raises(ContractError, match="missing or incomplete"):
        resolve_model_base()
