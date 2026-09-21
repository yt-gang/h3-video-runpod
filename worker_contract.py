"""Catline-compatible upload and readiness contract for the H3 worker."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests

OUTPUT_CONTENT_TYPE = "video/mp4"


class ContractError(ValueError):
    """A validation error safe to return without leaking signed URL queries."""


@dataclass(frozen=True)
class OutputTarget:
    upload_url: str
    object_key: str
    required_headers: dict[str, str]


def validate_output_target(raw: object) -> OutputTarget:
    if not isinstance(raw, dict):
        raise ContractError("output must be an object")
    upload_url = raw.get("upload_url")
    object_key = raw.get("object_key")
    content_type = raw.get("content_type")
    headers = raw.get("required_headers") or {}
    if not isinstance(upload_url, str) or urlsplit(upload_url).scheme != "https":
        raise ContractError("output.upload_url must be HTTPS")
    if not isinstance(object_key, str) or not object_key.startswith("projects/") or not object_key.endswith(".mp4"):
        raise ContractError("output.object_key must be a versioned projects/.../*.mp4 key")
    if content_type != OUTPUT_CONTENT_TYPE:
        raise ContractError("output.content_type must be video/mp4")
    if not isinstance(headers, dict) or headers.get("Content-Type") != OUTPUT_CONTENT_TYPE:
        raise ContractError("output.required_headers must sign Content-Type: video/mp4")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()):
        raise ContractError("output.required_headers must contain strings")
    return OutputTarget(upload_url, object_key, dict(headers))


def upload_video(path: str, raw_target: object, *, request_put=requests.put) -> dict[str, object]:
    target = validate_output_target(raw_target)
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        source.seek(0)
        try:
            response = request_put(
                target.upload_url,
                data=source,
                headers=target.required_headers,
                timeout=(15, 300),
            )
            response.raise_for_status()
        except requests.RequestException:
            raise RuntimeError("R2 output upload failed") from None
    return {
        "object_key": target.object_key,
        "content_type": OUTPUT_CONTENT_TYPE,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def models_ready() -> bool:
    if os.environ.get("H3_BUILD_TEST", "").lower() == "true":
        return True
    manifest_path = Path(os.environ.get("MODEL_MANIFEST_PATH", "/opt/h3/models.json"))
    model_base = Path(os.environ.get("COMFY_MODEL_BASE", "/runpod-volume/models"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if item.get("optional"):
                continue
            candidate = model_base / item["target"]
            if not candidate.is_file() or candidate.stat().st_size < int(item.get("min_bytes", 1)):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
