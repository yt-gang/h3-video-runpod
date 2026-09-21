# MiniMax H3 FL2VA Turbo — RunPod Serverless worker for RTX 4090 / RTX 5090.
# Weights stay on a Network Volume, keeping the image small enough for fast cold pulls.

ARG BASE_IMAGE=runpod/pytorch:1.2.0-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

ARG COMFYUI_VERSION=v0.36.0
ARG BUILD_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFY_DIR=/ComfyUI \
    COMFY_PYTHON=python3 \
    COMFY_MODEL_BASE=/runpod-volume/models \
    H3_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
    H3_CACHED_MODEL_REPO= \
    H3_REQUIRE_CACHED_MODEL=false \
    MODEL_MANIFEST_PATH=/opt/h3/models.json \
    H3_I2V_WORKFLOW=/workflows/h3_i2v_api.json \
    H3_TURBO_LORA=minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
    BUILD_VERSION=${BUILD_VERSION}

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# H3 nodes are part of ComfyUI core. No Manager/Jupyter/custom-node bootstrap is
# installed: fewer layers, less attack surface and less startup work.
RUN git clone --depth 1 --branch "${COMFYUI_VERSION}" \
        https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && python3 -m pip install --no-cache-dir -r /ComfyUI/requirements.txt

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fail at image-build time if the base torch wheel cannot execute on Ada
# (RTX 4090, sm_89) or Blackwell (RTX 5090, sm_120).
COPY check_arch.py /tmp/check_arch.py
RUN python3 /tmp/check_arch.py sm_89 sm_120 && rm /tmp/check_arch.py

COPY handler.py h3_graph.py worker_contract.py /opt/h3/
COPY models.json /opt/h3/models.json
COPY workflows /workflows
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p /ComfyUI/input /ComfyUI/output /ComfyUI/temp

ENV PYTHONPATH=/opt/h3
CMD ["/entrypoint.sh"]
