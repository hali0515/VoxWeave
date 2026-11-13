# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ghcr.io/astral-sh/uv:0.10.9 AS uv

FROM ${CUDA_IMAGE}

ARG TORCH_BACKEND=cu128

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VOXWEAVE_CACHE_ROOT=/cache/voxweave \
    XDG_CONFIG_HOME=/config \
    XDG_DATA_HOME=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /cache/voxweave /config /data /work \
    && chmod 0777 /cache/voxweave /config /data /work

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /tmp/voxweave-src
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md MANIFEST.in overrides.txt ./
COPY voxweave ./voxweave

RUN uv tool install \
        --python 3.12 \
        --no-cache \
        --torch-backend="${TORCH_BACKEND}" \
        --overrides overrides.txt \
        ".[cuda]" \
    && rm -rf /tmp/voxweave-src \
    && voxweave --version

VOLUME ["/cache/voxweave"]
WORKDIR /work

ENTRYPOINT ["voxweave"]
CMD ["--help"]
