# docker build -t registry.webis.de/code-research/authorship/generative-ai-detection .
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying actual source files to make image updates faster.
# Install flash-attn separately, as it cannot be installed with build isolation and thus Poetry right now.
COPY pyproject.toml poetry.lock /opt/generative-ai-detection/
WORKDIR /opt/generative-ai-detection

RUN --mount=type=cache,target=/root/.cache set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install poetry packaging setuptools \
    && python3 -m poetry config virtualenvs.create false \
    && python3 -m poetry install --no-root \
    && MAX_JOBS=$(nproc) python3 -m pip install --no-build-isolation flash-attn

COPY . /opt/generative-ai-detection/

RUN --mount=type=cache,target=/root/.cache set -x && \
    python3 -m poetry install

