FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies with cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir uv

# Copy dependency files first
COPY requirements.txt pyproject.toml ./

# Install Python dependencies with caching in smaller chunks
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# Clone Zonos repository with minimal depth
RUN git clone --depth 1 https://github.com/Zyphra/Zonos.git /app/zonos \
    && cd /app/zonos \
    && git submodule update --init --recursive

# Install Zonos and its dependencies in smaller chunks
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /app/zonos \
    && uv pip install --system -e . \
    && rm -rf /root/.cache/pip/*

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system \
    kanjize>=1.5.0 \
    inflect>=7.5.0 \
    && rm -rf /root/.cache/pip/*

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system \
    phonemizer>=3.3.0 \
    sudachidict-full>=20241021 \
    sudachipy>=0.6.10 \
    && rm -rf /root/.cache/pip/*

# Install GPU dependencies with caching in separate steps
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install --system flash-attn --no-build-isolation \
    && rm -rf /root/.cache/pip/*

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-build-isolation mamba-ssm==2.2.4 \
    && rm -rf /root/.cache/pip/*

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-build-isolation causal-conv1d==1.5.0.post8 \
    && rm -rf /root/.cache/pip/*

# Copy application code last
COPY app/ app/

# Environment variables
ENV PYTHONPATH=/app:/app/zonos \
    USE_GPU=true \
    PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
