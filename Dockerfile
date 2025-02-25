FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set Zonos working directory
WORKDIR /app/zonos

# System packages
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

# Clone Zonos directly into working directory
RUN git clone --depth 1 https://github.com/Zyphra/Zonos.git . \
    && git submodule update --init --recursive

# Copy dependency specs and application code
COPY requirements.txt pyproject.toml ./
COPY app/ app/

# Install basic Python dependencies first
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r requirements.txt -e .[compile]

# Install Flash Attention with specific compiler flags
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV FLASH_ATTENTION_FORCE_BUILD=1
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-build-isolation \
    git+https://github.com/Dao-AILab/flash-attention.git@v2.5.6

# Install remaining ML dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-build-isolation \
    mamba-ssm==2.2.4 \
    causal-conv1d==1.5.0.post8

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

# Copy application code last
COPY app/ app/

# Environment variables
ENV PYTHONPATH=/app:/app/zonos \
    USE_GPU=true \
    PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
