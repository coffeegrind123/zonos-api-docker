FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY requirements.txt pyproject.toml ./

# Install system dependencies and Python dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv \
    && uv pip install --system -r requirements.txt

# Clone the zonos repository and update submodules
RUN git clone https://github.com/Zyphra/Zonos.git /app/zonos \
    && cd /app/zonos \
    && git submodule update --init --recursive \
    && cd ..

# Install Zonos and its dependencies
RUN uv pip install --system -e /app/zonos \
    && uv pip install --system kanjize>=1.5.0 \
    inflect>=7.5.0 \
    phonemizer>=3.3.0 \
    sudachidict-full>=20241021 \
    sudachipy>=0.6.10

# Install GPU dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation \
    && pip install \
       https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
       https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy application code last (changes most frequently)
COPY app/ app/
COPY pyproject.toml .

# Environment variables
ENV PYTHONPATH=/app:/app/zonos
ENV USE_GPU=true

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
