FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone the zonos repository and update submodules
RUN git clone https://github.com/Zyphra/Zonos.git /app/zonos \
    && cd /app/zonos \
    && git submodule update --init --recursive \
    && cd ..

# Install Zonos and its dependencies
RUN pip3 install /app/zonos \
    && pip3 install kanjize>=1.5.0 \
    inflect>=7.5.0 \
    phonemizer>=3.3.0 \
    sudachidict-full>=20241021 \
    sudachipy>=0.6.10

# Install specific wheel files with GPU support
RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install flash-attn --no-build-isolation \
    && pip3 install --no-cache-dir https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && pip3 install --no-cache-dir https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy application code
COPY app/ app/
COPY pyproject.toml .

# Environment variables
ENV PYTHONPATH=/app:/app/zonos
ENV USE_GPU=true

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
