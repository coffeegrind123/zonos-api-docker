# Zonos API

> ⚠️ **WARNING: UNSTABLE API - INITIAL RELEASE** ⚠️
> 
> This API is currently in its initial release phase (v1.0.0) and is considered unstable.
> Breaking changes may occur without notice. Use in production at your own risk.
> For development and testing purposes only.

A production-grade FastAPI implementation of the Zonos Text-to-Speech model.

## Credits

This API is built on top of the [Zonos-v0.1-hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) and [Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) models created by [Zyphra](https://huggingface.co/Zyphra). The models feature:

- Zero-shot TTS with voice cloning capabilities
- Support for multiple languages (100+ languages via eSpeak-ng)
- High-quality 44kHz audio output
- Fine-grained control over speaking rate, pitch, audio quality, and emotions
- Real-time performance (~2x real-time on RTX 4090)

For more information, visit the model cards on Hugging Face: [Hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) | [Transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer).

## Features

- FastAPI-based REST API for Zonos Text-to-Speech model
- Support for both Transformer and Hybrid model variants
- Docker and docker-compose support with NVIDIA GPU acceleration
- Production-ready with Gunicorn workers and optimizations
- Prometheus and Grafana monitoring integration
- Health checks and comprehensive logging
- CORS support and Swagger documentation
- Voice cloning and audio continuation support
- Fine-grained emotion and audio quality control

## Quick Start

### Using Pre-built Image

The fastest way to get started is using our pre-built Docker image:
```bash
docker pull ghcr.io/manascb1344/zonos-api-gpu:v1.0.0
docker run -d \
  --name zonos-api-gpu \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  zonos-api-gpu
```

### Manual Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/manascb1344/zonos-api
cd zonos-api
```

The API will be available at `http://localhost:8000`

## Running with Docker

1. Build the container:
```bash
docker build -t zonos-api .
```

2. Run the container:
```bash
docker run -d \
  --name zonos-api \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  zonos-api
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPU(s) to use (default: 0)
- `USE_GPU`: Enable/disable GPU usage (default: true)

## Requirements

- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support
- At least 8GB of GPU memory recommended

## Verifying the Installation

Check if the API is running:
```bash
curl http://localhost:8000/health
```

## API Endpoints

### GET /
Root endpoint that returns basic API information

### GET /models
Returns a list of available TTS models

### GET /languages
Returns a list of supported languages

### GET /model/{model_name}/conditioners
Returns available conditioners for a specific model

### POST /synthesize
Generate speech from text using the full API format. Example request:

```json
{
  "model_choice": "Zyphra/Zonos-v0.1-transformer",
  "text": "Hello, this is a test.",
  "language": "en-us",
  "emotion_values": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
  "vq_score": 0.78,
  "cfg_scale": 2.0,
  "min_p": 0.15
}
```

### POST /v1/audio/text-to-speech
**SillyTavern TTSSorcery Extension Compatible Endpoint**

Generate speech using the simplified format expected by the SillyTavern TTSSorcery extension. This endpoint automatically handles format conversion and model name mapping.

Example request:
```json
{
  "text": "Hello, this is a test.",
  "model": "zonos-v0.1-transformer",
  "speaking_rate": 15.0,
  "emotion": {
    "happiness": 0.8,
    "neutral": 0.2
  },
  "vqscore": 0.78,
  "speaker_noised": false,
  "speaker_audio": null
}
```

**Supported Models:**
- `zonos-v0.1-transformer` (maps to `Zyphra/Zonos-v0.1-transformer`)
- `zonos-v0.1-hybrid` (maps to `Zyphra/Zonos-v0.1-hybrid`)

**Features:**
- Automatic emotion mapping from named emotions to 8-element arrays
- Base64 speaker audio support for voice cloning
- No API key required when running locally
- CORS enabled for browser requests

## Environment Variables

- `USE_GPU`: Set to "true" to enable GPU acceleration (default: true)
- `PYTHONPATH`: Set to the application root directory

## SillyTavern Integration

This API is fully compatible with the [SillyTavern TTSSorcery Extension](https://github.com/coffeegrind123/SillyTavern-TTSSorcery-Fork). 

### Setup Instructions:
1. Install the fixed TTSSorcery extension that supports local APIs without API keys
2. Configure the extension settings:
   - ✅ Check "Use Local Zonos API"
   - Set "Local API URL" to: `http://localhost:8181`
   - Leave "Zyphra API Key" field empty (not required for local usage)
3. The extension will use the `/v1/audio/text-to-speech` endpoint automatically

### Docker Setup for SillyTavern:
```bash
# Using docker-compose (recommended)
docker-compose up --build

# Or using docker directly
docker run -d --gpus all -p 8181:8000 --name zonos-api zonos-local
```

## GPU Support

The API uses NVIDIA GPU acceleration by default. Make sure you have:
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. NVIDIA Container Toolkit installed and configured

## Development

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- Docker and docker-compose (for containerized deployment)

### Local Development
```bash
# Start in development mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or with docker-compose
docker-compose up --build
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 