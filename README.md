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

### Using Docker Compose (Recommended)
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/manascb1344/zonos-api
cd zonos-api

# Or if you already cloned without --recursive:
git submodule update --init --recursive

# Start the services (API, Prometheus, Grafana)
docker-compose up -d

# The services will be available at:
# - API: http://localhost:8000
# - Swagger docs: http://localhost:8000/docs
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)

# To update submodules to latest version:
git submodule update --remote
docker-compose up -d --build
```

### Manual Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/manascb1344/zonos-api
cd zonos-api
```

2. Install system dependencies:
```bash
apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    espeak-ng \
    curl
```

3. Install Python dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
pip install --no-build-isolation -e .[compile]  # For GPU optimizations

# Install Zonos from submodule
cd Zonos
pip install -e .
cd ..
```

4. Run the application:
```bash
# Development
uvicorn app.main:app --reload

# Production
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Endpoints

### GET /
Root endpoint that returns API status and available models

### GET /health
Health check endpoint that returns current model status

### POST /tts
Text-to-speech conversion endpoint

Request body:
```json
{
    "text": "Text to convert to speech",
    "model_type": "Transformer",  // or "Hybrid"
    "language": "en-us",
    "emotion1": 0.6,  // Happiness
    "emotion2": 0.05, // Sadness
    "emotion3": 0.05, // Disgust
    "emotion4": 0.05, // Fear
    "emotion5": 0.05, // Surprise
    "emotion6": 0.05, // Anger
    "emotion7": 0.5,  // Other
    "emotion8": 0.6,  // Neutral
    "speaker_audio": null,  // Optional: Path to reference voice
    "prefix_audio": null,   // Optional: Path to continue from
    "cfg_scale": 2.0,
    "min_p": 0.1,
    "seed": 420
}
```

Response: Audio file (WAV format, 44.1kHz)

## Environment Variables

- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of Gunicorn workers (default: 4)
- `MODEL_TYPE`: Model variant to use (default: "Transformer")
- `MODEL_CACHE_DIR`: Directory for model caching
- Various CUDA optimization settings (see Dockerfile)

## Production Deployment

The application is containerized and optimized for production use. Features include:

- NVIDIA GPU support with CUDA optimizations
- Resource limits and monitoring
- Automatic model caching
- Health checks and automatic restarts
- Prometheus metrics and Grafana dashboards
- Proper logging with rotation
- Shared memory optimization
- Security considerations (non-root user, proper permissions)

## Development

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- Docker and docker-compose (for containerized deployment)

### Local Development
```bash
# Start in development mode
uvicorn app.main:app --reload

# Or with docker-compose
docker-compose up --build
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 
