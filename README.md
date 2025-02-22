# Zonos API

A FastAPI-based REST API for the Zonos text-to-speech model. This API provides endpoints for generating high-quality speech from text using state-of-the-art machine learning models.

## Features

- Text-to-speech generation using Zonos models
- Support for multiple languages
- Voice cloning capabilities
- Emotion control
- Various audio quality parameters
- GPU acceleration support

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zonos-api.git
cd zonos-api
```

2. Start the API using Docker Compose:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

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
Generate speech from text. Example request:

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

## Environment Variables

- `USE_GPU`: Set to "true" to enable GPU acceleration (default: true)
- `PYTHONPATH`: Set to the application root directory

## GPU Support

The API uses NVIDIA GPU acceleration by default. Make sure you have:
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. NVIDIA Container Toolkit installed and configured

## Development

To run the API in development mode:

```bash
docker-compose up --build
```

The API will reload automatically when code changes are detected.

## API Documentation

Once the API is running, you can access:
- Swagger UI documentation at `http://localhost:8000/docs`
- ReDoc documentation at `http://localhost:8000/redoc`

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 
