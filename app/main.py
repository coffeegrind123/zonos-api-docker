from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
import wave
import numpy as np
from typing import List, Optional

from .models import TTSRequest
from .services.tts import TTSService
from .config import MAX_CHARACTERS

# Initialize FastAPI app
app = FastAPI(
    title="Zonos Text-to-Speech API",
    description="API for generating high-quality speech using Zonos models",
    version="0.1.0"
)

# TTS service instance
_tts_service: Optional[TTSService] = None

def get_tts_service() -> TTSService:
    """Get or initialize TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Zonos Text-to-Speech API"}

@app.get("/models")
async def get_models():
    """Get available TTS models."""
    service = get_tts_service()
    return {"models": service.get_model_names()}

@app.get("/languages")
async def get_languages():
    """Get supported languages."""
    service = get_tts_service()
    return {"languages": service.get_supported_languages()}

@app.get("/model/conditioners")
async def get_model_conditioners(model_name: str):
    """Get available conditioners for a specific model."""
    service = get_tts_service()
    print(f"Requested model: {model_name}")  # Debugging line
    print(f"Available models: {service.get_model_names()}")  # Debugging line
    if model_name not in service.get_model_names():
        raise HTTPException(status_code=404, detail="Model not found")
    return {"conditioners": service.get_model_conditioners(model_name)}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Generate speech from text."""
    if len(request.text) > MAX_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds maximum of {MAX_CHARACTERS} characters"
        )

    service = get_tts_service()
    
    try:
        # Generate audio using TTS service
        (sample_rate, audio_data), seed = service.generate_audio(
            model_choice=request.model_choice,
            text=request.text,
            language=request.language,
            speaker_audio=request.speaker_audio,
            prefix_audio=request.prefix_audio,
            emotion_values=request.emotion_values,
            vq_score=request.vq_score,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            speaker_noised=request.speaker_noised,
            cfg_scale=request.cfg_scale,
            min_p=request.min_p,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            unconditional_keys=request.unconditional_keys,
            top_p=request.top_p,
            top_k=request.top_k,
            linear=request.linear,
            confidence=request.confidence,
            quadratic=request.quadratic,
        )

        # Convert to WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Prepare response
        wav_buffer.seek(0)
        response = StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"x-seed": str(seed)}
        )
        
        return response

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Added verbose logging
        raise HTTPException(status_code=500, detail=str(e)) 