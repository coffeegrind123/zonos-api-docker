from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import wave
import numpy as np
from typing import List, Optional
import logging
import sys

from .models import TTSRequest, SillyTavernTTSRequest
from .services.tts import TTSService
from .config import MAX_CHARACTERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("zonos-tts-api")

# Initialize FastAPI app
app = FastAPI(
    title="Zonos Text-to-Speech API",
    description="API for generating high-quality speech using Zonos models",
    version="0.1.0"
)

# Add CORS middleware to handle browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# TTS service instance
_tts_service: Optional[TTSService] = None

def get_tts_service() -> TTSService:
    """Get or initialize TTS service singleton."""
    global _tts_service
    if _tts_service is None:
        logger.info("Initializing TTS service")
        _tts_service = TTSService()
    return _tts_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting up Zonos TTS API - initializing services")
    # Initialize TTS service and load default model
    service = get_tts_service()
    logger.info("✅ TTS service initialized and default model loaded")

@app.get("/")
async def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    return {"message": "Zonos Text-to-Speech API"}

@app.get("/models")
async def get_models():
    """Get available TTS models."""
    logger.debug("Fetching available models")
    service = get_tts_service()
    models = service.get_model_names()
    logger.info(f"Retrieved {len(models)} models")
    return {"models": models}

@app.get("/languages")
async def get_languages():
    """Get supported languages."""
    logger.debug("Fetching supported languages")
    service = get_tts_service()
    languages = service.get_supported_languages()
    logger.info(f"Retrieved {len(languages)} supported languages")
    return {"languages": languages}

@app.get("/model/conditioners")
async def get_model_conditioners(model_name: str):
    """Get available conditioners for a specific model."""
    logger.debug(f"Fetching conditioners for model: {model_name}")
    service = get_tts_service()
    logger.info(f"Requested model: {model_name}")
    logger.debug(f"Available models: {service.get_model_names()}")
    if model_name not in service.get_model_names():
        logger.warning(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail="Model not found")
    conditioners = service.get_model_conditioners(model_name)
    logger.info(f"Retrieved {len(conditioners)} conditioners for model {model_name}")
    return {"conditioners": conditioners}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Generate speech from text."""
    logger.info(f"Speech synthesis requested for text of length {len(request.text)} chars using model {request.model_choice}")
    
    if len(request.text) > MAX_CHARACTERS:
        logger.warning(f"Text length ({len(request.text)}) exceeds maximum of {MAX_CHARACTERS} characters")
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds maximum of {MAX_CHARACTERS} characters"
        )

    service = get_tts_service()
    
    try:
        logger.debug(f"Starting audio generation with params: language={request.language}, "
                    f"speaking_rate={request.speaking_rate}, seed={request.seed}")
        
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

        logger.info(f"Successfully generated audio with sample rate {sample_rate}Hz, seed {seed}")

        # Convert to WAV format optimized for browser compatibility
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)  # Use original sample rate
            # Ensure proper scaling and clipping for 16-bit audio
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        # Prepare response with browser-friendly headers
        wav_buffer.seek(0)
        response = StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "x-seed": str(seed),
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes",
                "Content-Disposition": "inline; filename=\"tts_audio.wav\""
            }
        )
        
        logger.debug("Returning WAV audio stream response")
        return response

    except Exception as e:
        logger.error(f"Error during speech synthesis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/text-to-speech")
async def sillytavern_synthesize_speech(request: SillyTavernTTSRequest):
    """Generate speech from text using SillyTavern TTSSorcery extension format."""
    logger.info(f"SillyTavern TTS request for text of length {len(request.text)} chars using model {request.model}")
    
    if len(request.text) > MAX_CHARACTERS:
        logger.warning(f"Text length ({len(request.text)}) exceeds maximum of {MAX_CHARACTERS} characters")
        raise HTTPException(
            status_code=400,
            detail=f"Text length exceeds maximum of {MAX_CHARACTERS} characters"
        )

    service = get_tts_service()
    
    try:
        # Convert SillyTavern request to internal format
        internal_request = request.to_tts_request()
        
        logger.debug(f"Starting audio generation with SillyTavern params: model={request.model}, "
                    f"speaking_rate={request.speaking_rate}, emotions={request.emotion}")
        
        # Generate audio using TTS service
        (sample_rate, audio_data), seed = service.generate_audio(
            model_choice=internal_request.model_choice,
            text=internal_request.text,
            language=internal_request.language,
            speaker_audio=internal_request.speaker_audio,
            prefix_audio=internal_request.prefix_audio,
            emotion_values=internal_request.emotion_values,
            vq_score=internal_request.vq_score,
            fmax=internal_request.fmax,
            pitch_std=internal_request.pitch_std,
            speaking_rate=internal_request.speaking_rate,
            dnsmos_ovrl=internal_request.dnsmos_ovrl,
            speaker_noised=internal_request.speaker_noised,
            cfg_scale=internal_request.cfg_scale,
            min_p=internal_request.min_p,
            seed=internal_request.seed,
            randomize_seed=internal_request.randomize_seed,
            unconditional_keys=internal_request.unconditional_keys,
            top_p=internal_request.top_p,
            top_k=internal_request.top_k,
            linear=internal_request.linear,
            confidence=internal_request.confidence,
            quadratic=internal_request.quadratic,
        )

        logger.info(f"Successfully generated audio with sample rate {sample_rate}Hz, seed {seed}")

        # Convert to WAV format optimized for browser compatibility
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)  # Use original sample rate
            # Ensure proper scaling and clipping for 16-bit audio
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        # Clean up temporary speaker audio file if created
        if internal_request.speaker_audio and internal_request.speaker_audio != request.speaker_audio:
            try:
                import os
                os.unlink(internal_request.speaker_audio)
            except Exception:
                pass

        # Prepare response with browser-friendly headers
        wav_buffer.seek(0)
        response = StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "x-seed": str(seed),
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes",
                "Content-Disposition": "inline; filename=\"tts_audio.wav\""
            }
        )
        
        logger.debug("Returning WAV audio stream response for SillyTavern")
        return response

    except Exception as e:
        logger.error(f"Error during SillyTavern speech synthesis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 