from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import soundfile as sf
import io
from typing import Optional
import logging

from .config import settings
from .model import model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.MAX_TEXT_LENGTH)
    model_type: str = Field(default="Transformer", pattern="^(Transformer|Hybrid)$")
    language: str = Field(default="en-us", pattern=r"^[a-z]{2}-[A-Z]{2}$")
    speaker_audio: Optional[str] = None
    prefix_audio: Optional[str] = None
    skip_speaker: bool = False
    skip_emotion: bool = False
    emotion1: float = Field(0.6, ge=0.0, le=1.0, description="Happiness")
    emotion2: float = Field(0.05, ge=0.0, le=1.0, description="Sadness")
    emotion3: float = Field(0.05, ge=0.0, le=1.0, description="Disgust")
    emotion4: float = Field(0.05, ge=0.0, le=1.0, description="Fear")
    emotion5: float = Field(0.05, ge=0.0, le=1.0, description="Surprise")
    emotion6: float = Field(0.05, ge=0.0, le=1.0, description="Anger")
    emotion7: float = Field(0.5, ge=0.0, le=1.0, description="Other")
    emotion8: float = Field(0.6, ge=0.0, le=1.0, description="Neutral")
    skip_vqscore_8: bool = True
    vq_single: float = Field(0.78, ge=0.5, le=0.8)
    fmax: int = Field(22050, ge=0, le=24000)
    skip_fmax: bool = False
    pitch_std: float = Field(20.0, ge=0.0, le=400.0)
    skip_pitch_std: bool = False
    speaking_rate: float = Field(15.0, ge=0.0, le=40.0)
    skip_speaking_rate: bool = False
    dnsmos_ovrl: float = Field(4.0, ge=1.0, le=5.0)
    skip_dnsmos_ovrl: bool = True
    speaker_noised: bool = False
    skip_speaker_noised: bool = False
    cfg_scale: float = Field(2.0, ge=1.0, le=5.0)
    min_p: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = Field(420, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test of the Zonos text-to-speech system.",
                "model_type": "Transformer",
                "language": "en-us",
                "cfg_scale": 2.0,
                "min_p": 0.1,
                "seed": 420
            }
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        await model.load()
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize the application")

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Welcome to Zonos API",
        "version": settings.API_VERSION,
        "status": "active",
        "available_models": list(settings.MODEL_NAMES.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "current_model": model.current_model_type}

@app.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    """Generate speech from text."""
    try:
        logger.info(f"Processing TTS request for text: {request.text[:50]}... using {request.model_type} model")
        
        # Generate audio
        audio_data, sample_rate = await model.generate_speech(
            **request.dict()
        )
        
        # Convert to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{hash(request.text)[:8]}.wav"
            }
        )

    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    ) 