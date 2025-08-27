from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import base64
import tempfile
import os

class TTSRequest(BaseModel):
    model_choice: str = Field(description="Model variant to use")
    text: str = Field(description="Text to convert to speech")
    language: str = Field(description="Language code", default="en-us")
    speaker_audio: Optional[str] = Field(default=None, description="Path to speaker audio file")
    prefix_audio: Optional[str] = Field(default=None, description="Path to prefix audio file")
    emotion_values: List[float] = Field(
        default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
        description="List of 8 emotion values: happiness, sadness, disgust, fear, surprise, anger, other, neutral"
    )
    vq_score: float = Field(default=0.78, description="VQ Score value")
    fmax: float = Field(default=24000, description="Maximum frequency")
    pitch_std: float = Field(default=45.0, description="Pitch standard deviation")
    speaking_rate: float = Field(default=15.0, description="Speaking rate")
    dnsmos_ovrl: float = Field(default=4.0, description="DNSMOS overall score")
    speaker_noised: bool = Field(default=False, description="Whether to denoise speaker audio")
    cfg_scale: float = Field(default=2.0, description="CFG scale value")
    min_p: float = Field(default=0.15, description="Minimum probability")
    seed: int = Field(default=420, description="Random seed")
    randomize_seed: bool = Field(default=True, description="Whether to randomize seed")
    unconditional_keys: List[str] = Field(
        default=["emotion"],
        description="List of conditioning keys to make unconditional"
    )
    top_p: float = Field(default=0.95, description="Top-p sampling value")
    top_k: int = Field(default=50, description="Top-k sampling value") 
    linear: float = Field(default=1.0, description="Linear scaling factor")
    confidence: float = Field(default=0.1, description="Confidence scaling factor")
    quadratic: float = Field(default=1.0, description="Quadratic scaling factor")

class SillyTavernTTSRequest(BaseModel):
    """Request model matching SillyTavern TTSSorcery extension format"""
    text: str = Field(description="Text to convert to speech")
    speaking_rate: float = Field(default=15.0, description="Speaking rate")
    model: str = Field(default="zonos-v0.1-hybrid", description="Model name")
    speaker_audio: Optional[str] = Field(default=None, description="Base64 encoded speaker audio")
    emotion: Optional[Dict[str, float]] = Field(default=None, description="Emotion values")
    vqscore: Optional[float] = Field(default=0.78, description="VQ score for hybrid model")
    speaker_noised: Optional[bool] = Field(default=False, description="Whether speaker is noised")

    def to_tts_request(self) -> TTSRequest:
        """Convert SillyTavern request to internal TTSRequest format"""
        # Handle emotion mapping
        emotion_values = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]  # defaults
        if self.emotion:
            # Map SillyTavern emotions to Zonos format
            emotion_mapping = {
                'happiness': 0, 'sadness': 1, 'disgust': 2, 'fear': 3,
                'surprise': 4, 'anger': 5, 'other': 6, 'neutral': 7
            }
            for emotion_name, value in self.emotion.items():
                if emotion_name in emotion_mapping:
                    emotion_values[emotion_mapping[emotion_name]] = float(value)

        # Handle speaker audio - it comes as base64 from SillyTavern
        speaker_audio = None
        if self.speaker_audio:
            try:
                # Decode base64 and save to temporary file
                audio_data = base64.b64decode(self.speaker_audio)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                    f.write(audio_data)
                    speaker_audio = f.name
            except Exception:
                # If decoding fails, ignore speaker audio
                pass

        return TTSRequest(
            model_choice=self.model,
            text=self.text,
            speaking_rate=self.speaking_rate,
            speaker_audio=speaker_audio,
            emotion_values=emotion_values,
            vq_score=self.vqscore or 0.78,
            speaker_noised=self.speaker_noised or False
        )

class AudioResponse(BaseModel):
    sample_rate: int
    audio_data: bytes
    seed: int