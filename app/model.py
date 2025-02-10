import torch
import torchaudio
from transformers import AutoTokenizer
from Zonos.zonos.model import Zonos
from Zonos.zonos.conditioning import make_cond_dict
import logging
from typing import Tuple, Optional
import numpy as np
from .config import settings

logger = logging.getLogger(__name__)

class ZonosModel:
    def __init__(self):
        self.device = settings.DEVICE
        self.model = None
        self.tokenizer = None
        self.sample_rate = settings.SAMPLE_RATE
        self.current_model_type = None
        
    async def load(self, model_type: str = None) -> None:
        """Load the model and tokenizer."""
        if model_type is None:
            model_type = settings.MODEL_TYPE
            
        if self.current_model_type == model_type and self.is_loaded():
            logger.info(f"{model_type} model is already loaded")
            return
            
        try:
            logger.info(f"Loading Zonos {model_type} model on {self.device}...")
            
            # Clear CUDA cache if switching models
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            model_name = settings.MODEL_NAMES[model_type]
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self.model = Zonos.from_pretrained(
                repo_id=model_name,
                device=self.device
            )
            
            # Move model to device and optimize
            self.model.to(self.device)
            if self.device == "cuda":
                self.model.bfloat16()
            self.model.eval()
            
            self.current_model_type = model_type
            logger.info(f"{model_type} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    async def generate_speech(
        self,
        text: str,
        model_type: str = None,
        language: str = "en-us",
        speaker_audio: Optional[str] = None,
        prefix_audio: Optional[str] = None,
        skip_speaker: bool = False,
        skip_emotion: bool = False,
        emotion1: float = 0.6,
        emotion2: float = 0.05,
        emotion3: float = 0.05,
        emotion4: float = 0.05,
        emotion5: float = 0.05,
        emotion6: float = 0.05,
        emotion7: float = 0.5,
        emotion8: float = 0.6,
        skip_vqscore_8: bool = True,
        vq_single: float = 0.78,
        fmax: int = 22050,
        skip_fmax: bool = False,
        pitch_std: float = 20.0,
        skip_pitch_std: bool = False,
        speaking_rate: float = 15.0,
        skip_speaking_rate: bool = False,
        dnsmos_ovrl: float = 4.0,
        skip_dnsmos_ovrl: bool = True,
        speaker_noised: bool = False,
        skip_speaker_noised: bool = False,
        cfg_scale: float = 2.0,
        min_p: float = 0.1,
        seed: int = 420,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text."""
        # Load or switch model if needed
        await self.load(model_type)
        
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # Handle speaker embedding
            speaker_embedding = None
            if speaker_audio and not skip_speaker:
                wav, sr = torchaudio.load(speaker_audio)
                speaker_embedding = self.model.make_speaker_embedding(wav, sr)
                speaker_embedding = speaker_embedding.to(self.device, dtype=torch.bfloat16)

            # Handle audio prefix
            audio_prefix_codes = None
            if prefix_audio:
                wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
                wav_prefix = wav_prefix.mean(0, keepdim=True)
                wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, 
                                                          self.model.autoencoder.sampling_rate)
                wav_prefix = wav_prefix.to(self.device, dtype=torch.float32)
                with torch.autocast(self.device, dtype=torch.float32):
                    audio_prefix_codes = self.model.autoencoder.encode(wav_prefix.unsqueeze(0))

            # Prepare conditioning
            uncond_keys = []
            if skip_speaker: uncond_keys.append("speaker")
            if skip_emotion: uncond_keys.append("emotion")
            if skip_vqscore_8: uncond_keys.append("vqscore_8")
            if skip_fmax: uncond_keys.append("fmax")
            if skip_pitch_std: uncond_keys.append("pitch_std")
            if skip_speaking_rate: uncond_keys.append("speaking_rate")
            if skip_dnsmos_ovrl: uncond_keys.append("dnsmos_ovrl")
            if skip_speaker_noised: uncond_keys.append("speaker_noised")

            emotion_tensor = torch.tensor(
                [[emotion1, emotion2, emotion3, emotion4, 
                  emotion5, emotion6, emotion7, emotion8]],
                device=self.device
            )

            vq_tensor = torch.tensor([vq_single] * 8, device=self.device).unsqueeze(0)

            cond_dict = make_cond_dict(
                text=text,
                language=language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=float(fmax),
                pitch_std=float(pitch_std),
                speaking_rate=float(speaking_rate),
                dnsmos_ovrl=float(dnsmos_ovrl),
                speaker_noised=speaker_noised,
                device=self.device,
                unconditional_keys=uncond_keys,
            )

            # Prepare generation parameters
            torch.manual_seed(seed)
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Generate audio
            codes = self.model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params=dict(min_p=min_p),
            )

            # Decode and return audio
            wav_out = self.model.autoencoder.decode(codes).cpu().detach()
            if wav_out.dim() == 2 and wav_out.size(0) > 1:
                wav_out = wav_out[0:1, :]
                
            return wav_out.squeeze().numpy(), self.model.autoencoder.sampling_rate

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise RuntimeError(f"Speech generation failed: {str(e)}")

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust the speed of the audio."""
        if speed == 1.0:
            return audio
            
        # This is a simple implementation - you might want to use a more sophisticated method
        return np.interp(
            np.arange(0, len(audio) / speed),
            np.arange(0, len(audio)),
            audio
        )

# Create a global instance
model = ZonosModel()