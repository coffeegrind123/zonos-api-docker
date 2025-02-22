import torch
import torchaudio
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

class TTSService:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_names = ["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]
        self.models = {
            name: Zonos.from_pretrained(name, device=device) 
            for name in self.model_names
        }
        
        # Set models to eval mode
        for model in self.models.values():
            model.requires_grad_(False).eval()

    def get_model_names(self) -> List[str]:
        return self.model_names

    def get_supported_languages(self) -> List[str]:
        return supported_language_codes

    def get_model_conditioners(self, model_choice: str) -> List[str]:
        """Get list of conditioner names for a model"""
        model = self.models[model_choice]
        return [c.name for c in model.prefix_conditioner.conditioners]

    def generate_audio(
        self,
        model_choice: str,
        text: str,
        language: str,
        speaker_audio: Optional[str],
        prefix_audio: Optional[str],
        emotion_values: List[float],
        vq_score: float,
        fmax: float,
        pitch_std: float,
        speaking_rate: float,
        dnsmos_ovrl: float,
        speaker_noised: bool,
        cfg_scale: float,
        min_p: float,
        seed: int,
        randomize_seed: bool,
        unconditional_keys: List[str],
    ) -> Tuple[Tuple[int, np.ndarray], int]:
        """
        Generate audio using the specified model and parameters.
        Returns a tuple of ((sample_rate, audio_data), seed).
        """
        selected_model = self.models[model_choice]

        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)

        # Process speaker audio if provided
        speaker_embedding = None
        if speaker_audio is not None and "speaker" not in unconditional_keys:
            wav, sr = torchaudio.load(speaker_audio)
            speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
            speaker_embedding = speaker_embedding.to(self.device, dtype=torch.bfloat16)

        # Process prefix audio if provided
        audio_prefix_codes = None
        if prefix_audio is not None:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = torchaudio.functional.resample(
                wav_prefix, 
                sr_prefix, 
                selected_model.autoencoder.sampling_rate
            )
            wav_prefix = wav_prefix.to(self.device, dtype=torch.float32)
            with torch.autocast(self.device, dtype=torch.float32):
                audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

        # Prepare emotion tensor
        emotion_tensor = torch.tensor(emotion_values, device=self.device)

        # Prepare VQ score tensor
        vq_tensor = torch.tensor([vq_score] * 8, device=self.device).unsqueeze(0)

        # Create conditioning dictionary
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
            speaker_noised=bool(speaker_noised),
            device=self.device,
            unconditional_keys=unconditional_keys,
        )
        conditioning = selected_model.prepare_conditioning(cond_dict)

        # Generate audio
        max_new_tokens = 86 * 30  # ~30 seconds of audio
        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=float(cfg_scale),
            batch_size=1,
            sampling_params=dict(min_p=float(min_p)),
        )

        # Decode generated codes to waveform
        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]

        return (sr_out, wav_out.squeeze().numpy()), seed