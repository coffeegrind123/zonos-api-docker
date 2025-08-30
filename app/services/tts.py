import torch
import torchaudio
import logging
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

# Configure logging first (commit 5fa603a)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_deepspeed_available():
    """Check if DeepSpeed is available for acceleration"""
    try:
        import deepspeed
        return True
    except ImportError:
        return False

def normalize_language_code(language: str) -> str:
    """Normalize language codes from 'EN_en' to 'en-en' format"""
    return language.lower().replace("_", "-")

def get_optimal_device(prefer: str = "fastest") -> str:
    """
    Select optimal CUDA device based on preference.
    
    Args:
        prefer: 'fastest' for best performance, 'memory' for most VRAM
    
    Returns:
        Device string (e.g., 'cuda:0' or 'cpu')
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return "cpu"
    
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return "cuda:0"
    
    logger.info(f"Found {device_count} CUDA devices")
    
    devices_info = []
    for device_idx in range(device_count):
        props = torch.cuda.get_device_properties(device_idx)
        
        # Skip devices with compute capability < 7.0
        if props.major < 7:
            logger.info(f"Device {device_idx}: {props.name} has CC {props.major}.{props.minor} < 7.0, skipping")
            continue
        
        devices_info.append({
            "idx": device_idx,
            "name": props.name,
            "compute_capability": (props.major, props.minor),
            "multi_processor_count": props.multi_processor_count,
            "total_memory": props.total_memory,
        })
        
        mem_gb = props.total_memory / (1024 ** 3)
        logger.info(f"Device {device_idx}: {props.name}, CC {props.major}.{props.minor}, "
                   f"MPs: {props.multi_processor_count}, Memory: {mem_gb:.1f}GB")
    
    if not devices_info:
        logger.warning("No suitable CUDA devices found, falling back to CPU")
        return "cpu"
    
    if prefer == "memory":
        selected = max(devices_info, key=lambda d: d["total_memory"])
    elif prefer == "fastest":
        selected = max(devices_info, key=lambda d: (
            d["compute_capability"], 
            d["multi_processor_count"], 
            d["total_memory"]
        ))
    else:
        raise ValueError(f"Unknown preference '{prefer}', use 'fastest' or 'memory'")
    
    logger.info(f"Selected device {selected['idx']}: {selected['name']}")
    return f"cuda:{selected['idx']}"

class TTSService:
    def __init__(self, device: str = None, enable_deepspeed: bool = None):
        logger.info("🚀 Initializing Zonos TTS Service with performance optimizations")
        
        # Use optimal device selection if not explicitly specified
        if device is None:
            gpu_preference = os.getenv("GPU_PREFERENCE", "fastest")
            logger.info(f"⚙️ GPU preference: {gpu_preference}")
            device = get_optimal_device(gpu_preference)
        
        logger.info(f"🎯 Selected device: {device}")
        self.device = device
        
        # Available models with their compatibility status
        self.available_models = {
            "Zyphra/Zonos-v0.1-transformer": {"status": "available", "supports_deepspeed": True},
            "Zyphra/Zonos-v0.1-hybrid": {"status": "backbone_incompatible", "supports_deepspeed": False}
        }
        
        # Single model slot (unload/reload on demand)
        self.current_model = None
        self.current_model_name = None
        
        # Default model to load on startup
        self.default_model = "Zyphra/Zonos-v0.1-transformer"
        
        # Configure offline mode for better performance in containers (commit 8dbc896)
        self.offline_mode = os.getenv("OFFLINE_MODE", "false").lower() in ("true", "1", "yes")
        if self.offline_mode:
            logger.info("🔒 Offline mode enabled - will not download models from HuggingFace")
        else:
            logger.info("🌐 Online mode enabled - can download models from HuggingFace if needed")
        
        # Handle DeepSpeed configuration
        if enable_deepspeed is None:
            enable_deepspeed = os.getenv("ENABLE_DEEPSPEED", "false").lower() in ("true", "1", "yes")
        
        self.enable_deepspeed = enable_deepspeed
        
        # Log DeepSpeed availability and status
        if is_deepspeed_available():
            logger.info("🚀 DeepSpeed acceleration available (optional)")
            if self.enable_deepspeed:
                logger.info("🚀 DeepSpeed acceleration enabled")
            else:
                logger.info("ℹ️ DeepSpeed available but not enabled - using standard PyTorch optimizations")
        else:
            logger.info("ℹ️ DeepSpeed not available - using standard PyTorch optimizations")
            if self.enable_deepspeed:
                logger.warning("⚠️ DeepSpeed requested but not available - falling back to standard mode")
                self.enable_deepspeed = False
        
        # Model name mapping for compatibility with different naming conventions
        self.model_aliases = {
            "zonos-v0.1-transformer": "Zyphra/Zonos-v0.1-transformer",
            "zonos-v0.1-hybrid": "Zyphra/Zonos-v0.1-hybrid",
        }
        
        # Load default model on startup for faster first request
        logger.info(f"📦 Loading default model on startup: {self.default_model}")
        try:
            self.current_model = self._load_single_model(self.default_model)
            self.current_model_name = self.default_model
            logger.info(f"🎉 Successfully loaded default model: {self.default_model}")
        except Exception as e:
            logger.error(f"💥 Failed to load default model {self.default_model}: {e}")
            logger.warning("⚠️ Service starting without pre-loaded model - will load on first request")

    def _load_single_model(self, model_name: str):
        """Load a single model with all optimizations"""
        logger.info(f"📥 Loading model: {model_name}")
        
        # Check if model is available
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not in available models: {list(self.available_models.keys())}")
        
        model_info = self.available_models[model_name]
        if model_info["status"] != "available":
            raise ValueError(f"Model {model_name} is not available: {model_info['status']}")
        
        # Load model with standard PyTorch
        load_kwargs = {"device": self.device, "backbone": "torch"}
        logger.info(f"🔧 Loading {model_name} with PyTorch")
        
        model = Zonos.from_pretrained(model_name, **load_kwargs)
        
        # Convert model to consistent dtype (before any DeepSpeed wrapper)
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(dtype=target_dtype)
        
        # Apply DeepSpeed wrapper if available and enabled and supported
        if (self.enable_deepspeed and 
            is_deepspeed_available() and 
            model_info.get("supports_deepspeed", False)):
            try:
                logger.info(f"🚀 Applying DeepSpeed optimization to {model_name}")
                model = self._wrap_with_deepspeed(model, model_name)
                logger.info(f"✅ Successfully loaded {model_name} with DeepSpeed optimization")
            except Exception as e:
                logger.warning(f"⚠️ DeepSpeed optimization failed for {model_name}: {e}")
                logger.info(f"✅ Falling back to standard PyTorch (float16) for {model_name}")
        else:
            logger.info(f"✅ Successfully loaded {model_name} with standard PyTorch (float16)")
        
        # Set to eval mode for inference
        model.requires_grad_(False).eval()
        logger.debug(f"🔒 Set {model_name} to evaluation mode")
        
        return model

    def _ensure_model_loaded(self, model_choice: str):
        """Ensure the requested model is loaded, unloading others if necessary"""
        # Normalize model name
        actual_model_name = self.model_aliases.get(model_choice, model_choice)
        
        # If already loaded, return current model
        if self.current_model_name == actual_model_name and self.current_model is not None:
            logger.debug(f"✅ Model {actual_model_name} already loaded")
            return self.current_model
        
        # Unload current model if different model requested
        if self.current_model is not None and self.current_model_name != actual_model_name:
            logger.info(f"🔄 Switching from {self.current_model_name} to {actual_model_name}")
            logger.info(f"📤 Unloading {self.current_model_name}")
            del self.current_model
            torch.cuda.empty_cache()  # Free GPU memory
        
        # Load the requested model
        logger.info(f"📥 Loading requested model: {actual_model_name}")
        self.current_model = self._load_single_model(actual_model_name)
        self.current_model_name = actual_model_name
        
        return self.current_model

    def _wrap_with_deepspeed(self, model, model_name: str):
        """Apply DeepSpeed optimization wrapper to model"""
        import deepspeed
        
        # Simple DeepSpeed inference optimization (based on langfod commit)
        try:
            # Use bfloat16 for better DeepSpeed compatibility
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = model.to(dtype=target_dtype, device=self.device)
            
            # Use init_inference with modern DeepSpeed configuration
            ds_model = deepspeed.init_inference(
                model=model,
                mp_size=1,  # Single GPU
                dtype=target_dtype,
                replace_with_kernel_inject=False,  # Disable to avoid dtype conflicts
            )
            
            logger.info(f"🚀 DeepSpeed init_inference applied to {model_name}")
            
            # Create a wrapper that preserves access to original model methods
            return DeepSpeedModelWrapper(ds_model, model)
            
        except Exception as e:
            logger.error(f"DeepSpeed init_inference failed: {e}")
            # Fallback to basic DeepSpeed initialization with consistent dtype
            try:
                # Ensure model is in consistent dtype before fallback initialization
                model = model.to(dtype=target_dtype, device=self.device)
                
                # Configure precision based on target dtype
                precision_config = {}
                if target_dtype == torch.bfloat16:
                    precision_config = {"bf16": {"enabled": True}}
                else:
                    precision_config = {"fp16": {"enabled": True, "auto_cast": False}}
                
                ds_model = deepspeed.initialize(
                    model=model,
                    config_params={
                        "train_batch_size": 1,
                        **precision_config,
                        "zero_optimization": {"stage": 0}  # No ZeRO for inference
                    }
                )[0]  # Get model from (model, optimizer, lr_scheduler, dataloader) tuple
                
                logger.info(f"🚀 DeepSpeed basic initialization applied to {model_name}")
                return DeepSpeedModelWrapper(ds_model, model)
                
            except Exception as e2:
                logger.error(f"DeepSpeed basic initialization also failed: {e2}")
                logger.warning("⚠️ DeepSpeed failed completely, using standard PyTorch model")
                # Return the original model in consistent dtype
                target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                return model.to(dtype=target_dtype, device=self.device)

    def get_model_names(self) -> List[str]:
        logger.info("Retrieving model names")
        return list(self.available_models.keys())

    def get_supported_languages(self) -> List[str]:
        logger.info("Retrieving supported languages")
        return supported_language_codes

    def get_model_conditioners(self, model_choice: str) -> List[str]:
        """Get list of conditioner names for a model"""
        logger.info(f"Retrieving conditioners for model: {model_choice}")
        model = self._ensure_model_loaded(model_choice)
        return [c.name for c in model.prefix_conditioner.conditioners]

    def generate_audio(
        self,
        model_choice: str,
        text: str,
        language: str = "en-us",
        speaker_audio: Optional[str] = None,
        prefix_audio: Optional[str] = None,
        emotion_values: List[float] = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
        vq_score: float = 0.78,
        fmax: float = 24000,
        pitch_std: float = 45.0,
        speaking_rate: float = 15.0,
        dnsmos_ovrl: float = 4.0,
        speaker_noised: bool = False,
        cfg_scale: float = 2.0,
        min_p: float = 0.15,
        seed: int = 420,
        randomize_seed: bool = True,
        unconditional_keys: List[str] = ["emotion"],
        top_p: float = 0.95,
        top_k: int = 50,
        linear: float = 1.0,
        confidence: float = 0.1,
        quadratic: float = 1.0,
        disable_torch_compile: bool = True,
    ) -> Tuple[Tuple[int, np.ndarray], int]:
        """
        Generate audio using the specified model and parameters.
        Returns a tuple of ((sample_rate, audio_data), seed).
        """
        logger.info(f"Generating audio for model: {model_choice}")
        
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > 10000:  # Reasonable limit
            raise ValueError(f"Text too long ({len(text)} chars). Maximum 10000 characters allowed")
        
        # Normalize language code
        normalized_language = normalize_language_code(language)
        logger.debug(f"Normalized language: {language} -> {normalized_language}")
        
        # Validate language code with better error message
        if normalized_language not in supported_language_codes:
            raise ValueError(f"Language code '{normalized_language}' isn't supported. "
                           f"Please pick a supported language code from: {supported_language_codes[:10]}... "
                           f"(total {len(supported_language_codes)} languages)")
        
        # Ensure the requested model is loaded
        selected_model = self._ensure_model_loaded(model_choice)
        actual_model_name = self.current_model_name
        logger.info(f"Mapped model name: {model_choice} -> {actual_model_name}")

        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)

        # Process speaker audio if provided
        speaker_embedding = None
        if speaker_audio is not None and "speaker" not in unconditional_keys:
            logger.info(f"Processing speaker audio: {speaker_audio}")
            wav, sr = torchaudio.load(speaker_audio)
            speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
            
            # Use consistent precision for all models
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            speaker_embedding = speaker_embedding.to(self.device, dtype=target_dtype)

        # Process prefix audio if provided
        audio_prefix_codes = None
        if prefix_audio is not None:
            logger.info(f"Processing prefix audio: {prefix_audio}")
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

        # Determine model's working dtype - ensure consistency
        working_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Prepare emotion tensor with consistent dtype
        emotion_tensor = torch.tensor(emotion_values, device=self.device, dtype=working_dtype)

        # Prepare VQ score tensor with consistent dtype  
        vq_tensor = torch.tensor([vq_score] * 8, device=self.device, dtype=working_dtype).unsqueeze(0)

        # Create conditioning dictionary
        logger.info("Creating conditioning dictionary")
        cond_dict = make_cond_dict(
            text=text,
            language=normalized_language,
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

        # sampling parameters
        sampling_params = {
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "linear": float(linear),
            "conf": float(confidence),
            "quad": float(quadratic)
        }

        # Generate audio
        logger.info("Generating audio")
        max_new_tokens = 86 * 60  # ~30 seconds of audio
        
        # Build generation kwargs
        generation_kwargs = {
            "prefix_conditioning": conditioning,
            "audio_prefix_codes": audio_prefix_codes,
            "max_new_tokens": max_new_tokens,
            "cfg_scale": float(cfg_scale),
            "batch_size": 1,
            "sampling_params": sampling_params,
        }
        
        # Add torch compile control if supported
        if disable_torch_compile:
            generation_kwargs["disable_torch_compile"] = True
            logger.debug("Torch compile disabled for better single-sample performance")
        
        codes = selected_model.generate(**generation_kwargs)

        # Check for empty generation (stability fix from commit 876e486)
        if hasattr(codes, 'shape') and len(codes.shape) >= 3 and codes.shape[2] == 0:
            logger.warning("Empty audio codes generated, retrying with different seed")
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            codes = selected_model.generate(**generation_kwargs)
        
        # Decode generated codes to waveform
        logger.info("Decoding generated audio to waveform")
        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]

        # Apply audio post-processing improvements from commit 3e14c9e
        wav_out = self._post_process_audio(wav_out, sr_out)
        
        logger.info("Audio generation complete")
        return (sr_out, wav_out.squeeze().numpy()), seed

    def _post_process_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Post-process generated audio with quality improvements.
        Based on improvements from coezbek/Zonos commit 3e14c9e.
        """
        try:
            # Normalize loudness to -23 LUFS for better consistency
            audio = self._normalize_loudness(audio, sr, target_lufs=-23.0)
            
            # Trim silence
            audio = self._trim_silence(audio)
            
            # Add fade in/out to avoid clicks
            audio = self._add_fade_inout(audio)
            
        except Exception as e:
            logger.warning(f"Audio post-processing failed: {e}, using original audio")
        
        return audio
    
    def _normalize_loudness(self, audio: torch.Tensor, sr: int, target_lufs: float = -23.0) -> torch.Tensor:
        """Normalize audio loudness using pyloudnorm"""
        try:
            import pyloudnorm
            
            # Set block size based on audio duration  
            block_size = 0.400 if audio.shape[1] > 2.0 * sr else 0.100
            meter = pyloudnorm.Meter(sr, block_size=block_size)
            loudness = meter.integrated_loudness(audio.cpu().numpy().T)
            
            gain_lufs = target_lufs - loudness
            gain = 10 ** (gain_lufs / 20.0)
            
            logger.debug(f"Loudness normalization: {loudness:.1f} -> {target_lufs:.1f} LUFS (gain: {gain:.3f})")
            return audio * gain
            
        except Exception as e:
            logger.debug(f"Loudness normalization skipped: {e}")
            return audio
    
    def _trim_silence(self, audio: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
        """Trim leading and trailing silence"""
        try:
            # Convert to energy (RMS)
            energy = audio.pow(2).mean(dim=0)
            threshold = 10 ** (threshold_db / 20.0)
            threshold = threshold ** 2  # Convert to energy
            
            # Find non-silent regions
            above_threshold = energy > threshold
            if not above_threshold.any():
                return audio
            
            # Find start and end indices
            start_idx = above_threshold.argmax().item()
            end_idx = len(above_threshold) - above_threshold.flip(0).argmax().item()
            
            # Add small padding
            padding = int(0.1 * audio.shape[1] / audio.shape[1] * 1000)  # 0.1s padding
            start_idx = max(0, start_idx - padding)
            end_idx = min(audio.shape[1], end_idx + padding)
            
            return audio[:, start_idx:end_idx]
            
        except Exception as e:
            logger.debug(f"Silence trimming skipped: {e}")
            return audio
    
    def _add_fade_inout(self, audio: torch.Tensor, fade_samples: int = 512) -> torch.Tensor:
        """Add fade-in and fade-out to prevent clicks"""
        try:
            if audio.shape[1] <= fade_samples * 2:
                return audio
            
            # Fade-in (linear)
            fade_in = torch.linspace(0, 1, fade_samples, device=audio.device)
            audio[:, :fade_samples] *= fade_in.unsqueeze(0)
            
            # Fade-out (improved to handle short audio - from commit f6e3288)
            blocksize = 512
            num_blocks = min((audio.shape[1] // blocksize) // 4, 20)  # Max 0.23s or 1/4 of audio
            if num_blocks > 0:
                fade_len = num_blocks * blocksize
                fade_out = torch.logspace(0, -10, fade_len, device=audio.device)
                audio[:, -fade_len:] *= fade_out.unsqueeze(0)
            
            return audio
            
        except Exception as e:
            logger.debug(f"Fade in/out skipped: {e}")
            return audio


class DeepSpeedModelWrapper:
    """Wrapper to preserve access to original model methods when using DeepSpeed"""
    
    def __init__(self, deepspeed_model, original_model):
        self.deepspeed_model = deepspeed_model
        self.original_model = original_model
        
    def generate(self, *args, **kwargs):
        """Use DeepSpeed-optimized model for generation"""
        return self.deepspeed_model.generate(*args, **kwargs)
    
    def make_speaker_embedding(self, *args, **kwargs):
        """Use original model for speaker embedding (not optimized by DeepSpeed)"""
        return self.original_model.make_speaker_embedding(*args, **kwargs)
    
    def prepare_conditioning(self, *args, **kwargs):
        """Use original model for conditioning preparation"""
        return self.original_model.prepare_conditioning(*args, **kwargs)
    
    @property
    def autoencoder(self):
        """Access original model's autoencoder"""
        return self.original_model.autoencoder
    
    def requires_grad_(self, requires_grad=True):
        """Pass through to both models"""
        self.deepspeed_model.requires_grad_(requires_grad)
        self.original_model.requires_grad_(requires_grad)
        return self
    
    def eval(self):
        """Pass through to both models"""
        self.deepspeed_model.eval()
        self.original_model.eval()
        return self
    
    def __getattr__(self, name):
        """Fallback to original model for any other attributes"""
        return getattr(self.original_model, name)
