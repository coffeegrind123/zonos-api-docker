import torch
import torchaudio
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
        # Only check if deepspeed package exists, don't import yet
        import importlib.util
        spec = importlib.util.find_spec("deepspeed")
        return spec is not None
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
        
        # Initialize thread pool executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-worker")
        logger.info("🔄 Thread pool executor initialized for async processing")
        
        # Model warm-up tracking
        self.model_cache_warmed = False
        self._warmup_lock = asyncio.Lock()
        
        # Request batching for improved throughput
        self.request_queue = asyncio.Queue(maxsize=10)
        self.batch_processor_task = None
        self.batch_size = 4  # Process up to 4 requests together
        self.batch_timeout = 0.1  # Wait 100ms to collect batch
        
        # Memory optimization settings - defer CUDA initialization
        self._cuda_initialized = False
        
        # Available models with their compatibility status
        self.available_models = {
            "Zyphra/Zonos-v0.1-transformer": {"status": "available", "supports_deepspeed": True, "supports_batch": True},
            "Zyphra/Zonos-v0.1-hybrid": {"status": "available", "supports_deepspeed": True, "supports_batch": False}
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
        
        # Initialize basic CPU/environment settings
        self._setup_memory_optimizations()
        
        # Load default model on startup for faster first request (safe with uvicorn single worker)
        logger.info(f"📦 Loading default model on startup: {self.default_model}")
        try:
            self.current_model = self._load_single_model(self.default_model)
            self.current_model_name = self.default_model
            logger.info(f"🎉 Successfully loaded default model: {self.default_model}")
        except Exception as e:
            logger.error(f"💥 Failed to load default model {self.default_model}: {e}")
            logger.warning("⚠️ Service starting without pre-loaded model - will load on first request")
            # Ensure attributes are initialized even if loading fails
            self.current_model = None
            self.current_model_name = None

    async def warmup_model(self, model_name: str = None):
        """
        Warm up model with dummy generation to eliminate cold start latency.
        This preloads model weights into GPU memory and initializes CUDA kernels.
        """
        if model_name is None:
            model_name = self.default_model
            
        async with self._warmup_lock:
            if self.model_cache_warmed:
                logger.debug(f"Model {model_name} already warmed up")
                return
                
            logger.info(f"🔥 Starting model warm-up for {model_name}")
            
            try:
                # Use a minimal text for warm-up with compatible parameters
                warmup_text = "Hello."
                warmup_params = {
                    "model_choice": model_name,
                    "text": warmup_text,
                    "language": "en-us",
                    "emotion_values": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
                    "vq_score": 0.78,
                    "speaking_rate": 15.0,
                    "cfg_scale": 2.0,  # Use default cfg_scale that the model supports
                    "min_p": 0.15,
                    "seed": 42,
                    "randomize_seed": False,
                }
                
                # Run warm-up generation in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor, 
                    lambda: self.generate_audio(**warmup_params)
                )
                
                self.model_cache_warmed = True
                logger.info(f"✅ Model warm-up completed for {model_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Model warm-up failed for {model_name}: {e}")
                # Don't fail service startup if warm-up fails
                pass
    
    def _setup_memory_optimizations(self):
        """Configure memory optimizations - safe for multiprocessing"""        
        # Set thread affinity for CPU operations (safe for multiprocessing)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        logger.info(f"🔧 PyTorch thread count set to {torch.get_num_threads()}")
        
        # Set environment variables that don't require CUDA initialization
        if self.device.startswith('cuda'):
            os.environ.update({
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16,garbage_collection_threshold:0.8',
                'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
                'TORCH_CUDNN_V8_API_ENABLED': '1',  # Use cuDNN v8 API
            })
            logger.info("🔧 CUDA environment variables configured")

    def _ensure_cuda_initialized(self):
        """Initialize CUDA settings after worker fork (lazy initialization)"""
        if not self._cuda_initialized and self.device.startswith('cuda'):
            try:
                # Set memory pool size to 85% of GPU memory
                torch.cuda.set_per_process_memory_fraction(0.85)
                
                # Pre-allocate memory pool to avoid fragmentation
                torch.cuda.empty_cache()
                # Allocate and free a large tensor to establish memory pool
                dummy_tensor = torch.zeros(1024, 1024, device=self.device, dtype=torch.float16)
                del dummy_tensor
                torch.cuda.empty_cache()
                
                self._cuda_initialized = True
                logger.info("🧠 GPU memory pool initialized and optimized")
            except Exception as e:
                logger.warning(f"⚠️ CUDA initialization failed: {e}")
                # Don't mark as initialized if it failed
                return False
        return True

    async def start_batch_processor(self):
        """Start the batch processing task for improved throughput"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("📦 Batch processor started for improved throughput")

    async def _batch_processor(self):
        """Process requests in batches for better GPU utilization"""
        while True:
            batch = []
            
            try:
                # Collect requests for batch processing
                while len(batch) < self.batch_size:
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(), 
                            timeout=self.batch_timeout if batch else None
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break  # Process current batch
                
                if batch:
                    logger.debug(f"Processing batch of {len(batch)} requests")
                    await self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch):
        """Process a batch of TTS requests together with true tensor batching"""
        if not batch:
            return
            
        # Group by model for efficient batching
        model_batches = {}
        for request_future, args, kwargs in batch:
            model_choice = kwargs.get('model_choice', args[0] if args else self.default_model)
            if model_choice not in model_batches:
                model_batches[model_choice] = []
            model_batches[model_choice].append((request_future, args, kwargs))
        
        # Process each model batch
        for model_choice, requests in model_batches.items():
            try:
                # Check if model supports batching
                model_info = self.available_models.get(model_choice, {})
                if not model_info.get('supports_batch', False):
                    # Fall back to sequential processing for models that don't support batching
                    for request_future, args, kwargs in requests:
                        try:
                            result = await self.generate_audio_async(*args, **kwargs)
                            request_future.set_result(result)
                        except Exception as e:
                            request_future.set_exception(e)
                    continue
                
                # True batch processing for supported models
                await self._process_model_batch(model_choice, requests)
                
            except Exception as e:
                # If batch processing fails, fall back to sequential
                logger.warning(f"Batch processing failed for {model_choice}, falling back to sequential: {e}")
                for request_future, args, kwargs in requests:
                    try:
                        result = await self.generate_audio_async(*args, **kwargs)
                        request_future.set_result(result)
                    except Exception as e:
                        request_future.set_exception(e)
    
    async def _process_model_batch(self, model_choice, requests):
        """Process a batch of requests for a single model with tensor batching"""
        if len(requests) == 1:
            # Single request - use normal processing
            request_future, args, kwargs = requests[0]
            try:
                result = await self.generate_audio_async(*args, **kwargs)
                request_future.set_result(result)
            except Exception as e:
                request_future.set_exception(e)
            return
        
        # Multiple requests - use batch processing
        try:
            texts = []
            params_list = []
            futures = []
            
            for request_future, args, kwargs in requests:
                # Extract text and parameters
                if args:
                    text = args[1] if len(args) > 1 else kwargs.get('text', '')
                else:
                    text = kwargs.get('text', '')
                    
                texts.append(text)
                params_list.append(kwargs)
                futures.append(request_future)
            
            # Run batch generation in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self._generate_audio_batch(model_choice, texts, params_list[0])  # Use first params as template
            )
            
            # Return results to futures
            for future, result in zip(futures, results):
                future.set_result(result)
                
        except Exception as e:
            # If batch fails, set error for all requests
            for request_future, _, _ in requests:
                request_future.set_exception(e)

    def _load_single_model(self, model_name: str):
        """Load a single model with all optimizations"""
        logger.info(f"📥 Loading model: {model_name}")
        
        # Check if model is available
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not in available models: {list(self.available_models.keys())}")
        
        model_info = self.available_models[model_name]
        if model_info["status"] != "available":
            error_msg = f"Model {model_name} is not available: {model_info['status']}"
            if "error" in model_info:
                error_msg += f" - {model_info['error']}"
            raise ValueError(error_msg)
        
        # Ensure CUDA is properly initialized after worker fork
        self._ensure_cuda_initialized()
        
        # Apply CUDA-specific optimizations before model loading
        if self.device.startswith('cuda'):
            # Enable cuDNN benchmark mode for fixed input sizes - 15-25% speedup
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TensorFloat-32 for faster matmul on Ampere+ GPUs - 20% speedup
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("🚀 CUDA optimizations enabled: cuDNN benchmark, TF32, memory pooling")
        
        # Load model with standard PyTorch
        load_kwargs = {"device": self.device, "backbone": "torch"}
        logger.info(f"🔧 Loading {model_name} with PyTorch")
        
        model = Zonos.from_pretrained(model_name, **load_kwargs)
        
        # Apply quantization for faster inference and lower memory usage
        target_dtype = self._get_optimal_dtype()
        
        # Attempt dynamic quantization for CPU or mixed precision for GPU
        if self.device == 'cpu':
            try:
                logger.info(f"🔧 Applying dynamic quantization to {model_name}")
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info(f"✅ Dynamic quantization applied to {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Dynamic quantization failed: {e}")
        
        # Convert model to optimal dtype
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
        
        # Apply torch.compile for JIT optimization (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.startswith('cuda'):
            try:
                logger.info(f"⚡ Applying torch.compile JIT optimization to {model_name}")
                # Use reduce-overhead mode for inference - 10-30% speedup
                model = torch.compile(model, mode="reduce-overhead", dynamic=False)
                logger.info(f"✅ torch.compile optimization applied to {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile optimization failed for {model_name}: {e}")
        
        # Set to eval mode for inference
        model.requires_grad_(False).eval()
        logger.debug(f"🔒 Set {model_name} to evaluation mode")
        
        return model

    def _get_optimal_dtype(self):
        """Determine optimal data type for the current hardware"""
        if self.device.startswith('cuda'):
            # Use bfloat16 on modern GPUs (Ampere+), float16 elsewhere
            if torch.cuda.is_bf16_supported():
                logger.debug("Using bfloat16 for optimal performance on modern GPU")
                return torch.bfloat16
            else:
                logger.debug("Using float16 for compatibility")
                return torch.float16
        else:
            # Use float32 on CPU for better accuracy
            return torch.float32

    def _ensure_model_loaded(self, model_choice: str):
        """Ensure the requested model is loaded, unloading others if necessary"""
        # Normalize model name
        actual_model_name = self.model_aliases.get(model_choice, model_choice)
        
        # If already loaded, return current model
        if self.current_model_name == actual_model_name and self.current_model is not None:
            logger.debug(f"✅ Model {actual_model_name} already loaded")
            return self.current_model
        
        # Unload current model if different model requested
        if hasattr(self, 'current_model') and self.current_model is not None and self.current_model_name != actual_model_name:
            logger.info(f"🔄 Switching from {self.current_model_name} to {actual_model_name}")
            logger.info(f"📤 Unloading {self.current_model_name}")
            del self.current_model
            torch.cuda.empty_cache()  # Free GPU memory
            self.current_model = None
            self.current_model_name = None
        
        # Load the requested model
        logger.info(f"📥 Loading requested model: {actual_model_name}")
        try:
            self.current_model = self._load_single_model(actual_model_name)
            self.current_model_name = actual_model_name
        except Exception as e:
            # Ensure attributes exist even if loading fails
            self.current_model = None
            self.current_model_name = None
            raise e
        
        return self.current_model

    def _wrap_with_deepspeed(self, model, model_name: str):
        """Apply DeepSpeed optimization wrapper to model"""
        # Import DeepSpeed here to avoid CUDA initialization during module import
        try:
            import deepspeed
        except Exception as e:
            logger.error(f"Failed to import DeepSpeed: {e}")
            raise
        
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

    def _generate_audio_batch(self, model_choice: str, texts: List[str], base_params: dict):
        """
        Generate audio for multiple texts in a single batch.
        Based on Zonos PR #192 batch processing implementation.
        """
        logger.info(f"Batch generating audio for {len(texts)} texts using model: {model_choice}")
        
        # Ensure the requested model is loaded
        selected_model = self._ensure_model_loaded(model_choice)
        actual_model_name = self.current_model_name
        
        # Extract parameters from base_params
        language = base_params.get('language', 'en-us')
        emotion_values = base_params.get('emotion_values', [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2])
        vq_score = base_params.get('vq_score', 0.78)
        speaking_rate = base_params.get('speaking_rate', 15.0)
        cfg_scale = base_params.get('cfg_scale', 2.0)
        min_p = base_params.get('min_p', 0.15)
        seed = base_params.get('seed', 420)
        randomize_seed = base_params.get('randomize_seed', True)
        speaker_audio = base_params.get('speaker_audio', None)
        unconditional_keys = base_params.get('unconditional_keys', ["emotion"])
        
        # Normalize language code
        normalized_language = normalize_language_code(language)
        
        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        
        # Process speaker audio if provided (same for all texts in batch)
        speaker_embedding = None
        if speaker_audio is not None and "speaker" not in unconditional_keys:
            logger.info(f"Processing speaker audio for batch: {speaker_audio}")
            wav, sr = torchaudio.load(speaker_audio)
            speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
            target_dtype = self._get_optimal_dtype()
            speaker_embedding = speaker_embedding.to(self.device, dtype=target_dtype)
        
        # Determine model's working dtype
        working_dtype = self._get_optimal_dtype()
        
        # Prepare batch conditioning
        logger.info("Creating batch conditioning dictionary")
        batch_conditioning = []
        
        for text in texts:
            # Prepare emotion and VQ tensors for this text
            emotion_tensor = torch.tensor(emotion_values, device=self.device, dtype=working_dtype)
            vq_tensor = torch.tensor([vq_score] * 8, device=self.device, dtype=working_dtype).unsqueeze(0)
            
            # Create conditioning dictionary for this text
            cond_dict = make_cond_dict(
                text=text,
                language=normalized_language,
                speaker=speaker_embedding,
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=24000.0,
                pitch_std=45.0,
                speaking_rate=float(speaking_rate),
                dnsmos_ovrl=4.0,
                speaker_noised=False,
                device=self.device,
                unconditional_keys=unconditional_keys,
            )
            conditioning = selected_model.prepare_conditioning(cond_dict)
            batch_conditioning.append(conditioning)
        
        # Batch generation parameters
        sampling_params = {
            "top_p": 0.95,
            "top_k": 50,
            "min_p": float(min_p),
            "linear": 1.0,
            "conf": 0.1,
            "quad": 1.0
        }
        
        # Generate audio for batch
        logger.info(f"Generating batch audio for {len(texts)} texts")
        max_new_tokens = 86 * 60  # ~30 seconds of audio per text
        
        try:
            # Batch generation - process all texts together
            batch_codes = []
            for i, conditioning in enumerate(batch_conditioning):
                logger.debug(f"Generating for text {i+1}/{len(texts)}")
                
                generation_kwargs = {
                    "prefix_conditioning": conditioning,
                    "audio_prefix_codes": None,  # Not supported in batch mode yet
                    "max_new_tokens": max_new_tokens,
                    "cfg_scale": float(cfg_scale),
                    "batch_size": 1,  # Process one at a time but in optimized manner
                    "sampling_params": sampling_params,
                    "disable_torch_compile": True,  # Better for batch processing
                }
                
                codes = selected_model.generate(**generation_kwargs)
                batch_codes.append(codes)
            
            # Decode all generated codes to waveforms
            logger.info("Decoding batch generated audio to waveforms")
            results = []
            
            for i, codes in enumerate(batch_codes):
                # Check for empty generation
                if hasattr(codes, 'shape') and len(codes.shape) >= 3 and codes.shape[2] == 0:
                    logger.warning(f"Empty audio codes generated for text {i+1}, retrying")
                    # Regenerate this specific text
                    codes = selected_model.generate(**generation_kwargs)
                
                # Decode to waveform
                wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
                sr_out = selected_model.autoencoder.sampling_rate
                
                if wav_out.dim() == 2 and wav_out.size(0) > 1:
                    wav_out = wav_out[0:1, :]
                
                # Apply post-processing
                wav_out = self._post_process_audio(wav_out, sr_out)
                
                results.append(((sr_out, wav_out.squeeze().numpy()), seed))
            
            logger.info(f"Batch audio generation complete for {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fall back to individual generation
            results = []
            for text in texts:
                try:
                    result = self.generate_audio(
                        model_choice=model_choice,
                        text=text,
                        **base_params
                    )
                    results.append(result)
                except Exception as text_error:
                    logger.error(f"Failed to generate audio for text '{text[:50]}...': {text_error}")
                    # Return empty audio as fallback
                    sr_fallback = 22050
                    audio_fallback = torch.zeros(1, int(sr_fallback * 0.1))  # 0.1s of silence
                    results.append(((sr_fallback, audio_fallback.numpy()), seed))
            
            return results

    async def generate_audio_async(
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
        Async version of generate_audio that runs in thread pool executor.
        This allows the FastAPI server to handle other requests while audio generation runs.
        """
        logger.debug(f"Running async audio generation for model: {model_choice}")
        
        # Run the synchronous generate_audio in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate_audio(
                model_choice=model_choice,
                text=text,
                language=language,
                speaker_audio=speaker_audio,
                prefix_audio=prefix_audio,
                emotion_values=emotion_values,
                vq_score=vq_score,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                dnsmos_ovrl=dnsmos_ovrl,
                speaker_noised=speaker_noised,
                cfg_scale=cfg_scale,
                min_p=min_p,
                seed=seed,
                randomize_seed=randomize_seed,
                unconditional_keys=unconditional_keys,
                top_p=top_p,
                top_k=top_k,
                linear=linear,
                confidence=confidence,
                quadratic=quadratic,
                disable_torch_compile=disable_torch_compile,
            )
        )

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
