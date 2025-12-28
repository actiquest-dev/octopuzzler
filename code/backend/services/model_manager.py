"""
Dynamic Model Manager

Manages GPU VRAM by dynamically loading/unloading models.
Critical for fitting Qwen3-VL-30B (14GB) + DIA2 (6GB) on 24GB GPU.

Strategy:
- Always loaded: SenseVoice, EmoVIT, InsightFace (total: 4.5GB)
- On-demand: Qwen3-VL (14GB) OR DIA2 (6GB)
- Never simultaneously: Qwen3-VL + DIA2

VRAM Budget (24GB):
- Base services: 4.5GB
- Large model: 14GB (Qwen3-VL) or 6GB (DIA2)
- CUDA overhead: 0.5GB
- Buffer: 5GB

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.1
"""

import torch
import gc
import time
import logging
from typing import Optional, Dict, List
from enum import Enum

# Import services (these will be loaded dynamically)
from qwen3_vl_service import Qwen3VLService
from dia2_service import DIA2Service
from sensevoice_service import SenseVoiceService
from emovit_service import EmoVITService
from face_recognition_service import FaceRecognitionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for model types"""
    QWEN3_VL = "qwen3_vl"
    DIA2 = "dia2"
    NONE = "none"


class VRAMMonitor:
    """Monitor GPU VRAM usage"""
    
    @staticmethod
    def get_usage() -> Dict[str, float]:
        """Get current VRAM usage"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": free
        }
    
    @staticmethod
    def print_usage(prefix: str = ""):
        """Print VRAM usage"""
        usage = VRAMMonitor.get_usage()
        logger.info(
            f"{prefix}VRAM: "
            f"{usage['allocated']:.1f}GB allocated, "
            f"{usage['reserved']:.1f}GB reserved, "
            f"{usage['free']:.1f}GB free"
        )


class DynamicModelManager:
    """
    Dynamic model manager for GPU memory optimization
    
    Usage:
    ------
    manager = DynamicModelManager()
    
    # Generate LLM response (loads Qwen3-VL if needed)
    result = await manager.generate_llm_response(
        text="Hello",
        image=None,
        user_profile=user_profile,
        emotion="happy"
    )
    
    # Generate TTS (loads DIA2, unloads Qwen3-VL)
    audio = await manager.generate_tts(
        text="Hello world",
        emotion="happy"
    )
    
    Attributes:
    ----------
    loaded_model : ModelType
        Currently loaded large model (QWEN3_VL, DIA2, or NONE)
    
    qwen3_vl : Optional[Qwen3VLService]
        Qwen3-VL service instance (loaded on-demand)
    
    dia2 : Optional[DIA2Service]
        DIA2 service instance (loaded on-demand)
    
    Performance:
    -----------
    - Model swap time: 1-2s (A100), 2-3s (RTX 4090)
    - First load: Additional 3-5s
    - Subsequent loads: Fast (if weights cached)
    """
    
    def __init__(
        self,
        enable_profiling: bool = False
    ):
        """
        Initialize dynamic model manager
        
        Args:
            enable_profiling: Enable detailed timing profiling
        """
        logger.info("Initializing Dynamic Model Manager...")
        
        self.enable_profiling = enable_profiling
        
        # Track loaded model
        self.loaded_model = ModelType.NONE
        
        # On-demand services (large models)
        self.qwen3_vl: Optional[Qwen3VLService] = None
        self.dia2: Optional[DIA2Service] = None
        
        # Always-loaded services (small footprint)
        logger.info("Loading base services (always resident)...")
        self._load_base_services()
        
        VRAMMonitor.print_usage("Initial ")
        logger.info("✓ Dynamic Model Manager ready")
    
    def _load_base_services(self):
        """Load small services that are always resident"""
        
        try:
            # SenseVoice (STT + Audio Emotion) - ~1.5GB
            logger.info("Loading SenseVoice...")
            self.sensevoice = SenseVoiceService()
            
            # EmoVIT (Vision Emotion) - ~2GB
            logger.info("Loading EmoVIT...")
            self.emovit = EmoVITService()
            
            # InsightFace (Face Recognition) - ~1GB
            logger.info("Loading InsightFace...")
            self.insightface = FaceRecognitionService()
            
            logger.info("✓ Base services loaded")
            VRAMMonitor.print_usage("Base services: ")
            
        except Exception as e:
            logger.error(f"Failed to load base services: {e}")
            raise
    
    async def generate_llm_response(
        self,
        text: str,
        image: Optional[bytes] = None,
        user_profile: Optional[Dict] = None,
        emotion: str = "neutral",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate LLM response with Qwen3-VL
        
        Automatically manages model loading:
        1. Unload DIA2 if loaded
        2. Load Qwen3-VL if not loaded
        3. Generate response
        
        Args:
            text: User's text input
            image: Optional camera frame (JPEG bytes)
            user_profile: User personalization data
            emotion: Current emotion
            conversation_history: Recent messages
        
        Returns:
            {
                "text": str,
                "tokens": int,
                "has_vision": bool,
                "latency_ms": float,
                "generation_ms": float,
                "vision_ms": float,
                "swap_ms": float  # Model swap time (if swapped)
            }
        """
        start_time = time.time()
        
        # Ensure Qwen3-VL is loaded
        swap_time = await self._ensure_qwen_loaded()
        
        # Generate response
        try:
            result = await self.qwen3_vl.generate_response(
                text=text,
                image=image,
                user_profile=user_profile,
                emotion=emotion,
                conversation_history=conversation_history
            )
            
            # Add swap time to result
            result['swap_ms'] = swap_time
            
            total_time = (time.time() - start_time) * 1000
            logger.info(
                f"LLM response generated in {total_time:.0f}ms "
                f"(swap: {swap_time:.0f}ms, gen: {result['generation_ms']:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def generate_tts(
        self,
        text: str,
        emotion: str = "neutral",
        speaker_id: str = "S1"
    ) -> Dict:
        """
        Generate TTS with DIA2
        
        Automatically manages model loading:
        1. Unload Qwen3-VL if loaded
        2. Load DIA2 if not loaded
        3. Generate audio
        
        Args:
            text: Text to synthesize
            emotion: Emotion for TTS
            speaker_id: Speaker voice ID
        
        Returns:
            {
                "waveform": np.ndarray,
                "sample_rate": int,
                "duration_ms": float,
                "word_timestamps": List[Dict],
                "swap_ms": float  # Model swap time (if swapped)
            }
        """
        start_time = time.time()
        
        # Ensure DIA2 is loaded
        swap_time = await self._ensure_dia2_loaded()
        
        # Generate TTS
        try:
            result = await self.dia2.synthesize(
                text=text,
                speaker_id=speaker_id,
                emotion=emotion
            )
            
            # Add swap time to result
            result['swap_ms'] = swap_time
            
            total_time = (time.time() - start_time) * 1000
            logger.info(
                f"TTS generated in {total_time:.0f}ms "
                f"(swap: {swap_time:.0f}ms, synthesis: {result.get('synthesis_ms', 0):.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    async def _ensure_qwen_loaded(self) -> float:
        """
        Ensure Qwen3-VL is loaded, unload DIA2 if needed
        
        Returns:
            swap_time: Time spent swapping models (ms)
        """
        if self.loaded_model == ModelType.QWEN3_VL:
            # Already loaded
            logger.debug("Qwen3-VL already loaded")
            return 0.0
        
        swap_start = time.time()
        
        # Unload DIA2 if loaded
        if self.dia2 is not None:
            logger.info("Unloading DIA2 to make room for Qwen3-VL...")
            unload_start = time.time()
            
            del self.dia2
            self.dia2 = None
            
            # Force garbage collection and CUDA cache clear
            gc.collect()
            torch.cuda.empty_cache()
            
            unload_time = (time.time() - unload_start) * 1000
            logger.info(f"✓ DIA2 unloaded in {unload_time:.0f}ms")
            VRAMMonitor.print_usage("After unload: ")
        
        # Load Qwen3-VL if not loaded
        if self.qwen3_vl is None:
            logger.info("Loading Qwen3-VL...")
            load_start = time.time()
            
            self.qwen3_vl = Qwen3VLService()
            
            load_time = (time.time() - load_start) * 1000
            logger.info(f"✓ Qwen3-VL loaded in {load_time:.0f}ms")
            VRAMMonitor.print_usage("After load: ")
        
        self.loaded_model = ModelType.QWEN3_VL
        
        swap_time = (time.time() - swap_start) * 1000
        return swap_time
    
    async def _ensure_dia2_loaded(self) -> float:
        """
        Ensure DIA2 is loaded, unload Qwen3-VL if needed
        
        Returns:
            swap_time: Time spent swapping models (ms)
        """
        if self.loaded_model == ModelType.DIA2:
            # Already loaded
            logger.debug("DIA2 already loaded")
            return 0.0
        
        swap_start = time.time()
        
        # Unload Qwen3-VL if loaded
        if self.qwen3_vl is not None:
            logger.info("Unloading Qwen3-VL to make room for DIA2...")
            unload_start = time.time()
            
            del self.qwen3_vl
            self.qwen3_vl = None
            
            gc.collect()
            torch.cuda.empty_cache()
            
            unload_time = (time.time() - unload_start) * 1000
            logger.info(f"✓ Qwen3-VL unloaded in {unload_time:.0f}ms")
            VRAMMonitor.print_usage("After unload: ")
        
        # Load DIA2 if not loaded
        if self.dia2 is None:
            logger.info("Loading DIA2...")
            load_start = time.time()
            
            self.dia2 = DIA2Service()
            
            load_time = (time.time() - load_start) * 1000
            logger.info(f"✓ DIA2 loaded in {load_time:.0f}ms")
            VRAMMonitor.print_usage("After load: ")
        
        self.loaded_model = ModelType.DIA2
        
        swap_time = (time.time() - swap_start) * 1000
        return swap_time
    
    async def preload_model(self, model_type: ModelType):
        """
        Preload a model in advance
        
        Useful for reducing latency on first use.
        
        Args:
            model_type: Model to preload (QWEN3_VL or DIA2)
        """
        if model_type == ModelType.QWEN3_VL:
            await self._ensure_qwen_loaded()
        elif model_type == ModelType.DIA2:
            await self._ensure_dia2_loaded()
        else:
            logger.warning(f"Unknown model type: {model_type}")
    
    def get_loaded_model(self) -> ModelType:
        """Get currently loaded large model"""
        return self.loaded_model
    
    def get_vram_status(self) -> Dict:
        """
        Get detailed VRAM status
        
        Returns:
            {
                "usage": {...},
                "loaded_model": str,
                "models": {
                    "base": ["sensevoice", "emovit", "insightface"],
                    "large": "qwen3_vl" or "dia2" or None
                }
            }
        """
        return {
            "usage": VRAMMonitor.get_usage(),
            "loaded_model": self.loaded_model.value,
            "models": {
                "base": ["sensevoice", "emovit", "insightface"],
                "large": self.loaded_model.value if self.loaded_model != ModelType.NONE else None
            }
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        logger.info("Cleaning up Dynamic Model Manager...")
        
        if self.qwen3_vl is not None:
            del self.qwen3_vl
        
        if self.dia2 is not None:
            del self.dia2
        
        # Base services cleanup handled by their own __del__
        
        torch.cuda.empty_cache()
        logger.info("✓ Dynamic Model Manager cleaned up")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_model_manager():
        """Test dynamic model manager"""
        
        # Initialize manager
        manager = DynamicModelManager()
        
        print("\n=== Test 1: LLM Response (loads Qwen3-VL) ===")
        result = await manager.generate_llm_response(
            text="Hello! How are you?",
            user_profile={"name": "Test User"},
            emotion="happy"
        )
        print(f"Response: {result['text'][:100]}...")
        print(f"Swap time: {result['swap_ms']:.0f}ms")
        print(f"Generation time: {result['generation_ms']:.0f}ms")
        
        print(f"\nLoaded model: {manager.get_loaded_model()}")
        
        print("\n=== Test 2: TTS (swaps to DIA2) ===")
        result = await manager.generate_tts(
            text="Hello, this is a test.",
            emotion="neutral"
        )
        print(f"Audio duration: {result['duration_ms']:.0f}ms")
        print(f"Swap time: {result['swap_ms']:.0f}ms")
        
        print(f"\nLoaded model: {manager.get_loaded_model()}")
        
        print("\n=== Test 3: Another LLM Response (swaps back to Qwen3-VL) ===")
        result = await manager.generate_llm_response(
            text="Tell me about AI",
            emotion="curious"
        )
        print(f"Response: {result['text'][:100]}...")
        print(f"Swap time: {result['swap_ms']:.0f}ms")
        
        print(f"\nLoaded model: {manager.get_loaded_model()}")
        
        # VRAM status
        print("\n=== VRAM Status ===")
        status = manager.get_vram_status()
        print(f"Usage: {status['usage']}")
        print(f"Loaded: {status['loaded_model']}")
        print(f"Models: {status['models']}")
    
    asyncio.run(test_model_manager())