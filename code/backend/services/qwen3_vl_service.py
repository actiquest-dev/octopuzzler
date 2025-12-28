## qwen3_vl_service.py

## **Путь**: `code/backend/services/qwen3_vl_service.py`

```python
"""
Qwen3-VL-30B-A3B-Instruct Service

Vision-Language model service for Octopus AI.
Enables avatar to "see" and understand objects shown via camera.

CRITICAL REQUIREMENTS:
- NVIDIA A100/H100/L40S (professional GPU)
- 24GB+ VRAM
- CUDA 12.2+
- transformers >= 4.37.0

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.1
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
import time
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Generation configuration for Qwen3-VL"""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


class Qwen3VLService:
    """
    Qwen3-VL-30B-A3B-Instruct service
    
    Provides vision-language capabilities for multi-modal understanding.
    
    Hardware Requirements:
    ----------------------
    RECOMMENDED:
    - NVIDIA A100 (40GB/80GB)
    - NVIDIA H100 (80GB)
    - NVIDIA L40S (48GB)
    
    NOT RECOMMENDED:
    - Consumer GPUs (RTX 4090) - 2-3× slower
    
    Performance Targets (A100):
    --------------------------
    - Text query: <800ms
    - Vision query: <1200ms
    - Model loading: <5s
    
    VRAM Usage:
    ----------
    - INT4 quantization: ~14GB
    - BF16 (full precision): ~60GB (H100 only)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        use_flash_attention: bool = True,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize Qwen3-VL service
        
        Args:
            model_name: HuggingFace model identifier
            use_flash_attention: Enable Flash Attention 2 (if supported)
            generation_config: Custom generation parameters
        """
        logger.info(f"Initializing Qwen3-VL service: {model_name}")
        logger.warning("CRITICAL: This requires professional GPU for production latency")
        
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig()
        
        # GPU verification
        self._verify_gpu()
        
        # Load model
        start_time = time.time()
        self._load_model(use_flash_attention)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.1f}s")
        
        # Warm-up
        logger.info("Warming up model (compiling CUDA kernels)...")
        self._warmup()
        logger.info("✓ Qwen3-VL service ready")
    
    def _verify_gpu(self):
        """Verify GPU compatibility and warn if consumer GPU"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! Professional GPU required.")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_capability = torch.cuda.get_device_capability(0)
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_memory:.1f}GB")
        logger.info(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Check if consumer GPU
        if any(x in gpu_name.upper() for x in ["RTX", "GEFORCE", "GTX"]):
            logger.warning("⚠️  CONSUMER GPU DETECTED!")
            logger.warning("⚠️  Expected latency: 2-3× slower than professional GPU")
            logger.warning("⚠️  Recommended: NVIDIA A100, H100, or L40S")
        
        # Check VRAM
        if gpu_memory < 24:
            raise RuntimeError(
                f"Insufficient VRAM: {gpu_memory:.1f}GB < 24GB required"
            )
        
        # Check compute capability for Flash Attention
        if compute_capability[0] < 8:
            logger.warning(
                f"Compute capability {compute_capability[0]}.{compute_capability[1]} < 8.0"
            )
            logger.warning("Flash Attention 2 not supported (will use standard attention)")
    
    def _load_model(self, use_flash_attention: bool):
        """Load model with INT4 quantization"""
        
        # Quantization configuration (critical for 30B model)
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
        
        # Model kwargs
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            **quantization_config
        }
        
        # Flash Attention 2 (if available)
        if use_flash_attention and self._supports_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("✓ Flash Attention 2 enabled")
        else:
            logger.info("Using standard attention")
        
        # Load model
        logger.info("Loading model (this may take 30-60 seconds)...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256*28*28,   # Minimum image resolution
            max_pixels=1280*28*28   # Maximum image resolution
        )
        
        # Set to eval mode
        self.model.eval()
        
        # Log VRAM usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"VRAM allocated: {allocated:.1f}GB, reserved: {reserved:.1f}GB")
    
    def _supports_flash_attention(self) -> bool:
        """Check if GPU supports Flash Attention 2"""
        compute_capability = torch.cuda.get_device_capability(0)
        # Requires compute capability >= 8.0 (A100, H100, etc.)
        return compute_capability[0] >= 8
    
    def _warmup(self):
        """
        Warm-up inference to compile CUDA kernels
        
        First inference is slow due to kernel compilation.
        This pre-compiles kernels for faster subsequent inferences.
        """
        dummy_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        ]
        
        text = self.processor.apply_chat_template(
            dummy_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        logger.info("Warm-up complete")
    
    async def generate_response(
        self,
        text: str,
        image: Optional[bytes] = None,
        user_profile: Optional[Dict] = None,
        emotion: str = "neutral",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate vision-aware response
        
        Args:
            text: User's speech/text input
            image: Optional camera frame (JPEG bytes)
            user_profile: User personalization data
                {
                    "name": str,
                    "conversation_topics": List[str],
                    "preferences": {
                        "response_style": "concise"|"balanced"|"detailed"
                    }
                }
            emotion: Current emotion state
            conversation_history: Recent messages (max 5 turns)
                [
                    {"role": "user", "content": [...], ...},
                    {"role": "assistant", "content": "...", ...}
                ]
        
        Returns:
            {
                "text": str,              # Generated response
                "tokens": int,            # Number of tokens generated
                "has_vision": bool,       # Whether image was processed
                "latency_ms": float,      # Total latency
                "generation_ms": float,   # Generation time only
                "vision_ms": float        # Vision processing time (if applicable)
            }
        """
        start_time = time.time()
        
        try:
            # Build messages
            messages = self._build_messages(
                text, image, user_profile, emotion, conversation_history
            )
            
            # Prepare text prompt
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision if image provided
            vision_start = time.time()
            image_inputs = None
            video_inputs = None
            
            if image:
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    logger.debug(f"Vision processing: {(time.time() - vision_start)*1000:.0f}ms")
                except Exception as e:
                    logger.error(f"Vision processing failed: {e}")
                    # Continue without vision
                    image = None
            
            vision_time = (time.time() - vision_start) * 1000 if image else 0
            
            # Tokenize
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            generation_start = time.time()
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    do_sample=self.generation_config.do_sample,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generation_time = (time.time() - generation_start) * 1000
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            response_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            total_latency = (time.time() - start_time) * 1000
            
            result = {
                "text": response_text,
                "tokens": len(generated_ids[0]),
                "has_vision": image is not None,
                "latency_ms": total_latency,
                "generation_ms": generation_time,
                "vision_ms": vision_time
            }
            
            logger.info(
                f"Generated response: {len(generated_ids[0])} tokens, "
                f"{total_latency:.0f}ms total "
                f"(gen: {generation_time:.0f}ms, vision: {vision_time:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    def _build_messages(
        self,
        text: str,
        image: Optional[bytes],
        user_profile: Optional[Dict],
        emotion: str,
        conversation_history: Optional[List[Dict]]
    ) -> List[Dict]:
        """
        Build Qwen3-VL message format
        
        Format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [{"type": "text", "text": "..."}, ...]},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        
        # System prompt
        system_content = self._build_system_prompt(user_profile, emotion)
        
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Conversation history (last 5 turns)
        if conversation_history:
            for msg in conversation_history[-5:]:
                messages.append(msg)
        
        # Current message
        user_message = {"role": "user", "content": []}
        
        # Add image if provided
        if image:
            try:
                pil_image = Image.open(io.BytesIO(image))
                user_message["content"].append({
                    "type": "image",
                    "image": pil_image
                })
                logger.debug(f"Image added: {pil_image.size}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
        
        # Add text
        user_message["content"].append({
            "type": "text",
            "text": text
        })
        
        messages.append(user_message)
        
        return messages
    
    def _build_system_prompt(
        self,
        user_profile: Optional[Dict],
        emotion: str
    ) -> str:
        """Build personalized system prompt"""
        
        # Extract user info
        name = "friend"
        topics = []
        response_style = "balanced"
        
        if user_profile:
            name = user_profile.get('name', 'friend')
            topics = user_profile.get('conversation_topics', [])
            preferences = user_profile.get('preferences', {})
            response_style = preferences.get('response_style', 'balanced')
        
        # Style instructions
        style_map = {
            "concise": "Keep responses brief and to the point (2-3 sentences).",
            "balanced": "Provide helpful responses with moderate detail (3-5 sentences).",
            "detailed": "Give comprehensive, detailed responses with examples."
        }
        style_instruction = style_map.get(response_style, style_map["balanced"])
        
        prompt = f"""You are Ollie, a friendly SpongeBob-style octopus AI assistant with vision capabilities.

Current context:
- User's name: {name}
- User's emotion: {emotion}
- Recent conversation topics: {', '.join(topics[:3]) if topics else 'None yet'}
- Response style: {response_style}

Your capabilities:
- Vision: You can see images from the camera. Describe what you see naturally and enthusiastically.
- Recognition: Identify objects, people, text, scenes, and answer visual questions accurately.
- Multi-modal: Combine visual and textual information for comprehensive responses.

Personality guidelines:
- Be warm, friendly, and expressive (inspired by SpongeBob characters)
- Show excitement when you see interesting things via camera
- Match the user's emotional state appropriately
- Reference past conversations naturally when relevant
- Use visual information to provide context-aware, helpful responses

Response style:
- {style_instruction}
- Use simple, clear language
- Be encouraging and positive
- Ask clarifying questions when needed

Example vision interactions:
- User shows book: "Oh wow! I can see you're showing me a book! The title is '[book title]'. That's a great read! What would you like to know about it?"
- User shows object: "Cool! That's a [object name]! [Brief description]. How can I help you with it?"
- User shows text: "I can read that! It says '[text content]'. [Helpful response based on content]"
- User shows scene: "I can see [describe scene]. [Relevant observation or suggestion]"

Remember: Be helpful, friendly, and make good use of what you can see!"""
        
        return prompt
    
    def get_model_info(self) -> Dict:
        """Get model information and status"""
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "generation_config": {
                "max_new_tokens": self.generation_config.max_new_tokens,
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p
            }
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
            logger.info("Qwen3-VL service cleaned up")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_qwen3vl():
        """Test Qwen3-VL service"""
        
        # Initialize service
        service = Qwen3VLService()
        
        # Test 1: Text-only query
        print("\n=== Test 1: Text-only ===")
        result = await service.generate_response(
            text="Hello! How are you?",
            user_profile={"name": "Michael"},
            emotion="happy"
        )
        print(f"Response: {result['text']}")
        print(f"Latency: {result['latency_ms']:.0f}ms")
        
        # Test 2: Vision query (if image available)
        print("\n=== Test 2: Vision query ===")
        try:
            with open("test_image.jpg", "rb") as f:
                image_bytes = f.read()
            
            result = await service.generate_response(
                text="What is this?",
                image=image_bytes,
                user_profile={"name": "Michael"},
                emotion="curious"
            )
            print(f"Response: {result['text']}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
            print(f"Vision processing: {result['vision_ms']:.0f}ms")
        except FileNotFoundError:
            print("No test image found (test_image.jpg)")
        
        # Print model info
        print("\n=== Model Info ===")
        info = service.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    asyncio.run(test_qwen3vl())
```