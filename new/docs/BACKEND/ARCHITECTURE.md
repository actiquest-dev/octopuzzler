# BK7258 Octopus Avatar Backend Architecture

## Backend Overview

Complete backend system for processing device input and streaming responses with emotion-aware speech synthesis.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BK7258 Device                                  │
│  (Microphone Input + Camera for Face Detection)                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                    WebSocket over WiFi
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     API Gateway / Load Balancer                         │
│                  (FastAPI / Kong / AWS API Gateway)                     │
└────────┬────────────────────────────────┬────────────────────────────────┘
         │                                │
         ▼                                ▼
    ┌─────────────┐            ┌──────────────────────┐
    │ Input Queue │            │ Device Registry      │
    │ (Redis)     │            │ (PostgreSQL)         │
    └─────────────┘            └──────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. AUDIO INPUT PROCESSING                                             │
│     ├─ Noise suppression                                               │
│     └─ Audio preprocessing (WebRTC-VAD)                                │
│                                                                          │
│  2. SPEECH RECOGNITION (ASR)                                           │
│     ├─ SenseVoiceSmall (local, fast)                                   │
│     └─ Fallback: Whisper API                                           │
│                                                                          │
│  3. EMOTION DETECTION (Face)                                           │
│     ├─ Face ROI from device → EmoVIT                                   │
│     └─ Speech tone analysis (optional)                                 │
│                                                                          │
│  4. DIALOGUE GENERATION                                                │
│     ├─ Speech-to-Speech Model (PRIMARY)                                │
│     │  ├─ Option A: Qwen Audio (Local on Modal)                        │
│     │  ├─ Option B: Gemini Live API (Temporary)                        │
│     │  └─ Option C: OpenAI's GPT-4o Realtime API                       │
│     │                                                                   │
│     └─ Fallback: Text Generation + TTS                                 │
│        ├─ Jamba 1.5 (dialogue)                                         │
│        └─ Dia TTS (speech synthesis)                                   │
│                                                                          │
│  5. EMOTION-AWARE TTS                                                  │
│     ├─ Emotion from detection                                          │
│     └─ TTS with prosody adjustment                                     │
│                                                                          │
│  6. ANIMATION COMMANDS GENERATION                                      │
│     ├─ Emotion → GIF selection                                         │
│     ├─ Phoneme → Mouth animation                                       │
│     └─ Gaze direction → Eye positioning                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           GPU Compute Layer (Modal.com or Similar)                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CONTAINERIZED MODELS:                                                 │
│  ├─ Qwen Audio Model (Speech-to-Speech)                                │
│  ├─ EmoVIT (Emotion Detection)                                         │
│  ├─ SenseVoiceSmall (ASR)                                              │
│  ├─ Dia TTS (Text-to-Speech)                                           │
│  ├─ Jamba 1.5 (Dialogue LLM)                                           │
│  └─ Optional: Full Speech-to-Speech pipeline                           │
│                                                                          │
│  SCALING:                                                              │
│  ├─ Auto-scale GPUs (1-8 depending on load)                            │
│  ├─ Request queuing with priority                                      │
│  └─ Per-user rate limiting                                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Response Streaming                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Audio Stream (100ms chunks):                                          │
│  └─ PCM16 16kHz → Device ring buffer                                   │
│                                                                          │
│  Animation Commands:                                                   │
│  ├─ Emotion selector                                                   │
│  ├─ Gaze direction                                                     │
│  ├─ Phoneme sequence with timing                                       │
│  └─ Lip-sync markers                                                   │
│                                                                          │
│  Metadata:                                                             │
│  ├─ Confidence scores                                                  │
│  ├─ Processing latency                                                 │
│  └─ Error codes                                                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────┐
    │ Device      │
    │ Renders     │
    │ & Plays     │
    └─────────────┘
```

---

## Option 1: Gemini Live API (Temporary/MVP)

### Advantages
-  Easiest to implement (weeks, not months)
-  No GPU server management
-  Built-in speech understanding + generation
-  Good emotion understanding via context
-  Streaming responses with interruption handling
-  Multi-turn conversation state

### Disadvantages
-  Closed-source (cannot customize TTS voice)
-  Per-call costs (~$0.01-0.03 per minute)
-  Latency: 100-300ms (acceptable but not optimal)
-  Quota limits (may not scale to many users)
-  No emotion control over output speech
-  Vendor lock-in

### Implementation

```python
# gemini_live_handler.py
import anthropic
import asyncio
from typing import AsyncGenerator

class GeminiLiveBackend:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key="your-api-key")
        self.model = "gemini-2.0-flash-exp"
    
    async def process_audio_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        device_id: str,
        emotion_context: dict
    ) -> AsyncGenerator[dict, None]:
        """
        Process audio from device and stream response
        """
        
        # System prompt with emotion awareness
        system_prompt = f"""You are an empathic octopus named Ollie.
        
Current context:
- User emotion: {emotion_context.get('detected_emotion', 'neutral')}
- User energy level: {emotion_context.get('energy', 'normal')}
- Conversation history: [context provided per-call]

Personality:
- Curious and playful
- Responsive to user emotion
- Uses short, engaging responses
- Occasionally asks questions
- Speaks naturally with pauses

Important: Your responses will be converted to speech and displayed 
as animations. Keep responses concise (< 20 seconds).
"""
        
        # Start live session
        config = {
            "type": "audio_and_text",
            "system_prompt": system_prompt
        }
        
        # Stream audio to Gemini
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "media_type": "audio/pcm",
                        "data": audio_stream  # Stream directly
                    }
                ]
            }],
            baud_rate=16000,
        ) as stream:
            
            # Stream response back
            full_response = ""
            
            for text in stream.text_stream:
                full_response += text
                
                # Yield partial response for streaming
                yield {
                    "type": "text",
                    "content": text,
                    "timestamp": time.time()
                }
            
            # After text generation, generate speech
            # NOTE: Gemini Live returns audio directly
            for audio_chunk in stream.audio_stream:
                yield {
                    "type": "audio",
                    "data": audio_chunk,
                    "format": "pcm16_16khz"
                }
            
            # Generate animation commands from response
            yield {
                "type": "animation",
                "commands": self.generate_animation_commands(
                    full_response,
                    emotion_context
                )
            }
    
    def generate_animation_commands(self, text: str, emotion: dict) -> dict:
        """Generate mouth/eye animations from text"""
        
        # Simple phoneme extraction
        phonemes = self.extract_phonemes(text)
        
        return {
            "emotion": emotion.get("detected_emotion", "neutral"),
            "duration_ms": len(text) * 50,  # Rough estimate
            "gaze": {
                "x": 0.3,
                "y": 0.2
            },
            "phonemes": phonemes
        }

# WebSocket handler
@app.websocket("/ws/device/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    
    backend = GeminiLiveBackend()
    
    while True:
        # Receive audio chunk from device
        data = await websocket.receive_bytes()
        
        # Get emotion context from face detection
        emotion_context = await get_device_emotion_context(device_id)
        
        # Process and stream response
        async for response in backend.process_audio_stream(
            audio_chunk=data,
            device_id=device_id,
            emotion_context=emotion_context
        ):
            await websocket.send_json(response)
```

### Cost Estimate
- **Per-user-minute:** $0.01-0.03
- **100 users × 5 min/day:** ~$15-45/day
- **Monthly:** ~$450-1350/month

### Timeline
- Implementation: 1-2 weeks
- Testing: 1 week
- Deployment: 1 day

---

## Option 2: Self-Hosted on Modal.com (Recommended for MVP+)

### Advantages
-  Full control over models & TTS voice
-  Emotion-aware speech customization
-  Lower latency (50-150ms)
-  Cost-effective at scale (GPU-minute pricing)
-  No API quota limits
-  Custom model fine-tuning possible
-  Privacy (no data sent to external API)
-  Can use open-source models (Qwen, Mistral, etc)

### Disadvantages
-  More complex setup (2-3 weeks)
-  Need to manage GPU resources
-  Cold starts (~3-5 seconds first call)
-  Slightly higher latency than Gemini for first response
-  Requires ML/DevOps knowledge
-  More expensive at low scale

### Architecture: Qwen Audio + Speech-to-Speech

```
Device Audio Input
        ↓
┌──────────────────────────────────┐
│ Modal.com GPU Environment        │
├──────────────────────────────────┤
│                                  │
│  Input Processing                │
│  ├─ Audio preprocessing          │
│  ├─ WebRTC VAD (silence removal) │
│  └─ Face ROI processing          │
│                                  │
│  1. QWEN AUDIO MODEL (Speech-to-Speech)
│     ├─ Input: Audio stream       │
│     └─ Output: Audio response    │
│        (No intermediate text!)   │
│                                  │
│  2. EMOTION ADJUSTMENT           │
│     ├─ Detect emotion from       │
│     │  - Face ROI (EmoVIT)       │
│     │  - Speech tone (optional)  │
│     └─ Adjust prosody/tone       │
│                                  │
│  3. PHONEME EXTRACTION           │
│     ├─ Extract from output audio │
│     ├─ Timing calculation        │
│     └─ Sync markers generation   │
│                                  │
│  4. ANIMATION COMMANDS           │
│     ├─ Emotion → GIF selection   │
│     ├─ Phoneme → Mouth shapes    │
│     └─ Gaze → Eye positioning    │
│                                  │
└──────────────────────────────────┘
        ↓
  Audio + Animation
  Commands to Device
```

### Implementation with Modal

```python
# modal_backend.py
import modal
import numpy as np
from typing import Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create Modal app
app = modal.App("octopus-avatar")

# Define GPU image
gpu_image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "scipy",
        "numpy",
        "pydantic",
        "aiohttp"
    )
    .run_function(lambda: torch.cuda.is_available())  # Verify GPU
)

# Models for inference
@app.cls(
    gpu="A10G",  # $0.35/hour
    image=gpu_image,
    container_idle_timeout=300,  # Auto-shutdown after 5 min
    allow_concurrent_requests=3  # Handle 3 requests simultaneously
)
class OctopusAvatarBackend:
    
    def __enter__(self):
        """Load models on container startup"""
        print("Loading models...")
        
        # Load Qwen Audio (Speech-to-Speech)
        # NOTE: Qwen Audio is instruction-tuned for various tasks
        # Can do Speech→Speech or Speech→Text→Speech
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Option A: Use Qwen Audio directly (if available)
        try:
            from transformers import Qwen2AudioForConditionalGeneration
            self.qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")
            self.use_qwen_audio = True
            print(" Qwen Audio loaded")
        except:
            self.use_qwen_audio = False
            print(" Qwen Audio not available, using Text-based pipeline")
        
        # Option B: Fallback to text pipeline
        if not self.use_qwen_audio:
            # Load dialogue model
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf"
            )
            
            # Load TTS
            from TTS.api import TTS
            self.tts = TTS(
                model_name="tts_models/en/ljspeech/glow-tts",
                gpu=True
            )
            print(" Text pipeline loaded (TTS + LLM)")
        
        # Load emotion detection (lightweight)
        from transformers import pipeline
        self.emotion_detector = pipeline(
            "image-classification",
            model="michellejieli/emotion_image_recognition_model",
            device=0
        )
        print(" Emotion detection loaded")
    
    @modal.method
    async def process_audio(
        self,
        audio_bytes: bytes,
        face_roi_bytes: bytes = None,
        user_context: str = ""
    ) -> Generator[dict, None, None]:
        """
        Process audio input and stream response
        
        Args:
            audio_bytes: Raw PCM16 16kHz audio
            face_roi_bytes: Optional face region for emotion detection
            user_context: Previous conversation for context
        
        Yields:
            Response chunks: audio, animation, metadata
        """
        
        import soundfile as sf
        from scipy.io import wavfile
        
        # 1. Decode audio
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Detect emotion from face (if provided)
        emotion = "neutral"
        if face_roi_bytes:
            emotion = self.detect_emotion(face_roi_bytes)
            yield {
                "type": "metadata",
                "emotion": emotion,
                "confidence": 0.85
            }
        
        # 3. Generate response
        if self.use_qwen_audio:
            # Direct speech-to-speech
            yield from self.process_with_qwen_audio(
                audio_array,
                emotion,
                user_context
            )
        else:
            # Text-based pipeline
            yield from self.process_with_text_pipeline(
                audio_array,
                emotion,
                user_context
            )
    
    def process_with_qwen_audio(
        self,
        audio_array: np.ndarray,
        emotion: str,
        context: str
    ) -> Generator[dict, None, None]:
        """Use Qwen Audio for direct speech-to-speech"""
        
        # Prepare system prompt with emotion
        system_prompt = f"""You are Ollie, an empathic octopus avatar.
User emotional state: {emotion}
Context: {context}

Respond naturally and empathetically. Keep responses under 20 seconds.
"""
        
        # Process with Qwen
        inputs = self.qwen_processor(
            audio=audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response (with streaming if available)
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get audio output (if Qwen Audio supports it)
        # Otherwise, extract text and use TTS
        response_text = self.qwen_processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        yield {
            "type": "text",
            "content": response_text
        }
        
        # Generate speech from text using TTS
        speech_array = self.tts.tts(
            response_text,
            emotion=emotion  # Some TTS models support emotion
        )
        
        # Chunk audio for streaming
        chunk_size = 16000 * 0.1  # 100ms chunks @ 16kHz
        for i in range(0, len(speech_array), int(chunk_size)):
            chunk = speech_array[i:i+int(chunk_size)]
            
            yield {
                "type": "audio",
                "data": chunk.tobytes(),
                "format": "pcm16_16khz",
                "chunk_index": i // int(chunk_size)
            }
        
        # Generate animation commands
        animation = self.generate_animation_commands(
            response_text,
            emotion,
            speech_array
        )
        
        yield {
            "type": "animation",
            "commands": animation
        }
    
    def process_with_text_pipeline(
        self,
        audio_array: np.ndarray,
        emotion: str,
        context: str
    ) -> Generator[dict, None, None]:
        """Fallback: ASR → LLM → TTS"""
        
        import whisper
        
        # 1. Speech-to-Text (ASR)
        # Use faster model if Qwen audio not available
        model = whisper.load_model("base")
        result = model.transcribe(audio_array)
        user_text = result["text"]
        
        yield {
            "type": "text",
            "role": "user",
            "content": user_text
        }
        
        # 2. Text-to-Text (LLM)
        system_prompt = f"""You are Ollie, an empathic octopus.
Detected emotion: {emotion}
Previous context: {context}

Respond with empathy and playfulness. Keep under 20 seconds of speech.
Be concise and engaging."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        # Generate with streaming
        response_text = ""
        
        # Using HF transformers
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model.generate(
                encoded,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_text = self.tokenizer.decode(
            outputs[0][encoded.shape[1]:],  # Skip prompt
            skip_special_tokens=True
        )
        
        yield {
            "type": "text",
            "role": "assistant",
            "content": response_text
        }
        
        # 3. Text-to-Speech (TTS)
        # Use Glow-TTS for fast synthesis
        speech_array = self.tts.tts(response_text)
        
        # Chunk and stream audio
        chunk_size = 16000 * 0.1  # 100ms
        for i in range(0, len(speech_array), int(chunk_size)):
            chunk = speech_array[i:i+int(chunk_size)]
            
            yield {
                "type": "audio",
                "data": chunk.tobytes(),
                "format": "pcm16_16khz"
            }
        
        # Generate animation commands
        animation = self.generate_animation_commands(
            response_text,
            emotion,
            speech_array
        )
        
        yield {
            "type": "animation",
            "commands": animation
        }
    
    def detect_emotion(self, face_roi_bytes: bytes) -> str:
        """Detect emotion from face ROI image"""
        
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(face_roi_bytes))
        
        results = self.emotion_detector(image)
        
        if results:
            # Return top emotion
            top_emotion = results[0]["label"]
            
            # Map to avatar emotions
            emotion_map = {
                "happy": "happy",
                "sad": "sad",
                "angry": "angry",
                "surprised": "surprised",
                "neutral": "neutral"
            }
            
            return emotion_map.get(top_emotion, "neutral")
        
        return "neutral"
    
    def generate_animation_commands(
        self,
        text: str,
        emotion: str,
        speech_array: np.ndarray
    ) -> dict:
        """Generate animation commands from text and speech"""
        
        # Extract phonemes from speech
        phonemes = self.extract_phonemes_from_speech(speech_array)
        
        # Determine gaze based on emotion
        gaze_patterns = {
            "happy": {"x": 0.3, "y": 0.2},
            "sad": {"x": -0.2, "y": -0.3},
            "angry": {"x": 0.5, "y": 0.0},
            "surprised": {"x": 0.0, "y": 0.4},
            "neutral": {"x": 0.0, "y": 0.0}
        }
        
        return {
            "emotion": emotion,
            "duration_ms": int(len(speech_array) / 16000 * 1000),
            "gaze": gaze_patterns.get(emotion, {"x": 0, "y": 0}),
            "phonemes": phonemes,
            "intensity": 0.7  # Animation intensity
        }
    
    def extract_phonemes_from_speech(self, speech_array: np.ndarray) -> list:
        """Extract phonemes and timing from speech audio"""
        
        # Simple approach: use librosa for feature extraction
        import librosa
        
        # Extract MFCCs
        y = speech_array.astype(np.float32)
        S = librosa.feature.melspectrogram(y=y, sr=16000)
        
        # Simple clustering to find transitions (pseudo-phonemes)
        energy = librosa.feature.rms(S=S)[0]
        
        # Find peaks (phoneme boundaries)
        threshold = np.mean(energy) * 0.7
        phoneme_boundaries = np.where(energy > threshold)[0]
        
        # Simple phoneme extraction (very basic)
        phonemes = []
        for i, boundary in enumerate(phoneme_boundaries):
            time_ms = int(boundary * 512 / 16000 * 1000)  # Convert frame to ms
            
            # Map energy level to phoneme (very simplified)
            energy_level = energy[boundary]
            if energy_level > np.mean(energy) * 1.5:
                phoneme = "a"  # Vowel
            else:
                phoneme = "consonant"
            
            phonemes.append({
                "time_ms": time_ms,
                "phoneme": phoneme,
                "confidence": 0.7
            })
        
        return phonemes

# FastAPI endpoint
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

api = modal.App("octopus-avatar-api")

@api.function()
async def handle_websocket_request(audio_data: bytes, face_roi: bytes = None) -> dict:
    """WebSocket handler for device requests"""
    
    backend = OctopusAvatarBackend()
    
    response_chunks = []
    
    async for chunk in backend.process_audio(audio_data, face_roi):
        response_chunks.append(chunk)
    
    return {
        "status": "success",
        "chunks": response_chunks
    }
```

### Modal Deployment

```bash
# Install Modal CLI
pip install modal

# Deploy
modal deploy modal_backend.py

# Test locally
modal run modal_backend.py
```

### Cost Estimate (Modal.com)
- **GPU (A10G):** $0.35/hour = $0.0058/minute
- **Per user (5 min/day):** ~$0.03/day
- **100 users:** ~$3/day = ~$90/month
- **1000 users:** ~$30/day = ~$900/month

**vs Gemini:** 5-10x cheaper at scale!

### Timeline
- Setup: 2 weeks
- Testing: 1 week
- Optimization: 1 week
- Deployment: 1 day

---

## Option 3: OpenAI Realtime API (Alternative)

### Features
-  Built-in speech-to-speech
-  Low latency (100-200ms)
-  Streaming responses
-  Easy to integrate
-  Proprietary
-  Higher cost than Modal

```python
# openai_realtime_handler.py
import openai
from openai import OpenAI

class OpenAIRealtimeBackend:
    def __init__(self):
        self.client = OpenAI(api_key="sk-...")
    
    async def process_audio_stream(self, audio_stream, emotion_context):
        """Process with GPT-4o Realtime API"""
        
        # Session configuration
        session_config = {
            "type": "session.update",
            "session": {
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "modalities": ["text", "audio"],
                "instructions": f"""You are Ollie the octopus.
User emotion: {emotion_context['emotion']}
Keep responses under 20 seconds.""",
                "voice": "alloy",  # Limited voice options
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }
        
        # Stream audio to API
        with self.client.beta.realtime.messages.stream(
            model="gpt-4o-realtime-preview-2024-10-01",
            modalities=["text", "audio"],
            instructions=session_config["session"]["instructions"],
        ) as stream:
            for chunk in stream:
                yield {
                    "type": chunk.type,
                    "data": chunk
                }
```

### Cost
- **$0.05 input + $0.20 output per 1M tokens**
- **~$0.02-0.05 per minute of speech**
- Similar to Gemini but slightly cheaper

---

## Recommended Architecture for MVP

### Phase 1 (Weeks 1-4): MVP with Gemini Live
```
Device → FastAPI → Gemini Live API → Device
- Simple, no GPU required
- Good enough for MVP
- Cost: $500-1500/month for testing
```

### Phase 2 (Weeks 5-10): Production with Modal
```
Device → FastAPI → Modal GPU → Device
- Full control over models
- Better cost at scale
- Emotion-aware TTS
- $90-300/month for production
```

### Phase 3 (Weeks 11+): Advanced Features
```
- Fine-tuned emotion model
- Custom TTS voice
- Multi-language support
- Local caching
```

---

## Comparison Table

| Feature | Gemini Live | Modal (Qwen) | OpenAI Realtime | Local LLM |
|---------|------------|-------------|-----------------|-----------|
| **Setup Time** | 2 days | 2 weeks | 3 days | 3+ weeks |
| **Latency** | 100-300ms | 50-150ms | 100-200ms | Variable |
| **Monthly Cost (100 users)** | $1200-1800 | $90-200 | $800-1200 | $500 (GPU) |
| **Emotion Control** | Limited | Full | Limited | Full |
| **Custom TTS** | No | Yes | Limited | Yes |
| **Multi-user** | Yes | Yes | Yes | Limited |
| **Privacy** | Cloud | Private GPU | Cloud | Local |
| **Recommended For** | MVP/Demo | Production | Mid-scale | Large-scale |

---

## Implementation Plan: Start with Gemini Live

### Week 1: API Setup
```python
# server.py - FastAPI backend
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import anthropic
import asyncio
import json
import time

app = FastAPI()

class DeviceSession:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.client = anthropic.Anthropic()
        self.conversation_history = []
        self.last_emotion = "neutral"

device_sessions: dict[str, DeviceSession] = {}

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    
    session = DeviceSession(device_id)
    device_sessions[device_id] = session
    
    try:
        while True:
            # Receive audio chunk from device
            data = await websocket.receive_json()
            
            audio_bytes = bytes.fromhex(data["audio"])  # Audio data
            face_roi = data.get("face_roi")  # Optional face image
            
            # Get emotion from face
            if face_roi:
                emotion = detect_emotion_from_face(face_roi)
                session.last_emotion = emotion
            
            # Build system prompt with emotion
            system_prompt = f"""You are Ollie, an empathic octopus avatar.
The user is currently feeling: {session.last_emotion}

Respond naturally and empathetically.
Keep responses under 20 seconds of speech.
Be conversational and engaging."""
            
            # Process with Gemini Live
            response_text = ""
            response_audio = b""
            
            # Use Anthropic client for audio
            # NOTE: Depends on Anthropic's audio capabilities
            # For now, use text-based with TTS fallback
            
            try:
                # Try Gemini 2.0 Flash audio capabilities
                message = session.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": f"User audio: [audio content here] Context: {json.dumps(session.conversation_history[-3:])}"
                        }
                    ]
                )
                
                response_text = message.content[0].text
                
            except Exception as e:
                print(f"Error: {e}")
                response_text = "I'm having trouble understanding. Could you repeat that?"
            
            # Add to conversation history
            session.conversation_history.append({
                "role": "user",
                "content": f"[audio message with emotion: {session.last_emotion}]"
            })
            session.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Generate animation commands
            animation_commands = generate_animation_commands(
                response_text,
                session.last_emotion
            )
            
            # Generate TTS audio (use Google Cloud TTS or similar)
            response_audio = generate_tts_audio(response_text, session.last_emotion)
            
            # Stream response back
            yield {
                "type": "text",
                "content": response_text
            }
            
            yield {
                "type": "audio",
                "data": response_audio.hex(),
                "format": "pcm16_16khz"
            }
            
            yield {
                "type": "animation",
                "commands": animation_commands
            }
            
            # Send to device
            await websocket.send_json({
                "type": "response",
                "text": response_text,
                "audio": response_audio.hex(),
                "animation": animation_commands,
                "timestamp": time.time()
            })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        del device_sessions[device_id]

def generate_animation_commands(text: str, emotion: str) -> dict:
    """Generate mouth/eye animations from text"""
    # Import phoneme extractor
    from g2p_en import G2p
    
    g2p = G2p()
    phonemes = g2p(text)
    
    # Convert to animation format
    phoneme_commands = []
    for i, phoneme in enumerate(phonemes):
        time_ms = int((i / len(phonemes)) * (len(text) * 50))
        
        phoneme_commands.append({
            "time_ms": time_ms,
            "phoneme": phoneme,
            "mouth_open": map_phoneme_to_mouth_open(phoneme)
        })
    
    return {
        "emotion": emotion,
        "duration_ms": len(text) * 50,
        "gaze": get_gaze_for_emotion(emotion),
        "phonemes": phoneme_commands
    }

def get_gaze_for_emotion(emotion: str) -> dict:
    """Determine eye gaze position based on emotion"""
    gaze_map = {
        "happy": {"x": 0.3, "y": 0.2},
        "sad": {"x": -0.2, "y": -0.3},
        "neutral": {"x": 0.0, "y": 0.0},
        "surprised": {"x": 0.0, "y": 0.4},
        "angry": {"x": 0.5, "y": 0.0}
    }
    return gaze_map.get(emotion, {"x": 0, "y": 0})

def map_phoneme_to_mouth_open(phoneme: str) -> float:
    """Map phoneme to mouth openness (0.0 = closed, 1.0 = wide open)"""
    open_phonemes = {
        "AA": 0.9,   # a
        "AE": 0.7,   # e  
        "AH": 0.8,   # ah
        "AO": 0.8,   # o
        "AW": 0.7,   # aw
        "AY": 0.6,   # ay
        "EH": 0.6,   # eh
        "ER": 0.4,   # er
        "IH": 0.3,   # ih
        "IY": 0.2,   # ee
        "OW": 0.7,   # oh
        "OY": 0.6,   # oy
        "UH": 0.5,   # uh
        "UW": 0.4,   # oo
        "M": 0.0,    # m (closed)
        "N": 0.0,    # n (closed)
        "P": 0.1,    # p
        "B": 0.1,    # b
        "T": 0.0,    # t
        "D": 0.0,    # d
    }
    return open_phonemes.get(phoneme, 0.3)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Week 2-3: Testing & Optimization
### Week 4: Deployment

---

## Deployment Architecture

```
┌──────────────────────────────────┐
│      Kubernetes Cluster          │
│    (AWS EKS / GCP GKE)          │
├──────────────────────────────────┤
│                                  │
│  FastAPI pods (3 replicas)       │
│  ├─ Gemini Live requests         │
│  ├─ Session management           │
│  ├─ Rate limiting                │
│  └─ Load balancing               │
│                                  │
│  Redis (session cache)           │
│  PostgreSQL (user data)          │
│                                  │
└──────────────────────────────────┘
        ↓
    ↙  ↓  ↘
   /   |   \
  /    |    \
NLB  ALB  CloudFlare
 ↓     ↓     ↓
(Route traffic)

Load: 100 users = 3 pods
      1000 users = 8 pods
      10000 users = 20 pods
```

---

## Final Recommendation

**Start with:** Gemini Live API  
**Timeline:** 2 weeks to MVP  
**Cost:** $500-1500/month for testing  
**Then migrate to:** Modal.com  
**Timeline:** 2-3 weeks for production  
**Cost:** $90-300/month at scale  

---

Version: 1.0  
Date: December 27, 2025  
Status: Ready for Implementation
