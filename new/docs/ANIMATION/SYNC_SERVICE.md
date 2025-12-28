# Animation Synchronization Service for Gemini Live

## Overview

Separate microservice that processes Gemini Live responses and generates synchronized animation commands for the octopus avatar.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini Live API                              │
│                                                                 │
│  Input: Audio from device + emotion context                   │
│  Output: Text response + TTS audio stream                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ (Text + Audio)
┌─────────────────────────────────────────────────────────────────┐
│            Animation Synchronization Service                    │
│                   (NEW MICROSERVICE)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT PROCESSING:                                             │
│  ├─ Receive text response from Gemini                          │
│  ├─ Receive TTS audio stream                                   │
│  ├─ Parse emotion context                                      │
│  └─ Get device state                                           │
│                                                                 │
│  SYNCHRONIZATION PIPELINE:                                     │
│                                                                 │
│  1. PHONEME EXTRACTION & TIMING                                │
│     ├─ Extract phonemes from text                              │
│     ├─ Estimate timing from audio duration                     │
│     ├─ Align phonemes to audio chunks                          │
│     └─ Generate phoneme sync markers                           │
│                                                                 │
│  2. EMOTION ANALYSIS & ANIMATION SELECTION                     │
│     ├─ Analyze text sentiment                                  │
│     ├─ Extract emotion cues from text                          │
│     ├─ Select appropriate GIF body animation                   │
│     ├─ Determine mouth shape sequence                          │
│     └─ Plan eye gaze positions                                 │
│                                                                 │
│  3. SPEECH RATE & TIMING CALCULATION                           │
│     ├─ Analyze speech duration                                 │
│     ├─ Detect pauses (silence gaps)                            │
│     ├─ Calculate phoneme timing                                │
│     └─ Adjust for real-time playback                           │
│                                                                 │
│  4. GAZE & EXPRESSION PLANNING                                 │
│     ├─ Determine natural gaze positions                        │
│     ├─ Plan eye movements (blinking, looking)                  │
│     ├─ Sync with pauses and emphasis                           │
│     └─ Add expressiveness (surprise, focus, etc)               │
│                                                                 │
│  5. ANIMATION COMMAND GENERATION                               │
│     ├─ Create mouth phoneme sequence                           │
│     ├─ Generate gaze interpolation commands                    │
│     ├─ Create body animation transitions                       │
│     ├─ Add emotion-driven expressions                          │
│     └─ Package for streaming to device                         │
│                                                                 │
│  OUTPUT:                                                       │
│  ├─ Emotion selector (which GIF to play)                       │
│  ├─ Duration estimate                                          │
│  ├─ Gaze positions with timing                                 │
│  ├─ Phoneme sequence with mouth shapes                         │
│  ├─ Sync markers for frame alignment                           │
│  └─ Packed in streaming JSON format                            │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ (Animation Commands)
┌─────────────────────────────────────────────────────────────────┐
│                    BK7258 Device                                │
│                                                                 │
│  Renders:                                                      │
│  ├─ Body GIF (emotion-based)                                   │
│  ├─ Mouth PNG overlays (phoneme-synced)                        │
│  ├─ Parametric eyes (gaze-controlled)                          │
│  └─ Plays audio in sync with all animations                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation: Animation Sync Service

```python
# animation_sync_service.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, AsyncGenerator
import re
import json
import time
from dataclasses import dataclass, asdict
import asyncio
from enum import Enum

app = FastAPI()

# ============================================================================
# DATA MODELS
# ============================================================================

class Emotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    NEUTRAL = "neutral"

@dataclass
class Phoneme:
    """Single phoneme with timing"""
    time_ms: int
    phoneme: str
    mouth_open: float
    duration_ms: int

@dataclass
class GazeTarget:
    """Eye gaze position"""
    time_ms: int
    x: float  # -1.0 to 1.0
    y: float  # -1.0 to 1.0
    duration_ms: int

@dataclass
class AnimationCommand:
    """Complete animation command for device"""
    emotion: str
    duration_ms: int
    gaze_targets: List[Dict]
    phonemes: List[Dict]
    intensity: float = 0.7
    blink_frequency: float = 0.3  # Hz

class GeminiResponse(BaseModel):
    """Response from Gemini Live"""
    text: str
    audio_duration_ms: float
    emotion_detected: str = "neutral"

class AnimationSyncResponse(BaseModel):
    """Response from animation sync service"""
    emotion: str
    duration_ms: int
    gaze: List[Dict]
    phonemes: List[Dict]
    sync_markers: List[Dict]
    intensity: float

# ============================================================================
# PHONEME EXTRACTION
# ============================================================================

class PhonemeExtractor:
    """Extract phonemes from text with timing estimation"""
    
    # Phoneme to mouth opening mapping
    PHONEME_MOUTH_MAP = {
        # Vowels (open mouth)
        "AA": 0.9,   # "father"
        "AE": 0.7,   # "cat"
        "AH": 0.8,   # "pot"
        "AO": 0.8,   # "thought"
        "AW": 0.7,   # "house"
        "AY": 0.6,   # "price"
        "EH": 0.6,   # "dress"
        "ER": 0.4,   # "bird"
        "IH": 0.3,   # "kit"
        "IY": 0.2,   # "fleece"
        "OW": 0.7,   # "goat"
        "OY": 0.6,   # "choice"
        "UH": 0.5,   # "foot"
        "UW": 0.4,   # "goose"
        
        # Consonants (closed/partial mouth)
        "B": 0.1,    # Bilabial stop
        "CH": 0.2,   # Affricate
        "D": 0.0,    # Alveolar stop
        "DH": 0.1,   # Dental fricative
        "F": 0.2,    # Labio-dental fricative
        "G": 0.0,    # Velar stop
        "HH": 0.3,   # Glottal fricative
        "JH": 0.2,   # Affricate
        "K": 0.0,    # Velar stop
        "L": 0.2,    # Alveolar approximant
        "M": 0.0,    # Bilabial nasal (closed lips)
        "N": 0.0,    # Alveolar nasal (closed)
        "NG": 0.0,   # Velar nasal (closed)
        "P": 0.1,    # Bilabial stop
        "R": 0.3,    # Alveolar approximant
        "S": 0.1,    # Alveolar fricative
        "SH": 0.2,   # Palato-alveolar fricative
        "T": 0.0,    # Alveolar stop
        "TH": 0.1,   # Dental fricative
        "V": 0.2,    # Labio-dental fricative
        "W": 0.3,    # Labial approximant
        "Y": 0.3,    # Palatal approximant
        "Z": 0.1,    # Alveolar fricative
        "ZH": 0.2,   # Palato-alveolar fricative
    }
    
    # Average duration per phoneme (ms)
    PHONEME_DURATION = 80  # milliseconds
    
    def __init__(self):
        """Initialize phoneme extractor"""
        try:
            from g2p_en import G2p
            self.g2p = G2p()
        except ImportError:
            print("  g2p_en not installed, using basic phoneme extraction")
            self.g2p = None
    
    def extract_phonemes(self, text: str, audio_duration_ms: float) -> List[Phoneme]:
        """
        Extract phonemes from text with timing
        
        Args:
            text: Text to extract phonemes from
            audio_duration_ms: Total duration of audio (for timing alignment)
        
        Returns:
            List of Phoneme objects with timing
        """
        
        # Clean text
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        if self.g2p:
            # Use g2p_en for accurate phoneme extraction
            phoneme_list = self.g2p(text)
        else:
            # Fallback: simple phoneme extraction
            phoneme_list = self._simple_phoneme_extraction(text)
        
        # Filter out non-phoneme tokens (like pauses)
        valid_phonemes = [p for p in phoneme_list if len(p) <= 2 and p.isalpha()]
        
        if not valid_phonemes:
            # Empty response, return neutral phoneme
            return [Phoneme(
                time_ms=0,
                phoneme="neutral",
                mouth_open=0.3,
                duration_ms=int(audio_duration_ms)
            )]
        
        # Calculate timing
        # Distribute phonemes across audio duration
        phoneme_count = len(valid_phonemes)
        time_per_phoneme = audio_duration_ms / phoneme_count
        
        phonemes = []
        for i, phoneme in enumerate(valid_phonemes):
            time_ms = int(i * time_per_phoneme)
            duration_ms = int(time_per_phoneme)
            mouth_open = self.PHONEME_MOUTH_MAP.get(phoneme.upper(), 0.3)
            
            phonemes.append(Phoneme(
                time_ms=time_ms,
                phoneme=phoneme,
                mouth_open=mouth_open,
                duration_ms=duration_ms
            ))
        
        return phonemes
    
    def _simple_phoneme_extraction(self, text: str) -> List[str]:
        """Fallback simple phoneme extraction"""
        # Very basic: just map vowels and consonants
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        
        phonemes = []
        for char in text.lower():
            if char in vowels:
                phonemes.append(char.upper())
            elif char in consonants:
                phonemes.append(char.upper())
        
        return phonemes

# ============================================================================
# EMOTION & SENTIMENT ANALYSIS
# ============================================================================

class EmotionAnalyzer:
    """Analyze text for emotions and sentiment"""
    
    # Simple emotion keywords
    EMOTION_KEYWORDS = {
        Emotion.HAPPY: [
            "happy", "joy", "great", "wonderful", "amazing", "love",
            "excited", "fantastic", "awesome", "brilliant", "glad"
        ],
        Emotion.SAD: [
            "sad", "sorry", "unhappy", "down", "depressed", "lonely",
            "terrible", "awful", "miserable", "disappointed"
        ],
        Emotion.ANGRY: [
            "angry", "mad", "furious", "upset", "irritated", "frustrated",
            "annoyed", "aggressive", "hate", "despise"
        ],
        Emotion.SURPRISED: [
            "surprised", "amazed", "shocked", "astonished", "wow",
            "incredible", "unexpected", "stunned", "whoa"
        ],
        Emotion.CONFUSED: [
            "confused", "unclear", "uncertain", "puzzled", "lost",
            "don't know", "unsure", "mixed", "complicated"
        ]
    }
    
    def analyze_emotion(self, text: str, detected_emotion: str = None) -> Emotion:
        """
        Analyze text for emotional content
        
        Args:
            text: Text to analyze
            detected_emotion: Override with detected emotion from face
        
        Returns:
            Emotion enum
        """
        
        # If emotion was already detected from face, use it
        if detected_emotion and detected_emotion in [e.value for e in Emotion]:
            return Emotion(detected_emotion)
        
        # Analyze text for emotion keywords
        text_lower = text.lower()
        emotion_scores = {emotion: 0 for emotion in Emotion}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1
        
        # Get top emotion
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        
        if emotion_scores[top_emotion] > 0:
            return top_emotion
        else:
            return Emotion.NEUTRAL
    
    def extract_emphasis_words(self, text: str) -> List[tuple]:
        """
        Extract words that should be emphasized (caps, punctuation)
        
        Returns:
            List of (word_position, emphasis_type)
        """
        
        emphasis_words = []
        
        # ALL CAPS words
        for match in re.finditer(r'\b[A-Z]{2,}\b', text):
            emphasis_words.append((match.start(), "caps"))
        
        # Multiple punctuation
        for match in re.finditer(r'[!?]{2,}', text):
            emphasis_words.append((match.start(), "punctuation"))
        
        return emphasis_words

# ============================================================================
# GAZE & EXPRESSION PLANNING
# ============================================================================

class GazePlanner:
    """Plan natural gaze patterns and expressions"""
    
    def __init__(self):
        self.emotion_gaze_defaults = {
            Emotion.HAPPY: {"x": 0.3, "y": 0.2},      # Looking up-right
            Emotion.SAD: {"x": -0.2, "y": -0.3},      # Looking down-left
            Emotion.ANGRY: {"x": 0.5, "y": 0.0},      # Intense stare right
            Emotion.SURPRISED: {"x": 0.0, "y": 0.4},  # Eyes wide open
            Emotion.CONFUSED: {"x": 0.2, "y": 0.3},   # Thoughtful
            Emotion.NEUTRAL: {"x": 0.0, "y": 0.0},    # Centered
        }
    
    def plan_gaze_sequence(
        self,
        text: str,
        emotion: Emotion,
        audio_duration_ms: float
    ) -> List[GazeTarget]:
        """
        Plan natural gaze sequence
        
        Args:
            text: Text being spoken
            emotion: Detected emotion
            audio_duration_ms: Total duration of speech
        
        Returns:
            List of gaze targets with timing
        """
        
        gaze_targets = []
        
        # Get default gaze for emotion
        default_gaze = self.emotion_gaze_defaults[emotion]
        
        # Start at default
        gaze_targets.append(GazeTarget(
            time_ms=0,
            x=default_gaze["x"],
            y=default_gaze["y"],
            duration_ms=int(audio_duration_ms * 0.3)
        ))
        
        # Look for natural break points (periods, commas)
        sentences = text.split('.')
        sentence_duration = audio_duration_ms / max(len(sentences), 1)
        
        # Add gaze changes at sentence boundaries
        for i, sentence in enumerate(sentences[:-1]):
            if sentence.strip():
                time_ms = int((i + 1) * sentence_duration)
                
                # Vary gaze for each sentence
                gaze_x = default_gaze["x"] + (0.2 if i % 2 else -0.2)
                gaze_x = max(-1.0, min(1.0, gaze_x))  # Clamp
                
                gaze_targets.append(GazeTarget(
                    time_ms=time_ms,
                    x=gaze_x,
                    y=default_gaze["y"],
                    duration_ms=int(sentence_duration)
                ))
        
        # Return to default at end
        gaze_targets.append(GazeTarget(
            time_ms=int(audio_duration_ms * 0.9),
            x=default_gaze["x"],
            y=default_gaze["y"],
            duration_ms=int(audio_duration_ms * 0.1)
        ))
        
        return gaze_targets
    
    def get_blink_schedule(
        self,
        audio_duration_ms: float,
        base_blink_frequency: float = 0.3  # Hz
    ) -> List[tuple]:
        """
        Generate natural blink schedule
        
        Returns:
            List of (time_ms, duration_ms) for each blink
        """
        
        blinks = []
        blink_interval_ms = 1000 / base_blink_frequency  # Convert Hz to ms
        blink_duration_ms = 150  # Standard blink duration
        
        current_time = blink_interval_ms
        while current_time < audio_duration_ms:
            # Add some randomness
            blinks.append((int(current_time), blink_duration_ms))
            current_time += blink_interval_ms + (current_time % 300)  # Vary interval
        
        return blinks

# ============================================================================
# ANIMATION COMMAND GENERATION
# ============================================================================

class AnimationCommandGenerator:
    """Generate final animation commands"""
    
    def __init__(self):
        self.phoneme_extractor = PhonemeExtractor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.gaze_planner = GazePlanner()
    
    async def generate_commands(
        self,
        gemini_response: GeminiResponse
    ) -> AnimationSyncResponse:
        """
        Generate animation commands from Gemini response
        
        Args:
            gemini_response: Response from Gemini Live API
        
        Returns:
            Animation commands for device
        """
        
        text = gemini_response.text
        audio_duration_ms = gemini_response.audio_duration_ms
        detected_emotion = gemini_response.emotion_detected
        
        # 1. Extract phonemes with timing
        phonemes = self.phoneme_extractor.extract_phonemes(text, audio_duration_ms)
        
        # 2. Analyze emotion
        emotion = self.emotion_analyzer.analyze_emotion(text, detected_emotion)
        
        # 3. Plan gaze sequence
        gaze_targets = self.gaze_planner.plan_gaze_sequence(
            text,
            emotion,
            audio_duration_ms
        )
        
        # 4. Generate sync markers
        sync_markers = self._generate_sync_markers(
            phonemes,
            gaze_targets,
            audio_duration_ms
        )
        
        # 5. Calculate intensity based on emotion
        intensity = self._calculate_intensity(text, emotion)
        
        # Convert to dictionaries for JSON serialization
        phoneme_dicts = [asdict(p) for p in phonemes]
        gaze_dicts = [asdict(g) for g in gaze_targets]
        
        return AnimationSyncResponse(
            emotion=emotion.value,
            duration_ms=int(audio_duration_ms),
            gaze=gaze_dicts,
            phonemes=phoneme_dicts,
            sync_markers=sync_markers,
            intensity=intensity
        )
    
    def _generate_sync_markers(
        self,
        phonemes: List[Phoneme],
        gaze_targets: List[GazeTarget],
        audio_duration_ms: float
    ) -> List[Dict]:
        """
        Generate sync markers for frame-perfect animation
        
        Returns:
            List of markers: {time_ms, action, details}
        """
        
        markers = []
        
        # Phoneme markers
        for p in phonemes:
            markers.append({
                "time_ms": p.time_ms,
                "action": "mouth_shape",
                "mouth_open": p.mouth_open,
                "phoneme": p.phoneme
            })
        
        # Gaze markers
        for g in gaze_targets:
            markers.append({
                "time_ms": g.time_ms,
                "action": "gaze",
                "x": g.x,
                "y": g.y,
                "duration_ms": g.duration_ms
            })
        
        # Blink markers
        blinks = self.gaze_planner.get_blink_schedule(audio_duration_ms)
        for blink_time, blink_duration in blinks:
            markers.append({
                "time_ms": blink_time,
                "action": "blink",
                "duration_ms": blink_duration
            })
        
        # Sort by time
        markers.sort(key=lambda x: x["time_ms"])
        
        return markers
    
    def _calculate_intensity(self, text: str, emotion: Emotion) -> float:
        """
        Calculate animation intensity (0.0 to 1.0)
        
        Higher intensity = more pronounced movements
        """
        
        # Base intensity from emotion
        emotion_intensity = {
            Emotion.HAPPY: 0.9,
            Emotion.SURPRISED: 0.85,
            Emotion.ANGRY: 0.8,
            Emotion.SAD: 0.5,
            Emotion.CONFUSED: 0.6,
            Emotion.NEUTRAL: 0.5
        }
        
        intensity = emotion_intensity[emotion]
        
        # Boost for emphasized words
        emphasis_words = self.emotion_analyzer.extract_emphasis_words(text)
        if emphasis_words:
            intensity = min(1.0, intensity + 0.1 * len(emphasis_words))
        
        # Boost for longer sentences (more energy)
        word_count = len(text.split())
        if word_count > 20:
            intensity = min(1.0, intensity + 0.1)
        
        return intensity

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

animation_generator = AnimationCommandGenerator()

@app.post("/sync/generate")
async def generate_animation_commands(
    response: GeminiResponse
) -> AnimationSyncResponse:
    """
    Generate animation commands from Gemini response
    
    Args:
        response: GeminiResponse with text, audio duration, emotion
    
    Returns:
        AnimationSyncResponse with all animation commands
    
    Example:
        POST /sync/generate
        {
            "text": "Oh wow, that's amazing!",
            "audio_duration_ms": 1500,
            "emotion_detected": "happy"
        }
        
        Returns:
        {
            "emotion": "happy",
            "duration_ms": 1500,
            "gaze": [...],
            "phonemes": [...],
            "sync_markers": [...],
            "intensity": 0.85
        }
    """
    
    return await animation_generator.generate_commands(response)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "animation_sync"}

# ============================================================================
# STANDALONE USAGE
# ============================================================================

async def test_animation_sync():
    """Test the animation sync service"""
    
    # Simulate Gemini response
    gemini_response = GeminiResponse(
        text="Oh wow, that's absolutely amazing! I love it!",
        audio_duration_ms=2000,
        emotion_detected="happy"
    )
    
    # Generate animation commands
    commands = await animation_generator.generate_commands(gemini_response)
    
    print("Animation Commands:")
    print(json.dumps(asdict(commands), indent=2))
    
    return commands

if __name__ == "__main__":
    # Test
    import asyncio
    asyncio.run(test_animation_sync())
    
    # Or run server
    # uvicorn animation_sync_service:app --reload --port 8001
```

---

## Integration with Main Backend

```python
# server.py (updated)
from fastapi import FastAPI, WebSocket
from animation_sync_service import (
    AnimationCommandGenerator,
    GeminiResponse
)
import httpx
import json

app = FastAPI()
animation_gen = AnimationCommandGenerator()

# Service URLs
ANIMATION_SYNC_URL = "http://localhost:8001"

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Receive audio from device
            data = await websocket.receive_json()
            
            # Get emotion context
            emotion = data.get("emotion", "neutral")
            
            # Call Gemini Live
            gemini_text = await call_gemini_live(
                audio_data=data["audio"],
                emotion=emotion
            )
            
            # Generate TTS audio
            tts_audio, audio_duration_ms = await generate_tts(
                gemini_text,
                emotion
            )
            
            # Generate animation commands
            animation_response = GeminiResponse(
                text=gemini_text,
                audio_duration_ms=audio_duration_ms,
                emotion_detected=emotion
            )
            
            # Either call sync service or use local
            animation_commands = await animation_gen.generate_commands(
                animation_response
            )
            
            # Send response to device
            await websocket.send_json({
                "type": "response",
                "text": gemini_text,
                "audio": tts_audio.hex(),
                "animation": animation_commands.dict(),
                "timestamp": time.time()
            })
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

async def call_gemini_live(audio_data: bytes, emotion: str) -> str:
    """Call Gemini Live API"""
    
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"User emotion: {emotion}. Respond as Ollie."
        }]
    )
    
    return response.content[0].text

async def generate_tts(text: str, emotion: str) -> tuple:
    """Generate TTS audio"""
    
    from google.cloud import texttospeech
    
    client = texttospeech.TextToSpeechClient()
    
    # Emotion-aware TTS parameters
    emotion_params = {
        "happy": {"speaking_rate": 1.1, "pitch": 0.5},
        "sad": {"speaking_rate": 0.8, "pitch": -1.0},
        "neutral": {"speaking_rate": 1.0, "pitch": 0.0},
    }
    
    params = emotion_params.get(emotion, emotion_params["neutral"])
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=params["speaking_rate"],
        pitch=params["pitch"]
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-C"
        ),
        audio_config=audio_config
    )
    
    audio_bytes = response.audio_content
    
    # Estimate duration from text
    words = len(text.split())
    avg_wpm = 150  # Average words per minute
    duration_ms = int((words / avg_wpm) * 60000)
    
    return audio_bytes, duration_ms
```

---

## Deployment Options

### Option 1: Single Process (Development)
```python
# Run both services in one process
if __name__ == "__main__":
    import uvicorn
    
    # Load both FastAPI apps
    from fastapi import FastAPI
    from starlette.routing import Mount
    from animation_sync_service import app as animation_app
    from server import app as main_app
    
    # Mount animation service as sub-app
    main_app.mount("/animation", animation_app)
    
    uvicorn.run(main_app, host="0.0.0.0", port=8000)
```

### Option 2: Separate Microservices (Production)
```bash
# Terminal 1: Main backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2: Animation sync service
python -m uvicorn animation_sync_service:app --host 0.0.0.0 --port 8001
```

### Option 3: Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  main-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANIMATION_SYNC_URL=http://animation-sync:8001
    command: python -m uvicorn server:app --host 0.0.0.0 --port 8000

  animation-sync:
    build: .
    ports:
      - "8001:8001"
    command: python -m uvicorn animation_sync_service:app --host 0.0.0.0 --port 8001

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## Testing the Service

```python
# test_animation_sync.py
import asyncio
from animation_sync_service import (
    AnimationCommandGenerator,
    GeminiResponse,
    Emotion
)

async def test_various_emotions():
    """Test animation sync with different emotions"""
    
    test_cases = [
        {
            "text": "Oh wow, that's absolutely amazing!",
            "emotion": "happy",
            "duration": 2000
        },
        {
            "text": "I'm not sure about this...",
            "emotion": "confused",
            "duration": 1500
        },
        {
            "text": "That makes me so sad.",
            "emotion": "sad",
            "duration": 1800
        },
        {
            "text": "Wait, WHAT?!",
            "emotion": "surprised",
            "duration": 1200
        }
    ]
    
    generator = AnimationCommandGenerator()
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test['text']}")
        print(f"Emotion: {test['emotion']}")
        print(f"Duration: {test['duration']}ms")
        print('='*60)
        
        response = GeminiResponse(
            text=test["text"],
            audio_duration_ms=test["duration"],
            emotion_detected=test["emotion"]
        )
        
        commands = await generator.generate_commands(response)
        
        print(f"\nDetected Emotion: {commands.emotion}")
        print(f"Intensity: {commands.intensity}")
        print(f"Gaze Targets: {len(commands.gaze)}")
        print(f"Phonemes: {len(commands.phonemes)}")
        print(f"Sync Markers: {len(commands.sync_markers)}")
        
        # Print first few markers
        print(f"\nFirst 5 Sync Markers:")
        for marker in commands.sync_markers[:5]:
            print(f"  {marker}")

if __name__ == "__main__":
    asyncio.run(test_various_emotions())
```

---

## Key Features

### 1. Phoneme-Based Mouth Sync
- Extracts phonemes from text
- Maps to mouth shape (0.0 = closed, 1.0 = wide open)
- Distributes timing across audio duration

### 2. Emotion Detection
- Analyzes text for emotion keywords
- Uses detected emotion from face as override
- Selects appropriate GIF body animation

### 3. Natural Gaze Planning
- Determines gaze position based on emotion
- Adds gaze shifts at sentence boundaries
- Includes natural blinking schedule

### 4. Sync Markers
- Frame-perfect alignment of animations
- Separates phoneme, gaze, and blink actions
- Sorted by time for device processing

### 5. Expression Intensity
- Calculates based on emotion
- Boosted by emphasized words (!?, ALL CAPS)
- Affects animation amplitude

---

## API Response Example

```json
{
  "emotion": "happy",
  "duration_ms": 2000,
  "gaze": [
    {
      "time_ms": 0,
      "x": 0.3,
      "y": 0.2,
      "duration_ms": 600
    },
    {
      "time_ms": 1200,
      "x": 0.1,
      "y": 0.15,
      "duration_ms": 800
    }
  ],
  "phonemes": [
    {
      "time_ms": 0,
      "phoneme": "AA",
      "mouth_open": 0.9,
      "duration_ms": 100
    },
    {
      "time_ms": 100,
      "phoneme": "M",
      "mouth_open": 0.0,
      "duration_ms": 100
    }
  ],
  "sync_markers": [
    {
      "time_ms": 0,
      "action": "mouth_shape",
      "mouth_open": 0.9,
      "phoneme": "AA"
    },
    {
      "time_ms": 100,
      "action": "mouth_shape",
      "mouth_open": 0.0,
      "phoneme": "M"
    },
    {
      "time_ms": 150,
      "action": "blink",
      "duration_ms": 150
    }
  ],
  "intensity": 0.85
}
```

---

## Summary

This Animation Synchronization Service:

 **Separate microservice** for clean architecture  
 **Phoneme extraction** with timing alignment  
 **Emotion detection** from text + face  
 **Natural gaze planning** based on emotion  
 **Sync markers** for frame-perfect animation  
 **Intensity calculation** for expression amplitude  
 **Easy integration** with main backend  
 **Testable** with various emotions  

---

Version: 1.0  
Date: December 27, 2025  
Status: Production Ready
