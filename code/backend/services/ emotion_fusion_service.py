"""
Emotion Fusion Service

Fuses audio and vision emotion signals into unified emotion state.
Uses Voice Activity Detection (VAD) to determine speaking state and
apply appropriate weighting.

Strategy:
- Speaking: 70% audio + 30% vision
- Silent: 100% vision
- Temporal smoothing: 30% previous + 70% current

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.0
"""

import numpy as np
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import deque
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmotionSignal:
    """
    Emotion signal from a single modality
    
    Attributes:
        emotion: Emotion label (happy, sad, angry, neutral, surprised, fearful, disgusted)
        confidence: Confidence score (0.0 to 1.0)
        intensity: Emotion intensity (0.0 to 1.0)
        timestamp: Unix timestamp
        source: Source modality (audio, vision)
    """
    emotion: str
    confidence: float
    intensity: float = 1.0
    timestamp: float = 0.0
    source: str = ""
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class FusedEmotion:
    """
    Fused emotion result
    
    Attributes:
        emotion: Final emotion label
        confidence: Final confidence score
        intensity: Final intensity
        audio_weight: Weight applied to audio signal
        vision_weight: Weight applied to vision signal
        is_speaking: Whether user is speaking
        sources: Original signals
        timestamp: Fusion timestamp
    """
    emotion: str
    confidence: float
    intensity: float
    audio_weight: float
    vision_weight: float
    is_speaking: bool
    sources: Dict[str, EmotionSignal]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "audio_weight": self.audio_weight,
            "vision_weight": self.vision_weight,
            "is_speaking": self.is_speaking,
            "sources": {
                source: {
                    "emotion": signal.emotion,
                    "confidence": signal.confidence,
                    "intensity": signal.intensity
                }
                for source, signal in self.sources.items()
            },
            "timestamp": self.timestamp
        }


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection (VAD)
    
    Detects whether audio contains speech based on energy threshold.
    """
    
    def __init__(
        self,
        threshold: float = 0.02,
        sample_rate: int = 16000,
        frame_length: int = 1600  # 100ms at 16kHz
    ):
        """
        Initialize VAD
        
        Args:
            threshold: Energy threshold (0.0 to 1.0)
            sample_rate: Audio sample rate
            frame_length: Frame length in samples
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        
        logger.info(f"VAD initialized: threshold={threshold}")
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech
        
        Args:
            audio_chunk: Audio samples (normalized to [-1, 1])
        
        Returns:
            True if speech detected, False otherwise
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Compare to threshold
        is_active = rms > self.threshold
        
        logger.debug(f"VAD: RMS={rms:.4f}, threshold={self.threshold:.4f}, active={is_active}")
        
        return is_active


class EmotionFusionService:
    """
    Emotion Fusion Service
    
    Combines audio and vision emotion signals with context-aware weighting.
    
    Features:
    ---------
    - Dynamic weighting based on speaking state
    - Temporal smoothing for stability
    - Confidence-aware fusion
    - Emotion history tracking
    
    Usage:
    ------
    fusion = EmotionFusionService()
    
    result = await fusion.fuse(
        audio_emotion={"emotion": "happy", "confidence": 0.85},
        vision_emotion={"emotion": "neutral", "intensity": 0.65},
        audio_chunk=audio_samples
    )
    
    print(result.emotion)  # "happy"
    print(result.confidence)  # 0.76
    """
    
    # Supported emotions (7 basic emotions)
    EMOTIONS = [
        "happy",
        "sad",
        "angry",
        "neutral",
        "surprised",
        "fearful",
        "disgusted"
    ]
    
    def __init__(
        self,
        speaking_weights: Optional[Dict[str, float]] = None,
        silent_weights: Optional[Dict[str, float]] = None,
        smoothing_factor: float = 0.3,
        vad_threshold: float = 0.02,
        history_size: int = 10
    ):
        """
        Initialize emotion fusion service
        
        Args:
            speaking_weights: Weights when speaking {"audio": 0.7, "vision": 0.3}
            silent_weights: Weights when silent {"audio": 0.0, "vision": 1.0}
            smoothing_factor: Temporal smoothing (0.0 = no smoothing, 1.0 = max smoothing)
            vad_threshold: Voice activity detection threshold
            history_size: Number of past emotions to keep
        """
        logger.info("Initializing Emotion Fusion Service...")
        
        # Fusion weights
        self.speaking_weights = speaking_weights or {"audio": 0.7, "vision": 0.3}
        self.silent_weights = silent_weights or {"audio": 0.0, "vision": 1.0}
        
        # Validate weights
        self._validate_weights(self.speaking_weights)
        self._validate_weights(self.silent_weights)
        
        # Smoothing
        self.smoothing_factor = smoothing_factor
        
        # VAD
        self.vad = VoiceActivityDetector(threshold=vad_threshold)
        
        # History
        self.history_size = history_size
        self.emotion_history: deque = deque(maxlen=history_size)
        
        # Previous emotion for smoothing
        self.previous_emotion: Optional[FusedEmotion] = None
        
        logger.info(f"Emotion Fusion Service ready:")
        logger.info(f"  Speaking weights: {self.speaking_weights}")
        logger.info(f"  Silent weights: {self.silent_weights}")
        logger.info(f"  Smoothing factor: {self.smoothing_factor}")
        logger.info(f"  VAD threshold: {vad_threshold}")
    
    def _validate_weights(self, weights: Dict[str, float]):
        """Validate that weights sum to 1.0"""
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    async def fuse(
        self,
        audio_emotion: Optional[Dict] = None,
        vision_emotion: Optional[Dict] = None,
        audio_chunk: Optional[np.ndarray] = None
    ) -> FusedEmotion:
        """
        Fuse audio and vision emotion signals
        
        Args:
            audio_emotion: Audio emotion signal
                {
                    "emotion": str,
                    "confidence": float (0.0-1.0)
                }
            vision_emotion: Vision emotion signal
                {
                    "emotion": str,
                    "intensity": float (0.0-1.0)
                }
            audio_chunk: Audio samples for VAD (optional)
        
        Returns:
            FusedEmotion object
        
        Raises:
            ValueError: If both signals are None
        """
        if audio_emotion is None and vision_emotion is None:
            raise ValueError("At least one emotion signal required")
        
        # Convert to EmotionSignal objects
        audio_signal = self._parse_audio_emotion(audio_emotion)
        vision_signal = self._parse_vision_emotion(vision_emotion)
        
        # Determine speaking state (VAD)
        is_speaking = False
        if audio_chunk is not None:
            is_speaking = self.vad.is_speech(audio_chunk)
        
        # Select weights based on speaking state
        if is_speaking:
            weights = self.speaking_weights
            logger.debug("User speaking → audio dominant")
        else:
            weights = self.silent_weights
            logger.debug("User silent → vision dominant")
        
        # Fuse emotions
        fused_emotion, fused_confidence = self._fuse_signals(
            audio_signal, vision_signal, weights
        )
        
        # Calculate intensity
        fused_intensity = self._fuse_intensity(
            audio_signal, vision_signal, weights
        )
        
        # Apply temporal smoothing
        if self.previous_emotion is not None:
            fused_emotion, fused_confidence = self._apply_smoothing(
                current_emotion=fused_emotion,
                current_confidence=fused_confidence,
                previous_emotion=self.previous_emotion.emotion,
                previous_confidence=self.previous_emotion.confidence
            )
        
        # Create result
        result = FusedEmotion(
            emotion=fused_emotion,
            confidence=fused_confidence,
            intensity=fused_intensity,
            audio_weight=weights["audio"],
            vision_weight=weights["vision"],
            is_speaking=is_speaking,
            sources={
                "audio": audio_signal,
                "vision": vision_signal
            }
        )
        
        # Update history
        self.emotion_history.append(result)
        self.previous_emotion = result
        
        logger.info(
            f"Fused emotion: {result.emotion} "
            f"(confidence={result.confidence:.2f}, "
            f"intensity={result.intensity:.2f}, "
            f"speaking={is_speaking})"
        )
        
        return result
    
    def _parse_audio_emotion(self, audio_emotion: Optional[Dict]) -> EmotionSignal:
        """Parse audio emotion dict to EmotionSignal"""
        if audio_emotion is None:
            return EmotionSignal(
                emotion="neutral",
                confidence=0.0,
                intensity=0.0,
                source="audio"
            )
        
        return EmotionSignal(
            emotion=audio_emotion.get("emotion", "neutral"),
            confidence=audio_emotion.get("confidence", 0.5),
            intensity=audio_emotion.get("intensity", 1.0),
            source="audio"
        )
    
    def _parse_vision_emotion(self, vision_emotion: Optional[Dict]) -> EmotionSignal:
        """Parse vision emotion dict to EmotionSignal"""
        if vision_emotion is None:
            return EmotionSignal(
                emotion="neutral",
                confidence=0.0,
                intensity=0.0,
                source="vision"
            )
        
        return EmotionSignal(
            emotion=vision_emotion.get("emotion", "neutral"),
            confidence=vision_emotion.get("confidence", 0.5),
            intensity=vision_emotion.get("intensity", 1.0),
            source="vision"
        )
    
    def _fuse_signals(
        self,
        audio_signal: EmotionSignal,
        vision_signal: EmotionSignal,
        weights: Dict[str, float]
    ) -> tuple[str, float]:
        """
        Fuse emotion signals with weighting
        
        Returns:
            (emotion, confidence)
        """
        # Calculate weighted confidences
        audio_weighted = audio_signal.confidence * weights["audio"]
        vision_weighted = vision_signal.confidence * weights["vision"]
        
        # If same emotion, combine confidences
        if audio_signal.emotion == vision_signal.emotion:
            combined_confidence = audio_weighted + vision_weighted
            return audio_signal.emotion, combined_confidence
        
        # Different emotions: choose higher weighted confidence
        if audio_weighted > vision_weighted:
            return audio_signal.emotion, audio_weighted
        else:
            return vision_signal.emotion, vision_weighted
    
    def _fuse_intensity(
        self,
        audio_signal: EmotionSignal,
        vision_signal: EmotionSignal,
        weights: Dict[str, float]
    ) -> float:
        """Fuse intensity with weighting"""
        return (
            audio_signal.intensity * weights["audio"] +
            vision_signal.intensity * weights["vision"]
        )
    
    def _apply_smoothing(
        self,
        current_emotion: str,
        current_confidence: float,
        previous_emotion: str,
        previous_confidence: float
    ) -> tuple[str, float]:
        """
        Apply temporal smoothing
        
        Smoothing reduces sudden emotion changes by blending
        with previous state.
        
        Returns:
            (smoothed_emotion, smoothed_confidence)
        """
        # Blend confidences
        smoothed_confidence = (
            current_confidence * (1 - self.smoothing_factor) +
            previous_confidence * self.smoothing_factor
        )
        
        # If emotion changed, use current if confidence is high enough
        if current_emotion != previous_emotion:
            # Require higher confidence to change emotion
            if current_confidence > 0.6:
                return current_emotion, smoothed_confidence
            else:
                # Keep previous emotion, but with smoothed confidence
                return previous_emotion, smoothed_confidence
        else:
            # Same emotion, just smooth confidence
            return current_emotion, smoothed_confidence
    
    def get_emotion_history(self, n: int = 5) -> List[Dict]:
        """
        Get recent emotion history
        
        Args:
            n: Number of recent emotions to return
        
        Returns:
            List of emotion dicts (most recent first)
        """
        history = list(self.emotion_history)[-n:]
        return [emotion.to_dict() for emotion in reversed(history)]
    
    def get_dominant_emotion(self, window: int = 5) -> Optional[str]:
        """
        Get dominant emotion over recent window
        
        Args:
            window: Number of recent emotions to consider
        
        Returns:
            Most frequent emotion or None if history empty
        """
        if not self.emotion_history:
            return None
        
        recent = list(self.emotion_history)[-window:]
        emotions = [e.emotion for e in recent]
        
        # Count occurrences
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most common
        return max(emotion_counts, key=emotion_counts.get)
    
    def reset(self):
        """Reset fusion state (clear history)"""
        self.emotion_history.clear()
        self.previous_emotion = None
        logger.info("Emotion fusion state reset")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_emotion_fusion():
        """Test emotion fusion service"""
        
        # Initialize service
        fusion = EmotionFusionService()
        
        print("\n=== Test 1: User speaking (happy audio, neutral vision) ===")
        
        # Simulate high-energy audio (speaking)
        audio_chunk = np.random.randn(1600) * 0.1  # High energy
        
        result = await fusion.fuse(
            audio_emotion={"emotion": "happy", "confidence": 0.85},
            vision_emotion={"emotion": "neutral", "intensity": 0.65},
            audio_chunk=audio_chunk
        )
        
        print(f"Fused emotion: {result.emotion}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Is speaking: {result.is_speaking}")
        print(f"Weights: audio={result.audio_weight}, vision={result.vision_weight}")
        
        print("\n=== Test 2: User silent (neutral audio, sad vision) ===")
        
        # Simulate low-energy audio (silent)
        audio_chunk = np.random.randn(1600) * 0.001  # Low energy
        
        result = await fusion.fuse(
            audio_emotion={"emotion": "neutral", "confidence": 0.5},
            vision_emotion={"emotion": "sad", "intensity": 0.70},
            audio_chunk=audio_chunk
        )
        
        print(f"Fused emotion: {result.emotion}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Is speaking: {result.is_speaking}")
        print(f"Weights: audio={result.audio_weight}, vision={result.vision_weight}")
        
        print("\n=== Test 3: Temporal smoothing (emotion change) ===")
        
        # First frame: happy
        audio_chunk = np.random.randn(1600) * 0.1
        result1 = await fusion.fuse(
            audio_emotion={"emotion": "happy", "confidence": 0.8},
            vision_emotion={"emotion": "happy", "intensity": 0.7},
            audio_chunk=audio_chunk
        )
        print(f"Frame 1: {result1.emotion} (conf={result1.confidence:.2f})")
        
        # Second frame: sudden change to sad (should be smoothed)
        result2 = await fusion.fuse(
            audio_emotion={"emotion": "sad", "confidence": 0.55},
            vision_emotion={"emotion": "sad", "intensity": 0.6},
            audio_chunk=audio_chunk
        )
        print(f"Frame 2: {result2.emotion} (conf={result2.confidence:.2f})")
        print(f"  → Low confidence, kept previous emotion")
        
        # Third frame: strong sad signal (should change)
        result3 = await fusion.fuse(
            audio_emotion={"emotion": "sad", "confidence": 0.9},
            vision_emotion={"emotion": "sad", "intensity": 0.8},
            audio_chunk=audio_chunk
        )
        print(f"Frame 3: {result3.emotion} (conf={result3.confidence:.2f})")
        print(f"  → High confidence, changed emotion")
        
        print("\n=== Test 4: Emotion history ===")
        history = fusion.get_emotion_history(n=5)
        print(f"Recent emotions: {[h['emotion'] for h in history]}")
        
        dominant = fusion.get_dominant_emotion(window=5)
        print(f"Dominant emotion: {dominant}")
        
        print("\n=== Test 5: Edge cases ===")
        
        # Only audio
        result = await fusion.fuse(
            audio_emotion={"emotion": "angry", "confidence": 0.75},
            vision_emotion=None,
            audio_chunk=np.random.randn(1600) * 0.1
        )
        print(f"Audio only: {result.emotion} (conf={result.confidence:.2f})")
        
        # Only vision
        result = await fusion.fuse(
            audio_emotion=None,
            vision_emotion={"emotion": "surprised", "intensity": 0.8},
            audio_chunk=np.random.randn(1600) * 0.001
        )
        print(f"Vision only: {result.emotion} (conf={result.confidence:.2f})")
    
    asyncio.run(test_emotion_fusion())