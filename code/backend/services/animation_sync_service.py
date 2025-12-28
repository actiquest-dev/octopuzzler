"""
Animation Sync Service

Generates synchronized animation commands from TTS output.
Maps phonemes to mouth shapes and creates timing markers for:
- Mouth movements (phoneme-based)
- Eye gaze shifts
- Blinks
- Body animations (emotion-based)

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.0
"""

import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SyncMarker:
    """
    Animation sync marker
    
    Represents a single animation event at a specific time.
    
    Attributes:
        time_ms: Timestamp in milliseconds
        action: Action type (mouth_shape, gaze, blink, body_effect)
        params: Action-specific parameters
    """
    time_ms: int
    action: str
    params: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "time_ms": self.time_ms,
            "action": self.action,
            **self.params
        }


class PhonemeMapper:
    """
    Maps phonemes to mouth shapes
    
    Uses ARPABET phoneme set mapped to 9 mouth shapes:
    0 - SIL (silence)
    1 - AA (wide open)
    2 - EH (mid-open)
    3 - IH (narrow)
    4 - OW (round)
    5 - UH (mid-narrow)
    6 - M (lips together)
    7 - F (friction)
    8 - L (tongue visible)
    """
    
    # Phoneme to mouth shape mapping
    PHONEME_MAP = {
        # Silence
        "SIL": 0, "SP": 0, "": 0,
        
        # Wide open vowels
        "AA": 1, "AO": 1, "AH": 1,
        
        # Mid-open vowels
        "EH": 2, "AE": 2, "EY": 2,
        
        # Narrow vowels
        "IH": 3, "IY": 3, "IX": 3,
        
        # Round vowels
        "OW": 4, "UW": 4, "OY": 4,
        
        # Mid-narrow vowels
        "UH": 5, "AW": 5, "AY": 5,
        
        # Bilabials (lips together)
        "M": 6, "P": 6, "B": 6,
        
        # Fricatives
        "F": 7, "V": 7, "TH": 7, "DH": 7, "S": 7, "Z": 7, "SH": 7, "ZH": 7,
        
        # Liquids/nasals (tongue visible)
        "L": 8, "R": 8, "N": 8, "NG": 8,
        
        # Stops/affricates (quick closure, use narrow)
        "T": 3, "D": 3, "K": 3, "G": 3, "CH": 3, "JH": 3, "Y": 3, "W": 3, "HH": 3
    }
    
    @staticmethod
    def phoneme_to_mouth_shape(phoneme: str) -> int:
        """
        Convert phoneme to mouth shape ID
        
        Args:
            phoneme: ARPABET phoneme (e.g., "AA", "M", "SIL")
        
        Returns:
            Mouth shape ID (0-8)
        """
        # Remove stress markers (0, 1, 2)
        phoneme_clean = re.sub(r'[012]', '', phoneme.upper())
        
        return PhonemeMapper.PHONEME_MAP.get(phoneme_clean, 0)
    
    @staticmethod
    def text_to_phonemes(text: str) -> List[str]:
        """
        Simple text-to-phoneme conversion (rule-based)
        
        NOTE: This is a simplified version. For production, use:
        - g2p_en library for English
        - espeak-ng for multi-language
        - Or get phonemes directly from TTS model
        
        Args:
            text: Input text
        
        Returns:
            List of phonemes
        """
        # This is a placeholder - actual implementation would use
        # proper G2P (grapheme-to-phoneme) library
        
        # For now, just split into characters and map roughly
        # Real implementation should come from TTS model directly
        
        phonemes = []
        words = text.upper().split()
        
        for word in words:
            # Very rough approximation
            for char in word:
                if char in "AEIOU":
                    phonemes.append("AA")  # Vowel
                elif char in "BPMW":
                    phonemes.append("M")   # Bilabial
                elif char in "FVTHSZ":
                    phonemes.append("F")   # Fricative
                elif char in "LR":
                    phonemes.append("L")   # Liquid
                else:
                    phonemes.append("T")   # Stop
            
            phonemes.append("SIL")  # Word boundary
        
        return phonemes


class AnimationSyncService:
    """
    Animation Sync Service
    
    Generates synchronized animation commands from text and TTS timing.
    
    Features:
    ---------
    - Phoneme-to-mouth shape mapping
    - Automatic gaze shift generation
    - Natural blink timing
    - Emotion-driven body animations
    
    Usage:
    ------
    sync = AnimationSyncService()
    
    animation = await sync.generate_animation(
        text="Hello world!",
        word_timestamps=[
            {"word": "Hello", "start_ms": 0, "end_ms": 500},
            {"word": "world", "start_ms": 500, "end_ms": 1000}
        ],
        emotion="happy",
        intensity=0.8
    )
    
    # animation contains:
    # - body_animation: "happy"
    # - emotion_intensity: 0.8
    # - sync_markers: [...]
    """
    
    # Emotion to body animation mapping
    EMOTION_ANIMATIONS = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "neutral": "neutral",
        "surprised": "surprised",
        "curious": "curious",
        "fearful": "fearful",
        "disgusted": "neutral"  # No specific animation
    }
    
    def __init__(
        self,
        blink_interval_ms: int = 3000,
        gaze_shift_interval_ms: int = 2000
    ):
        """
        Initialize animation sync service
        
        Args:
            blink_interval_ms: Average time between blinks
            gaze_shift_interval_ms: Average time between gaze shifts
        """
        logger.info("Initializing Animation Sync Service...")
        
        self.blink_interval_ms = blink_interval_ms
        self.gaze_shift_interval_ms = gaze_shift_interval_ms
        
        self.phoneme_mapper = PhonemeMapper()
        
        logger.info("✓ Animation Sync Service ready")
    
    async def generate_animation(
        self,
        text: str,
        word_timestamps: List[Dict],
        emotion: str = "neutral",
        intensity: float = 1.0,
        phonemes: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate animation from text and timing
        
        Args:
            text: Spoken text
            word_timestamps: Word timing information
                [
                    {"word": "Hello", "start_ms": 0, "end_ms": 500},
                    {"word": "world", "start_ms": 500, "end_ms": 1000}
                ]
            emotion: Emotion for body animation
            intensity: Emotion intensity (0.0 to 1.0)
            phonemes: Optional phoneme timing (if available from TTS)
                [
                    {"phoneme": "HH", "start_ms": 0, "end_ms": 50},
                    {"phoneme": "AH", "start_ms": 50, "end_ms": 150},
                    ...
                ]
        
        Returns:
            {
                "body_animation": str,
                "emotion_intensity": float,
                "sync_markers": List[Dict]
            }
        """
        logger.info(f"Generating animation for: '{text}' (emotion={emotion})")
        
        sync_markers = []
        
        # 1. Generate mouth shape markers
        if phonemes:
            # Use phonemes from TTS if available (preferred)
            mouth_markers = self._generate_mouth_markers_from_phonemes(phonemes)
        else:
            # Fallback: generate from word timestamps
            mouth_markers = self._generate_mouth_markers_from_words(
                text, word_timestamps
            )
        
        sync_markers.extend(mouth_markers)
        
        # 2. Generate gaze shift markers
        if word_timestamps:
            total_duration_ms = word_timestamps[-1]["end_ms"]
            gaze_markers = self._generate_gaze_markers(total_duration_ms)
            sync_markers.extend(gaze_markers)
        
        # 3. Generate blink markers
        if word_timestamps:
            blink_markers = self._generate_blink_markers(total_duration_ms)
            sync_markers.extend(blink_markers)
        
        # 4. Generate emotion-specific effects (optional)
        if emotion != "neutral" and intensity > 0.5:
            effect_markers = self._generate_emotion_effects(
                emotion, intensity, total_duration_ms
            )
            sync_markers.extend(effect_markers)
        
        # Sort markers by time
        sync_markers.sort(key=lambda m: m.time_ms)
        
        # Get body animation
        body_animation = self.EMOTION_ANIMATIONS.get(emotion, "neutral")
        
        result = {
            "body_animation": body_animation,
            "emotion_intensity": intensity,
            "sync_markers": [marker.to_dict() for marker in sync_markers]
        }
        
        logger.info(
            f"Generated {len(sync_markers)} sync markers "
            f"(animation={body_animation})"
        )
        
        return result
    
    def _generate_mouth_markers_from_phonemes(
        self,
        phonemes: List[Dict]
    ) -> List[SyncMarker]:
        """
        Generate mouth shape markers from phoneme timing
        
        Args:
            phonemes: List of phoneme dicts with timing
        
        Returns:
            List of SyncMarker objects
        """
        markers = []
        
        for phoneme_data in phonemes:
            phoneme = phoneme_data["phoneme"]
            start_ms = phoneme_data["start_ms"]
            
            mouth_shape = self.phoneme_mapper.phoneme_to_mouth_shape(phoneme)
            
            marker = SyncMarker(
                time_ms=start_ms,
                action="mouth_shape",
                params={
                    "mouth_shape": mouth_shape,
                    "phoneme": phoneme
                }
            )
            
            markers.append(marker)
        
        logger.debug(f"Generated {len(markers)} mouth markers from phonemes")
        return markers
    
    def _generate_mouth_markers_from_words(
        self,
        text: str,
        word_timestamps: List[Dict]
    ) -> List[SyncMarker]:
        """
        Generate mouth shape markers from word timing (fallback)
        
        This is less accurate than phoneme-based but works when
        phoneme timing is not available.
        
        Args:
            text: Full text
            word_timestamps: Word timing
        
        Returns:
            List of SyncMarker objects
        """
        markers = []
        
        for word_data in word_timestamps:
            word = word_data["word"]
            start_ms = word_data["start_ms"]
            end_ms = word_data["end_ms"]
            duration_ms = end_ms - start_ms
            
            # Get phonemes for word (rough approximation)
            phonemes = self.phoneme_mapper.text_to_phonemes(word)
            
            # Distribute phonemes evenly across word duration
            if len(phonemes) > 0:
                phoneme_duration = duration_ms / len(phonemes)
                
                for i, phoneme in enumerate(phonemes):
                    time_ms = int(start_ms + (i * phoneme_duration))
                    mouth_shape = self.phoneme_mapper.phoneme_to_mouth_shape(phoneme)
                    
                    marker = SyncMarker(
                        time_ms=time_ms,
                        action="mouth_shape",
                        params={
                            "mouth_shape": mouth_shape,
                            "phoneme": phoneme
                        }
                    )
                    
                    markers.append(marker)
        
        logger.debug(f"Generated {len(markers)} mouth markers from words")
        return markers
    
    def _generate_gaze_markers(
        self,
        total_duration_ms: int
    ) -> List[SyncMarker]:
        """
        Generate natural gaze shift markers
        
        Gaze shifts occur every 2-3 seconds with slight randomization.
        
        Args:
            total_duration_ms: Total animation duration
        
        Returns:
            List of SyncMarker objects
        """
        markers = []
        
        current_time = 0
        
        while current_time < total_duration_ms:
            # Random gaze direction
            gaze_x = random.uniform(-0.3, 0.3)
            gaze_y = random.uniform(-0.2, 0.1)  # Slightly prefer looking down
            
            marker = SyncMarker(
                time_ms=current_time,
                action="gaze",
                params={
                    "gaze_x": round(gaze_x, 2),
                    "gaze_y": round(gaze_y, 2)
                }
            )
            
            markers.append(marker)
            
            # Next gaze shift (with randomization)
            interval = self.gaze_shift_interval_ms + random.randint(-500, 500)
            current_time += interval
        
        logger.debug(f"Generated {len(markers)} gaze markers")
        return markers
    
    def _generate_blink_markers(
        self,
        total_duration_ms: int
    ) -> List[SyncMarker]:
        """
        Generate natural blink markers
        
        Blinks occur every 3-4 seconds with slight randomization.
        
        Args:
            total_duration_ms: Total animation duration
        
        Returns:
            List of SyncMarker objects
        """
        markers = []
        
        current_time = random.randint(500, 1000)  # First blink offset
        
        while current_time < total_duration_ms:
            # Blink duration (100-200ms)
            blink_duration = random.randint(100, 200)
            
            marker = SyncMarker(
                time_ms=current_time,
                action="blink",
                params={
                    "blink_duration_ms": blink_duration
                }
            )
            
            markers.append(marker)
            
            # Next blink (with randomization)
            interval = self.blink_interval_ms + random.randint(-1000, 1000)
            current_time += interval
        
        logger.debug(f"Generated {len(markers)} blink markers")
        return markers
    
    def _generate_emotion_effects(
        self,
        emotion: str,
        intensity: float,
        total_duration_ms: int
    ) -> List[SyncMarker]:
        """
        Generate emotion-specific effects
        
        Examples:
        - Happy: bounce at start
        - Surprised: jump at start
        - Angry: shake throughout
        
        Args:
            emotion: Emotion type
            intensity: Emotion intensity
            total_duration_ms: Total duration
        
        Returns:
            List of SyncMarker objects
        """
        markers = []
        
        if emotion == "happy" and intensity > 0.7:
            # Bounce effect at start
            marker = SyncMarker(
                time_ms=0,
                action="body_effect",
                params={
                    "effect": "bounce",
                    "intensity": intensity
                }
            )
            markers.append(marker)
        
        elif emotion == "surprised":
            # Jump effect at start
            marker = SyncMarker(
                time_ms=0,
                action="body_effect",
                params={
                    "effect": "jump",
                    "intensity": intensity
                }
            )
            markers.append(marker)
        
        elif emotion == "angry" and intensity > 0.6:
            # Shake effect throughout
            num_shakes = int((total_duration_ms / 1000) * intensity)
            for i in range(num_shakes):
                time_ms = i * 1000
                marker = SyncMarker(
                    time_ms=time_ms,
                    action="body_effect",
                    params={
                        "effect": "shake",
                        "intensity": intensity
                    }
                )
                markers.append(marker)
        
        logger.debug(f"Generated {len(markers)} emotion effect markers")
        return markers
    
    def get_phoneme_info(self) -> Dict:
        """Get phoneme mapping information"""
        return {
            "total_phonemes": len(self.phoneme_mapper.PHONEME_MAP),
            "mouth_shapes": 9,
            "mapping": self.phoneme_mapper.PHONEME_MAP
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_animation_sync():
        """Test animation sync service"""
        
        # Initialize service
        sync = AnimationSyncService()
        
        print("\n=== Test 1: Simple animation ===")
        
        text = "Hello world"
        word_timestamps = [
            {"word": "Hello", "start_ms": 0, "end_ms": 500},
            {"word": "world", "start_ms": 500, "end_ms": 1000}
        ]
        
        animation = await sync.generate_animation(
            text=text,
            word_timestamps=word_timestamps,
            emotion="happy",
            intensity=0.8
        )
        
        print(f"Body animation: {animation['body_animation']}")
        print(f"Emotion intensity: {animation['emotion_intensity']}")
        print(f"Total markers: {len(animation['sync_markers'])}")
        
        # Show first few markers
        print("\nFirst 5 markers:")
        for marker in animation['sync_markers'][:5]:
            print(f"  {marker['time_ms']}ms: {marker['action']} - {marker}")
        
        print("\n=== Test 2: With phoneme timing ===")
        
        # Simulate phoneme timing from TTS
        phonemes = [
            {"phoneme": "HH", "start_ms": 0, "end_ms": 50},
            {"phoneme": "AH", "start_ms": 50, "end_ms": 150},
            {"phoneme": "L", "start_ms": 150, "end_ms": 250},
            {"phoneme": "OW", "start_ms": 250, "end_ms": 400},
            {"phoneme": "SIL", "start_ms": 400, "end_ms": 500},
            {"phoneme": "W", "start_ms": 500, "end_ms": 550},
            {"phoneme": "ER", "start_ms": 550, "end_ms": 700},
            {"phoneme": "L", "start_ms": 700, "end_ms": 800},
            {"phoneme": "D", "start_ms": 800, "end_ms": 1000}
        ]
        
        animation = await sync.generate_animation(
            text=text,
            word_timestamps=word_timestamps,
            emotion="neutral",
            intensity=1.0,
            phonemes=phonemes
        )
        
        print(f"Total markers: {len(animation['sync_markers'])}")
        
        # Count marker types
        marker_types = {}
        for marker in animation['sync_markers']:
            action = marker['action']
            marker_types[action] = marker_types.get(action, 0) + 1
        
        print("\nMarker types:")
        for action, count in marker_types.items():
            print(f"  {action}: {count}")
        
        print("\n=== Test 3: Different emotions ===")
        
        emotions = ["happy", "sad", "angry", "surprised", "curious"]
        
        for emotion in emotions:
            animation = await sync.generate_animation(
                text="Test message",
                word_timestamps=[
                    {"word": "Test", "start_ms": 0, "end_ms": 300},
                    {"word": "message", "start_ms": 300, "end_ms": 800}
                ],
                emotion=emotion,
                intensity=0.9
            )
            
            print(f"{emotion}: {animation['body_animation']}, "
                  f"{len(animation['sync_markers'])} markers")
        
        print("\n=== Test 4: Phoneme info ===")
        info = sync.get_phoneme_info()
        print(f"Total phonemes: {info['total_phonemes']}")
        print(f"Mouth shapes: {info['mouth_shapes']}")
        print(f"\nSample mappings:")
        for phoneme in ["AA", "M", "F", "L", "SIL"]:
            shape = info['mapping'].get(phoneme, 0)
            print(f"  {phoneme} → shape {shape}")
    
    asyncio.run(test_animation_sync())