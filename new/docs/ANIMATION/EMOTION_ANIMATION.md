# Emotion Animation

## Overview

Map detected emotions to appropriate GIF animations and visual expressions.

```
Detected emotion (happy, sad, angry, neutral, surprised)
    ↓
Select appropriate body GIF
    ↓
Adjust eye shape, gaze, blink rate
    ↓
Render with emotion-specific colors/filters
```

## 5 Emotions → 5 GIFs

```python
EMOTION_TO_GIF = {
    "happy": "octopus_happy.gif",
    "sad": "octopus_sad.gif",
    "angry": "octopus_angry.gif",
    "neutral": "octopus_neutral.gif",
    "surprised": "octopus_surprised.gif"
}
```

## Happy Emotion

```python
HAPPY_CONFIG = {
    "gif": "octopus_happy.gif",
    "colors": {
        "primary": (255, 200, 0),      # Warm yellow
        "secondary": (255, 150, 0),    # Orange
        "background": (200, 255, 200)  # Soft green
    },
    "eyes": {
        "shape": "squinting",           # Squinting eyes
        "width": 0.7,                   # Narrower (happy expression)
        "color": (50, 50, 50),          # Dark
        "shine": True                   # Add shine/sparkle
    },
    "gaze": {
        "x": 0.2,                       # Slight right look
        "y": -0.3,                      # Upward (optimistic)
        "duration_ms": 500
    },
    "blink": {
        "rate": 0.5,                    # Hz (more frequent)
        "duration_ms": 100
    },
    "mouth": {
        "curve": 0.8,                   # Smile curve
        "width": 1.0                    # Wide smile
    },
    "animation_speed": 1.2,             # Faster movement
    "bounce": True,                     # Bouncy movement
    "filter": "brightness(1.1)"         # Slightly brighter
}
```

### Happy Animation Frames

```
Frame sequence:
0    - Relaxed expression
10   - Light bounce up
20   - Bounce down
30   - Small wiggle left
40   - Small wiggle right
50   - Back to start

Play continuously with happy GIF
```

## Sad Emotion

```python
SAD_CONFIG = {
    "gif": "octopus_sad.gif",
    "colors": {
        "primary": (100, 150, 200),     # Cool blue
        "secondary": (80, 120, 180),    # Darker blue
        "background": (220, 220, 230)   # Cold grey-blue
    },
    "eyes": {
        "shape": "drooping",            # Drooping eyelids
        "width": 0.8,                   # Normal width
        "droop": 0.6,                   # How much drooping
        "color": (50, 50, 50),
        "shine": False                  # No sparkle
    },
    "gaze": {
        "x": -0.3,                      # Look away
        "y": 0.3,                       # Downward
        "duration_ms": 1000
    },
    "blink": {
        "rate": 0.2,                    # Hz (less frequent)
        "duration_ms": 150
    },
    "mouth": {
        "curve": -0.7,                  # Frown
        "width": 0.8                    # Slightly narrower
    },
    "animation_speed": 0.7,             # Slower movement
    "bounce": False,                    # No bouncy movement
    "filter": "brightness(0.9)"         # Slightly darker
}
```

### Sad Animation Frames

```
Frame sequence:
0    - Sad expression
15   - Slight slump left
30   - Slight slump right
45   - Return to sad

Play slowly with sad GIF (melancholic)
```

## Angry Emotion

```python
ANGRY_CONFIG = {
    "gif": "octopus_angry.gif",
    "colors": {
        "primary": (255, 80, 80),       # Bright red
        "secondary": (200, 50, 50),     # Dark red
        "background": (255, 220, 220)   # Light red
    },
    "eyes": {
        "shape": "narrowed",            # Narrowed/angry eyes
        "width": 0.4,                   # Very narrow
        "angle": -15,                   # Angled down toward center
        "color": (200, 0, 0),           # Reddish
        "shine": False
    },
    "gaze": {
        "x": 0.0,                       # Direct look
        "y": -0.5,                      # Intense stare
        "duration_ms": 800
    },
    "blink": {
        "rate": 0.3,                    # Hz (infrequent)
        "duration_ms": 80               # Quick blinks
    },
    "mouth": {
        "curve": -0.9,                  # Angry frown
        "width": 0.6,                   # Narrow mouth
        "clench": True                  # Clenched jaw
    },
    "animation_speed": 1.5,             # Fast aggressive movement
    "shake": True,                      # Shake/vibrate effect
    "filter": "brightness(0.85) saturate(1.3)"  # Darker, more saturated
}
```

### Angry Animation Frames

```
Frame sequence:
0    - Angry expression
8    - Aggressive pose left
16   - Back to angry
24   - Aggressive pose right
32   - Back to angry

Play in aggressive pattern with shaking
```

## Neutral Emotion

```python
NEUTRAL_CONFIG = {
    "gif": "octopus_neutral.gif",
    "colors": {
        "primary": (200, 200, 200),     # Grey
        "secondary": (150, 150, 150),   # Dark grey
        "background": (240, 240, 240)   # Light grey
    },
    "eyes": {
        "shape": "normal",              # Normal open eyes
        "width": 1.0,                   # Normal width
        "color": (50, 50, 50),
        "shine": True
    },
    "gaze": {
        "x": 0.0,                       # Centered
        "y": 0.0,                       # Centered
        "duration_ms": 600
    },
    "blink": {
        "rate": 0.3,                    # Hz (normal)
        "duration_ms": 120              # Normal duration
    },
    "mouth": {
        "curve": 0.0,                   # Neutral mouth
        "width": 0.9                    # Normal width
    },
    "animation_speed": 1.0,             # Normal speed
    "bounce": False,
    "filter": None                      # No filter
}
```

### Neutral Animation Frames

```
Frame sequence:
0    - Neutral expression
20   - Subtle idle animation
40   - Back to neutral

Smooth, calm movement
```

## Surprised Emotion

```python
SURPRISED_CONFIG = {
    "gif": "octopus_surprised.gif",
    "colors": {
        "primary": (200, 100, 255),     # Purple
        "secondary": (150, 50, 200),    # Dark purple
        "background": (240, 200, 255)   # Light purple
    },
    "eyes": {
        "shape": "wide",                # Wide open eyes
        "width": 1.3,                   # Wider than normal
        "height": 1.2,                  # Taller than normal
        "color": (50, 50, 50),
        "shine": True,
        "pupil_size": 0.6               # Larger pupils
    },
    "gaze": {
        "x": 0.0,                       # Centered focus
        "y": -0.4,                      # Forward/upward
        "duration_ms": 400
    },
    "blink": {
        "rate": 0.4,                    # Hz (slightly more)
        "duration_ms": 110
    },
    "mouth": {
        "curve": 0.5,                   # Slight smile/O shape
        "width": 0.7,                   # Slightly open
        "shape": "O"                    # Round "O" for surprise
    },
    "animation_speed": 1.3,             # Quick reaction
    "jump": True,                       # Jump/startled effect
    "filter": "brightness(1.15)"        # Brighter
}
```

### Surprised Animation Frames

```
Frame sequence:
0    - Jump up
5    - Wide eyed expression
15   - Settle down
30   - Return to surprised

Quick startle effect, then settle
```

## Implementation

```python
# emotion_animation.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class EmotionType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    SURPRISED = "surprised"

@dataclass
class EyeConfig:
    """Eye configuration"""
    shape: str
    width: float
    height: float = 1.0
    color: tuple
    shine: bool = True
    droop: float = 0.0
    angle: float = 0.0
    pupil_size: float = 1.0

@dataclass
class MouthConfig:
    """Mouth configuration"""
    curve: float        # -1 (frown) to 1 (smile)
    width: float        # 0 (closed) to 1 (open)
    shape: str = "normal"
    clench: bool = False

@dataclass
class AnimationConfig:
    """Complete emotion animation config"""
    emotion: EmotionType
    gif: str
    colors: Dict[str, tuple]
    eyes: EyeConfig
    gaze: Dict
    blink: Dict
    mouth: MouthConfig
    animation_speed: float
    bounce: bool = False
    shake: bool = False
    jump: bool = False
    filter: Optional[str] = None

class EmotionAnimationManager:
    """Manage emotion-based animations"""
    
    CONFIGS = {
        EmotionType.HAPPY: AnimationConfig(
            emotion=EmotionType.HAPPY,
            gif="octopus_happy.gif",
            colors={"primary": (255, 200, 0), "secondary": (255, 150, 0)},
            eyes=EyeConfig(
                shape="squinting",
                width=0.7,
                color=(50, 50, 50),
                shine=True
            ),
            gaze={"x": 0.2, "y": -0.3, "duration_ms": 500},
            blink={"rate": 0.5, "duration_ms": 100},
            mouth=MouthConfig(curve=0.8, width=1.0),
            animation_speed=1.2,
            bounce=True
        ),
        EmotionType.SAD: AnimationConfig(
            emotion=EmotionType.SAD,
            gif="octopus_sad.gif",
            colors={"primary": (100, 150, 200), "secondary": (80, 120, 180)},
            eyes=EyeConfig(
                shape="drooping",
                width=0.8,
                color=(50, 50, 50),
                shine=False,
                droop=0.6
            ),
            gaze={"x": -0.3, "y": 0.3, "duration_ms": 1000},
            blink={"rate": 0.2, "duration_ms": 150},
            mouth=MouthConfig(curve=-0.7, width=0.8),
            animation_speed=0.7
        ),
        EmotionType.ANGRY: AnimationConfig(
            emotion=EmotionType.ANGRY,
            gif="octopus_angry.gif",
            colors={"primary": (255, 80, 80), "secondary": (200, 50, 50)},
            eyes=EyeConfig(
                shape="narrowed",
                width=0.4,
                color=(200, 0, 0),
                shine=False,
                angle=-15
            ),
            gaze={"x": 0.0, "y": -0.5, "duration_ms": 800},
            blink={"rate": 0.3, "duration_ms": 80},
            mouth=MouthConfig(curve=-0.9, width=0.6, clench=True),
            animation_speed=1.5,
            shake=True
        ),
        EmotionType.NEUTRAL: AnimationConfig(
            emotion=EmotionType.NEUTRAL,
            gif="octopus_neutral.gif",
            colors={"primary": (200, 200, 200), "secondary": (150, 150, 150)},
            eyes=EyeConfig(
                shape="normal",
                width=1.0,
                color=(50, 50, 50),
                shine=True
            ),
            gaze={"x": 0.0, "y": 0.0, "duration_ms": 600},
            blink={"rate": 0.3, "duration_ms": 120},
            mouth=MouthConfig(curve=0.0, width=0.9),
            animation_speed=1.0
        ),
        EmotionType.SURPRISED: AnimationConfig(
            emotion=EmotionType.SURPRISED,
            gif="octopus_surprised.gif",
            colors={"primary": (200, 100, 255), "secondary": (150, 50, 200)},
            eyes=EyeConfig(
                shape="wide",
                width=1.3,
                height=1.2,
                color=(50, 50, 50),
                shine=True,
                pupil_size=0.6
            ),
            gaze={"x": 0.0, "y": -0.4, "duration_ms": 400},
            blink={"rate": 0.4, "duration_ms": 110},
            mouth=MouthConfig(curve=0.5, width=0.7, shape="O"),
            animation_speed=1.3,
            jump=True
        ),
    }
    
    @staticmethod
    def get_config(emotion: str) -> AnimationConfig:
        """Get animation config for emotion"""
        
        try:
            emotion_type = EmotionType(emotion.lower())
            return EmotionAnimationManager.CONFIGS[emotion_type]
        except (ValueError, KeyError):
            # Fallback to neutral
            return EmotionAnimationManager.CONFIGS[EmotionType.NEUTRAL]
    
    @staticmethod
    def apply_emotion_animation(
        emotion: str,
        duration_ms: int
    ) -> Dict:
        """Apply emotion animation to response"""
        
        config = EmotionAnimationManager.get_config(emotion)
        
        return {
            "gif": config.gif,
            "colors": config.colors,
            "eyes": {
                "shape": config.eyes.shape,
                "width": config.eyes.width,
                "height": config.eyes.height,
                "color": config.eyes.color,
                "shine": config.eyes.shine,
                "droop": config.eyes.droop,
                "angle": config.eyes.angle,
                "pupil_size": config.eyes.pupil_size
            },
            "gaze": config.gaze,
            "blink": config.blink,
            "mouth": {
                "curve": config.mouth.curve,
                "width": config.mouth.width,
                "shape": config.mouth.shape,
                "clench": config.mouth.clench
            },
            "animation_speed": config.animation_speed,
            "effects": {
                "bounce": config.bounce,
                "shake": config.shake,
                "jump": config.jump,
                "filter": config.filter
            },
            "duration_ms": duration_ms
        }
```

## GIF Asset Structure

```
assets/
├── body_animations/
│   ├── octopus_happy.gif      (127MB total)
│   ├── octopus_sad.gif
│   ├── octopus_angry.gif
│   ├── octopus_neutral.gif
│   └── octopus_surprised.gif
│
├── eyes/
│   ├── eye_happy.png          (40x40)
│   ├── eye_sad.png
│   ├── eye_angry.png
│   ├── eye_neutral.png
│   └── eye_surprised.png
│
└── effects/
    ├── sparkle.png
    ├── glow.png
    └── shake.png
```

## Device Rendering

```python
# device_animation.py
class DeviceAnimationRenderer:
    """Render animations on BK7258 device"""
    
    def render_emotion_animation(
        self,
        emotion_config: Dict,
        phoneme_timings: list,
        duration_ms: int
    ):
        """Render complete emotion animation"""
        
        # 1. Load and render body GIF
        gif_path = f"assets/body_animations/{emotion_config['gif']}"
        self.display.load_gif(gif_path)
        
        # 2. Apply color filters
        colors = emotion_config['colors']
        self.display.set_color_filter(colors['primary'])
        
        # 3. Render eyes with emotion shape
        eyes_config = emotion_config['eyes']
        self.render_eyes(eyes_config)
        
        # 4. Render mouth with phoneme sync
        for timing in phoneme_timings:
            self.render_mouth_at_time(timing)
        
        # 5. Apply effects
        effects = emotion_config['effects']
        if effects['bounce']:
            self.add_bounce_effect()
        if effects['shake']:
            self.add_shake_effect()
        if effects['jump']:
            self.add_jump_effect()
        
        # 6. Play animation
        self.play_animation(
            speed=emotion_config['animation_speed'],
            duration_ms=duration_ms
        )
```

## Transition Between Emotions

```python
# emotion_transition.py
class EmotionTransition:
    """Smooth transition between emotions"""
    
    @staticmethod
    def transition(
        from_emotion: str,
        to_emotion: str,
        duration_ms: int = 500
    ) -> list:
        """Generate transition animation"""
        
        from_config = EmotionAnimationManager.get_config(from_emotion)
        to_config = EmotionAnimationManager.get_config(to_emotion)
        
        frames = []
        steps = 10
        
        for step in range(steps):
            progress = step / steps
            
            # Interpolate properties
            frame = {
                "time_ms": int((step / steps) * duration_ms),
                "animation_speed": (
                    from_config.animation_speed * (1 - progress) +
                    to_config.animation_speed * progress
                ),
                "colors": {
                    "primary": interpolate_color(
                        from_config.colors['primary'],
                        to_config.colors['primary'],
                        progress
                    )
                }
            }
            
            frames.append(frame)
        
        return frames

def interpolate_color(color1: tuple, color2: tuple, progress: float) -> tuple:
    """Interpolate between two colors"""
    
    return tuple(
        int(c1 * (1 - progress) + c2 * progress)
        for c1, c2 in zip(color1, color2)
    )
```

## Testing

```python
# test_emotion_animation.py
import pytest

def test_emotion_configs():
    """Test all emotion configs exist"""
    
    for emotion in EmotionType:
        config = EmotionAnimationManager.get_config(emotion.value)
        assert config is not None
        assert config.gif is not None

def test_emotion_colors():
    """Test emotion colors are valid"""
    
    config = EmotionAnimationManager.get_config("happy")
    
    for color_value in config.colors.values():
        assert isinstance(color_value, tuple)
        assert len(color_value) == 3
        assert all(0 <= c <= 255 for c in color_value)

def test_emotion_transition():
    """Test smooth emotion transitions"""
    
    frames = EmotionTransition.transition("happy", "sad", 500)
    
    assert len(frames) > 0
    assert frames[0]['time_ms'] == 0
    assert frames[-1]['time_ms'] <= 500
```

## Summary

- **5 emotions** - Happy, Sad, Angry, Neutral, Surprised
- **Complete configs** - GIF, colors, eyes, mouth, effects
- **Eye shapes** - Squinting, drooping, narrowed, normal, wide
- **Special effects** - Bounce, shake, jump, filters
- **Smooth transitions** - Between emotions
- **Production-ready** - Full implementation with testing

Perfect for emotional octopus expressions! 
