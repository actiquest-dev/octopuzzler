# Avatar Design

## Overview

Complete guide for designing and creating the Octopus Avatar (Ollie) with 5 emotion-based GIFs and parametric eyes.

```
Design Concept
    ↓
Asset Creation (GIFs, PNGs)
    ↓
Color Schemes & Emotion Mapping
    ↓
Animation Frame Sequence
    ↓
Device Integration & Testing
```

## Design Concept: Ollie the Octopus

### Personality
- Friendly and approachable
- Empathic and responsive
- Playful and curious
- Warm and encouraging
- Expressive and animated

### Visual Style
- Cartoon/stylized (not realistic)
- Soft rounded shapes
- Expressive features
- Colorful and inviting
- Optimistic and positive

### Color Palette

```
Base Colors:
- Octopus skin: #E8A0BF (rose/mauve)
- Eyes: #333333 (dark grey)
- Shine: #FFFFFF (white)
- Accent: #FF6B9D (pink)

Emotion Variations:
- Happy:    Warm yellows, oranges (#FFD700, #FFA500)
- Sad:      Cool blues (#4A90E2, #357ABD)
- Angry:    Reds (#FF4444, #CC0000)
- Neutral:  Greys (#CCCCCC, #999999)
- Surprised: Purples (#B74DD6, #9933CC)
```

## GIF Specifications

### Technical Requirements

```
Format:           GIF (animated)
Resolution:       160x160 pixels
Frame Rate:       15 FPS
Duration:         ~2 seconds (30 frames)
Color Depth:      8-bit (256 colors)
File Size:        ~500KB per GIF (total 127MB for avatar bank)
Transparency:     Alpha channel (RGBA)
Loop:             Yes (infinite)
```

### Directory Structure

```
assets/
├── avatars/
│   ├── octopus_happy.gif        (500KB)
│   ├── octopus_sad.gif          (500KB)
│   ├── octopus_angry.gif        (500KB)
│   ├── octopus_neutral.gif      (500KB)
│   └── octopus_surprised.gif    (500KB)
│
├── overlays/
│   ├── mouth/
│   │   ├── mouth_0.png          (0 - SIL, closed)
│   │   ├── mouth_1.png          (1 - AA, wide open)
│   │   ├── mouth_2.png          (2 - EH, mid-open)
│   │   ├── mouth_3.png          (3 - IH, narrow)
│   │   ├── mouth_4.png          (4 - OW, round)
│   │   ├── mouth_5.png          (5 - UH, mid-narrow)
│   │   ├── mouth_6.png          (6 - M, lips together)
│   │   ├── mouth_7.png          (7 - F, friction)
│   │   └── mouth_8.png          (8 - L, tongue visible)
│   │
│   └── eyes/
│       ├── eye_base.png         (eye container)
│       ├── pupil.png            (movable pupil)
│       └── shine.png            (reflection)
│
└── effects/
    ├── sparkle.png              (happy effect)
    ├── glow.png                 (emotion aura)
    └── hearts.png               (love effect)
```

### Mouth Shapes (PNG Overlays)

Each mouth shape: **40x40 pixels, RGBA, transparent background**

```
0 - SIL (Silence)
    ┌────────┐
    │        │
    │  ──── │  Closed, straight mouth line
    │        │
    └────────┘

1 - AA (Wide Open)
    ┌────────┐
    │   ╱╲   │
    │  ╱  ╲  │  Wide open circle/oval
    │ │    │ │
    │  ╲  ╱  │
    │   ╲╱   │
    └────────┘

2 - EH (Mid-Open)
    ┌────────┐
    │        │
    │   ╱─╲  │  Mid-open, vertical oval
    │  │   │ │
    │   ╲─╱  │
    │        │
    └────────┘

3 - IH (Narrow)
    ┌────────┐
    │        │
    │  ┌──┐  │  Narrow, horizontal
    │  └──┘  │
    │        │
    └────────┘

4 - OW (Round/O)
    ┌────────┐
    │        │
    │   ◯    │  Round O shape
    │        │
    │        │
    └────────┘

5 - UH (Mid-Narrow)
    ┌────────┐
    │        │
    │  ╱──╲  │  Mid-narrow rounded
    │ │    │ │
    │  ╲──╱  │
    │        │
    └────────┘

6 - M (Lips Together)
    ┌────────┐
    │        │
    │  ║  ║  │  Vertical lips pressed
    │  ║  ║  │
    │        │
    └────────┘

7 - F (Friction/Teeth)
    ┌────────┐
    │        │
    │  ─────  │  Lower lip to upper teeth
    │     │  Visible teeth/edge
    │  ─────  │
    │        │
    └────────┘

8 - L (Tongue Visible)
    ┌────────┐
    │        │
    │  ┌───┐ │  Tongue sticking out
    │  │ │ │
    │  └───┘ │
    └────────┘
```

## GIF Creation Process

### Option 1: Using Blender (Professional)

```python
# blender_gif_generator.py
import bpy
import os

class BlenderGIFCreator:
    """Create GIF animations using Blender"""
    
    def __init__(self):
        self.scene = bpy.context.scene
        self.output_dir = "output_gifs"
    
    def create_emotion_gif(self, emotion: str, num_frames: int = 30):
        """Create GIF for specific emotion"""
        
        # Load octopus model
        model_path = f"models/octopus_{emotion}.blend"
        bpy.ops.wm.open_mainfile(filepath=model_path)
        
        # Set animation length
        self.scene.frame_end = num_frames
        
        # Set render settings
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.filepath = f"{self.output_dir}/frame_"
        self.scene.render.resolution_x = 160
        self.scene.render.resolution_y = 160
        
        # Render all frames
        bpy.ops.render.render(animation=True, write_still=True)
        
        # Convert PNG sequence to GIF
        self._png_to_gif(emotion, num_frames)
    
    def _png_to_gif(self, emotion: str, num_frames: int):
        """Convert PNG sequence to GIF"""
        
        from PIL import Image
        
        frames = []
        
        for i in range(1, num_frames + 1):
            frame_path = f"{self.output_dir}/frame_{i:04d}.png"
            frame = Image.open(frame_path)
            frames.append(frame)
        
        # Save as GIF
        gif_path = f"assets/avatars/octopus_{emotion}.gif"
        
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=66,  # 15 FPS = 66ms per frame
            loop=0,       # Infinite loop
            optimize=False
        )
        
        print(f" Created {gif_path}")
```

### Option 2: Using Python PIL (Simple)

```python
# gif_generator.py
from PIL import Image, ImageDraw
import math

class SimpleGIFCreator:
    """Create GIFs using PIL"""
    
    def __init__(self, width=160, height=160, num_frames=30):
        self.width = width
        self.height = height
        self.num_frames = num_frames
    
    def create_happy_octopus(self):
        """Create happy octopus GIF"""
        
        frames = []
        
        for frame_idx in range(self.num_frames):
            # Create frame
            img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Animation progress
            t = frame_idx / self.num_frames
            
            # Draw body (octopus shape)
            self._draw_octopus_body(draw, t, emotion="happy")
            
            # Draw tentacles
            self._draw_tentacles(draw, t, emotion="happy")
            
            # Draw eyes
            self._draw_eyes(draw, t, emotion="happy")
            
            frames.append(img)
        
        # Save as GIF
        frames[0].save(
            'assets/avatars/octopus_happy.gif',
            save_all=True,
            append_images=frames[1:],
            duration=66,
            loop=0
        )
        
        print(" Created octopus_happy.gif")
    
    def _draw_octopus_body(self, draw, t, emotion="neutral"):
        """Draw octopus body"""
        
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Body circle
        body_radius = 40
        
        # Color based on emotion
        if emotion == "happy":
            color = (255, 200, 0, 255)      # Yellow
        elif emotion == "sad":
            color = (100, 150, 200, 255)    # Blue
        elif emotion == "angry":
            color = (255, 80, 80, 255)      # Red
        elif emotion == "surprised":
            color = (200, 100, 255, 255)    # Purple
        else:
            color = (200, 200, 200, 255)    # Grey
        
        # Add bounce animation
        bounce = math.sin(t * 2 * math.pi) * 3
        
        draw.ellipse(
            [
                center_x - body_radius,
                center_y - body_radius + bounce,
                center_x + body_radius,
                center_y + body_radius + bounce
            ],
            fill=color,
            outline=(50, 50, 50, 255),
            width=2
        )
    
    def _draw_tentacles(self, draw, t, emotion="neutral"):
        """Draw octopus tentacles"""
        
        center_x = self.width / 2
        center_y = self.height / 2
        
        # 8 tentacles
        for tentacle_idx in range(8):
            angle = (tentacle_idx / 8) * 2 * math.pi
            
            # Tentacle animation (wave motion)
            wave = math.sin(t * 2 * math.pi + angle) * 5
            
            # Draw tentacle line
            x1 = center_x + 40 * math.cos(angle)
            y1 = center_y + 40 * math.sin(angle)
            x2 = center_x + 60 * math.cos(angle)
            y2 = center_y + 60 * math.sin(angle) + wave
            
            draw.line([(x1, y1), (x2, y2)], fill=(200, 100, 150, 255), width=3)
    
    def _draw_eyes(self, draw, t, emotion="neutral"):
        """Draw eyes"""
        
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Eyes positioned above body
        left_eye_x = center_x - 15
        right_eye_x = center_x + 15
        eyes_y = center_y - 20
        
        # Eye color based on emotion
        if emotion == "happy":
            eye_color = (255, 200, 0, 255)
        elif emotion == "sad":
            eye_color = (100, 150, 200, 255)
        else:
            eye_color = (255, 255, 255, 255)
        
        # Draw eyes
        eye_radius = 8
        
        # Left eye
        draw.ellipse(
            [left_eye_x - eye_radius, eyes_y - eye_radius,
             left_eye_x + eye_radius, eyes_y + eye_radius],
            fill=eye_color,
            outline=(50, 50, 50, 255),
            width=1
        )
        
        # Right eye
        draw.ellipse(
            [right_eye_x - eye_radius, eyes_y - eye_radius,
             right_eye_x + eye_radius, eyes_y + eye_radius],
            fill=eye_color,
            outline=(50, 50, 50, 255),
            width=1
        )
        
        # Pupils with blinking animation
        blink = max(0, math.cos((t + 0.7) * 2 * math.pi))
        pupil_size = 3 * blink
        
        if pupil_size > 0.5:
            pupil_color = (50, 50, 50, 255)
            
            draw.ellipse(
                [left_eye_x - pupil_size, eyes_y - pupil_size,
                 left_eye_x + pupil_size, eyes_y + pupil_size],
                fill=pupil_color
            )
            
            draw.ellipse(
                [right_eye_x - pupil_size, eyes_y - pupil_size,
                 right_eye_x + pupil_size, eyes_y + pupil_size],
                fill=pupil_color
            )
```

### Option 3: Using Aseprite (Pixel Art)

```
1. Create new file: 160x160 pixels
2. Create layers:
   - Layer "Body" - main octopus shape
   - Layer "Tentacles" - 8 tentacles
   - Layer "Eyes" - eyes with pupils
   - Layer "Mouth" - optional mouth
   - Layer "Effects" - sparkles, blush, etc.

3. Create animation frames:
   Frame 0: Neutral pose
   Frame 1-5: Bounce up animation
   Frame 6-10: Bounce down animation
   Frame 11-20: Idle/wiggle animation
   Frame 21-30: Return to neutral

4. Export as GIF:
   - Frame rate: 15 FPS (66ms per frame)
   - Infinite loop: YES
   - Optimize: NO (keep quality)
```

## Eye Design

### Eye Structure

```
Container (40x40 PNG):
┌─────────────────┐
│                 │
│     ◯      ◯    │  Two eyes
│                 │
│                 │
└─────────────────┘

Components:
- Eye white (sclera) - circle, white
- Iris - colored circle (emotion-based)
- Pupil - black circle, movable
- Shine/reflection - small white spot
- Eyelids - optional, emotion-based
```

### Parametric Eyes (Device-side)

```python
# parametric_eyes.py
import math

class ParametricEyes:
    """Render parametric eyes on device"""
    
    def __init__(self, width=40, height=40):
        self.width = width
        self.height = height
    
    def render_eye(
        self,
        emotion: str,
        gaze_x: float = 0.0,  # -1 to 1
        gaze_y: float = 0.0,  # -1 to 1
        blink: float = 1.0    # 0 to 1
    ):
        """Render single eye with gaze and blink"""
        
        # Eye dimensions
        eye_radius = 12
        pupil_radius = 5
        shine_radius = 2
        
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Draw eye white
        draw_circle(
            center_x, center_y,
            eye_radius,
            color=(255, 255, 255, 255)
        )
        
        # Eye outline (emotion color)
        emotion_color = self._get_emotion_color(emotion)
        draw_circle(
            center_x, center_y,
            eye_radius,
            color=emotion_color,
            outline=True,
            width=2
        )
        
        # Eyelid animation (blink)
        if blink < 1.0:
            eyelid_height = eye_radius * 2 * (1 - blink)
            draw_rectangle(
                center_x - eye_radius,
                center_y - eye_radius,
                center_x + eye_radius,
                center_y - eye_radius + eyelid_height,
                color=(50, 50, 50, 255)
            )
        
        # Pupil position based on gaze
        pupil_x = center_x + (gaze_x * 5)
        pupil_y = center_y + (gaze_y * 4)
        
        # Draw pupil
        draw_circle(
            pupil_x, pupil_y,
            pupil_radius,
            color=(50, 50, 50, 255)
        )
        
        # Draw shine (reflection)
        shine_x = pupil_x - 2
        shine_y = pupil_y - 2
        
        draw_circle(
            shine_x, shine_y,
            shine_radius,
            color=(255, 255, 255, 255)
        )
    
    def _get_emotion_color(self, emotion: str):
        """Get iris color based on emotion"""
        
        colors = {
            "happy": (200, 150, 50, 255),      # Golden
            "sad": (80, 120, 180, 255),        # Blue
            "angry": (180, 50, 50, 255),       # Red
            "neutral": (100, 100, 100, 255),   # Grey
            "surprised": (150, 100, 200, 255)  # Purple
        }
        
        return colors.get(emotion, colors["neutral"])
```

## Animation Frame Sequence

### Happy Emotion (30 frames @ 15 FPS)

```
Frame 0-5:    Neutral to happy expression
Frame 5-10:   Bounce up (positive energy)
Frame 10-15:  Bounce down and wiggle left
Frame 15-20:  Wiggle right
Frame 20-25:  Small bounces
Frame 25-30:  Return to neutral, loop
```

### Sad Emotion (30 frames @ 15 FPS)

```
Frame 0-10:   Sad expression (drooping)
Frame 10-20:  Very subtle sway left/right (melancholic)
Frame 20-30:  Back to sad, loop
```

### Angry Emotion (30 frames @ 15 FPS)

```
Frame 0-5:    Angry expression (intense)
Frame 5-10:   Aggressive shake left
Frame 10-15:  Aggressive shake right
Frame 15-20:  Back to angry
Frame 20-30:  Repeat, loop
```

## Asset Specifications Summary

```
BODY GIFS:
├─ Format: GIF, 160x160px, 15 FPS, 30 frames
├─ Colors: Emotion-specific palette
├─ Animation: Emotion-appropriate movement
└─ Size: ~500KB each, 5 total (2.5MB)

MOUTH OVERLAYS:
├─ Format: PNG, 40x40px, RGBA
├─ Count: 8 shapes (0-8)
├─ Colors: Match body emotion
└─ Size: ~10KB each, 80KB total

EYE OVERLAYS:
├─ Format: PNG, parametric rendering
├─ Components: Sclera, iris, pupil, shine
├─ Animation: Blink, gaze follow
└─ Size: Rendered on-device

TOTAL STORAGE: ~127MB (body GIFs bank)
```

## Quality Checklist

```
Visual Design:
 Octopus is friendly and approachable
 5 emotions are clearly distinguishable
 Colors match emotion themes
 Animations are smooth (30fps smooth rendering)
 Eyes are expressive and responsive

Technical:
 GIFs are 160x160 pixels
 Frame rate is consistent (15 FPS)
 File sizes are optimized
 Transparency is correct
 Colors are Web-safe or optimized

Animation:
 Happy: Bouncy, energetic
 Sad: Slow, melancholic
 Angry: Fast, aggressive
 Neutral: Calm, idle
 Surprised: Quick, expressive

Performance:
 GIFs load under 500ms
 Animations don't stutter
 Memory usage is minimal
 CPU load is < 10% during animation
```

## Tools Needed

```
Required:
- Blender 3.6+ (3D modeling)
- Aseprite (pixel art animation)
- Python PIL (GIF manipulation)
- GIMP (image editing)

Optional:
- Clip Studio Paint (animation)
- OpenToonz (professional animation)
- Photoshop (image editing)
```

## Timeline

```
Week 1: Design & Concept Art
  ├─ Create design document
  ├─ Make concept sketches
  └─ Color palette refinement

Week 2: Body GIF Creation
  ├─ Create base octopus model
  ├─ Design 5 emotion variations
  ├─ Animate for each emotion
  └─ Export and optimize GIFs

Week 3: Mouth Shapes & Eyes
  ├─ Design 8 mouth shapes
  ├─ Create mouth PNG overlays
  ├─ Design parametric eye system
  └─ Test eye animations

Week 4: Integration & Testing
  ├─ Load GIFs on device
  ├─ Test mouth sync
  ├─ Test eye tracking
  ├─ Optimize assets
  └─ Final QA
```

## Summary

- **Design**: Friendly octopus mascot (Ollie)
- **5 Emotions**: Happy, Sad, Angry, Neutral, Surprised
- **Body GIFs**: 160x160px, 15 FPS, emotion-specific
- **Mouth Overlays**: 8 shapes, 40x40px PNG
- **Eyes**: Parametric, movable, emotion-aware
- **Storage**: ~127MB total for full avatar bank
- **Performance**: Smooth on BK7258 device

Complete avatar ready for production! 
