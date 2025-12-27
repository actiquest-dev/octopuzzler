# WowCube Integration Guide

## Hardware Overview

WowCube - modular cube with:
- 6 screens (240x240 LCD each)
- Microphone input
- WiFi connectivity
- Battery/USB power
- BK7258 microcontroller

## Integration Points

### 1. Display Output

WowCube has 6 LCD displays arranged as cube faces.

```
Main display (front):
- 240x240 LCD
- SPI interface
- Octopus avatar (mouth + eyes + body)

Side displays:
- Optional animations
- Status indicators
```

### 2. Audio Input

Microphone connects to BK7258 ADC:
- 16kHz sample rate
- 16-bit PCM
- Continuous capture
- Ring buffer management

### 3. Power Management

```
Battery: 3000mAh
├─ BK7258: 200-250mW
├─ Displays: 500-800mW
└─ WiFi: 100-150mW

Total: ~1W average
Battery life: 3-4 hours active use
```

### 4. WiFi Module

Built-in 2.4GHz WiFi:
- Connect to local network
- Backend IP: 192.168.x.x or cloud
- WebSocket for real-time audio/commands

## Physical Layout

```
        [Back]
           |
[Left]--[Front]--[Right]
           |
        [Bottom]
        [Top]

Main avatar on front display
Side animations on other displays (optional)
```

## Firmware Pinout

```
BK7258 GPIO:
├─ Display SPI: GPIO 4 (CLK), 5 (MOSI), 10 (DC), 11 (CS)
├─ Microphone ADC: GPIO 20 (AIN0)
├─ WiFi: Built-in module
└─ Status LED: GPIO 15
```

## Display Controller

Use ILI9341 or similar for each screen.

```c
// Initialize main display
configure_spi(10000000);  // 10MHz
init_ili9341_display();

// Set window for avatar
set_window(0, 0, 239, 239);

// Draw frame
display_write_frame(avatar_data);
```

## Animation Rendering

### Front Display (Main)

```
┌─────────────────┐
│    [GIF Body]   │
│  (160x160, top) │
├─────────────────┤
│  [Mouth Overlay]│
│    (40x40)      │
│  [Eyes Overlay] │
│    (40x40)      │
└─────────────────┘
```

### GIF Animation

Load GIF frames from storage:
```c
uint8_t* body_gif = load_asset("avatar_body.gif");
uint8_t* frame = get_gif_frame(body_gif, frame_idx);
display_draw_sprite(frame, 40, 20, 160, 160);
```

### Mouth Shapes

8 PNG overlays for phonemes:
```c
uint8_t* mouth_shapes = load_asset("mouth_shapes.png");
uint8_t* mouth = get_mouth_shape(mouth_shapes, phoneme_idx);
display_draw_sprite(mouth, 100, 140, 40, 40);
```

### Eyes Animation

Parametric eyes with emotion:
```c
void draw_eyes(int x, int y, float blink, int emotion) {
    // emotion: 0=sad, 1=neutral, 2=happy
    // blink: 0-1 (0=open, 1=closed)
    
    int eye_width = 20;
    int eye_height = (int)(20 * (1 - blink * 0.8));
    
    // Left eye
    draw_ellipse(80 + x, 80 + y, eye_width, eye_height);
    
    // Right eye
    draw_ellipse(160 + x, 80 + y, eye_width, eye_height);
}
```

## Side Displays (Optional)

Small animations on cube faces:
- Status indicator
- Waveform visualization
- Emotion indicator
- Floating animations

```c
// Update side displays
update_side_displays(emotion, language);
```

## Asset Storage

Store on microSD card or internal flash:
```
/assets/
├─ avatar_body.gif      (animated body)
├─ mouth_shapes.png     (8 phoneme shapes)
├─ eyes.json            (parametric eye config)
├─ animations.json      (animation definitions)
└─ status_icons.png     (status indicators)

Total: ~127MB
```

## Power Efficiency

Optimize for battery life:

```c
// Reduce display refresh when idle
if(idle_time > 30000) {
    display_refresh_rate = 5;  // 5 FPS instead of 15
    wifi_power_save(1);
}

// Turn off side displays if not needed
if(!show_side_animations) {
    power_down_side_displays();
}
```

## Testing Integration

### Hardware Test
```bash
# Test display
./test_display --device /dev/ttyUSB0

# Test audio
./test_audio --device /dev/ttyUSB0

# Test WiFi
./test_wifi --ssid "MyNetwork" --password "pass"
```

### End-to-End
```bash
# Start backend
python3 backend/main_gateway.py

# Connect device
make upload_firmware

# Test via web client
open http://localhost:8000
```

## Troubleshooting

**Display not showing:**
- Check SPI connections (CLK, MOSI, DC, CS)
- Verify chip select timing
- Increase SPI clock if working

**Audio glitches:**
- Increase ring buffer size
- Check WiFi signal strength
- Reduce processing latency

**WiFi drops:**
- Move closer to router
- Check WiFi credentials
- Implement auto-reconnect

**Battery drains fast:**
- Reduce display refresh rate
- Turn off side displays
- Implement idle power mode

## Deployment Checklist

- [ ] Display working (all faces)
- [ ] Audio capturing cleanly
- [ ] WiFi connecting reliably
- [ ] Avatar animating smoothly
- [ ] Battery lasting 3+ hours
- [ ] All emotions showing correctly
- [ ] Mouth sync accurate
- [ ] Eye blinks natural
- [ ] No glitches at high load
- [ ] Device survives 24h test

## Next Steps

1. Flash firmware to WowCube
2. Test each component separately
3. Integrate with backend
4. Run end-to-end test
5. Deploy to production

Typical integration time: 1-2 weeks
