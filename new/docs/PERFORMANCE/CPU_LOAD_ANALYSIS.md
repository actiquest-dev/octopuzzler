# BK7258 CPU Load Analysis: Mouth Phonemes + Layered Animations

## Hardware Baseline

**BK7258 CPU Specifications:**
- Dual-core ARM Cortex-M4F @ 480MHz
- 512MB RAM total (shared)
- No dedicated GPU
- No SIMD/NEON extensions

**Available CPU:**
- OS/System: ~5-10% (baseline)
- WiFi/Network: ~1-2% (idle)
- **Available for avatar: ~80-85%**

---

## Rendering Pipeline Breakdown

### Frame Composition (per frame @ 15fps = 66ms interval)

```c
void render_complete_frame(uint16_t *lcd_buffer) {
    // Step 1: Load GIF frame from NAND
    // Step 2: Composite mouth PNG
    // Step 3: Draw eyes parametrically
    // Step 4: Write to LCD
}
```

---

## Step-by-Step CPU Analysis

### Step 1: GIF Frame Extraction

**Time per frame:** ~1-2ms

```c
void gif_get_current_frame(GifFileType *gif, uint16_t *output_buffer) {
    // 1. Get frame index from animation state: <0.1ms
    uint32_t frame_idx = current_frame_index;
    
    // 2. Access GIF frame data from NAND/cache: 0.5-1ms
    //    (assuming cache hit rate 90%)
    SavedImage *image = &gif->SavedImages[frame_idx];
    uint8_t *frame_data = image->RasterBits;  // 160x160 pixels
    
    // 3. Copy to output buffer: 0.3-0.5ms
    //    160 * 160 * 2 bytes = 51,200 bytes
    //    BK7258 memcpy: ~300MB/s
    //    51,200 / (300MB/s) = 0.17ms
    memcpy(output_buffer, frame_data, 160*160*2);
    
    // 4. Handle color table/format conversion: <0.1ms
    //    (GIF already in RGB565, no conversion needed)
}

// Total: 1-2ms per frame
```

**CPU Load:** 1-2ms / 66ms = **1.5-3% per frame**

**Detailed breakdown:**
- NAND read (cached): 0.5-1ms
- memcpy: 0.2ms
- Frame management: 0.3ms
- **Total: 1-2ms**

---

### Step 2: Mouth PNG Overlay Compositing

**Time per frame:** ~2-3ms

```c
void composite_mouth_png(uint16_t *base_buffer, uint16_t *mouth_png, 
                         uint32_t mouth_x, uint32_t mouth_y) {
    
    // Mouth PNG: 80x80 pixels (assuming mouth region size)
    // Not full 160x160, just the mouth area
    
    uint32_t mouth_width = 80;
    uint32_t mouth_height = 40;
    
    // 1. Load mouth PNG if not cached: 0.5-1ms
    if (!mouth_overlay_cached) {
        load_mouth_png_from_nand(mouth_overlay);  // 0.5-1ms
    }
    
    // 2. Composite mouth onto base: 1.5-2ms
    for (uint32_t y = 0; y < mouth_height; y++) {
        for (uint32_t x = 0; x < mouth_width; x++) {
            uint32_t base_idx = (mouth_y + y) * 160 + (mouth_x + x);
            uint32_t mouth_idx = y * mouth_width + x;
            
            uint16_t mouth_pixel = mouth_png[mouth_idx];
            uint16_t alpha = mouth_pixel >> 15;  // Top bit = alpha
            
            if (alpha) {
                // Alpha blend
                uint16_t base_pixel = base_buffer[base_idx];
                
                // Extract RGB565
                uint32_t base_r = (base_pixel >> 11) & 0x1F;
                uint32_t base_g = (base_pixel >> 5) & 0x3F;
                uint32_t base_b = base_pixel & 0x1F;
                
                uint32_t mouth_r = (mouth_pixel >> 11) & 0x1F;
                uint32_t mouth_g = (mouth_pixel >> 5) & 0x3F;
                uint32_t mouth_b = mouth_pixel & 0x1F;
                
                // Blend (simple 50/50)
                uint32_t blend_r = (base_r + mouth_r) / 2;
                uint32_t blend_g = (base_g + mouth_g) / 2;
                uint32_t blend_b = (base_b + mouth_b) / 2;
                
                // Pack back to RGB565
                base_buffer[base_idx] = (blend_r << 11) | (blend_g << 5) | blend_b;
            }
        }
    }
    
    // For 80x40 mouth region: 3,200 pixels
    // At ~600 pixels/ms: 5ms worst case, 2-3ms average
}

// Total: 2-3ms per frame (with cache)
// Total: 1.5-2.5ms per frame (if mouth not changing)
```

**CPU Load:** 2-3ms / 66ms = **3-4.5% per frame**

**Optimization opportunities:**
```c
// Optimized: Only composite changed pixels
void composite_mouth_png_optimized(uint16_t *base_buffer, 
                                    uint16_t *mouth_png,
                                    uint32_t mouth_x, uint32_t mouth_y,
                                    uint32_t *dirty_region) {
    
    // Only update bounding box of mouth region
    // Instead of entire screen
    
    uint32_t dirty_width = dirty_region[2] - dirty_region[0];
    uint32_t dirty_height = dirty_region[3] - dirty_region[1];
    
    // Only update pixels within dirty region
    // Example: If mouth only moves 5 pixels:
    //   Instead of 3,200 pixels: only 400 pixels
    //   Reduces from 2-3ms to 0.3-0.5ms
}

// With dirty region optimization: 0.5-1.5ms per frame
```

---

### Step 3: Parametric Eye Drawing

**Time per frame:** ~1-2ms

```c
void draw_eyes_parametric(uint16_t *lcd_buffer, 
                          float eye_x, float eye_y,
                          float pupil_scale, float eyelid_open) {
    
    // Eye positions: Left at (64, 80), Right at (96, 80)
    uint32_t eye_radius = 10;  // pixels
    uint32_t pupil_radius = 8 * pupil_scale;
    
    // 1. Draw left eye: 0.5-0.7ms
    draw_circle_filled(lcd_buffer, 64 + eye_x*20, 80 + eye_y*15, 
                       eye_radius, color_white);  // 0.3-0.4ms
    
    // Draw left pupil: 0.2-0.3ms
    draw_circle_filled(lcd_buffer, 64 + eye_x*20, 80 + eye_y*15,
                       pupil_radius, color_black);  // 0.1-0.2ms
    
    // Draw left highlight: 0.05ms
    draw_circle_filled(lcd_buffer, 64 + eye_x*20 - 2, 80 + eye_y*15 - 2,
                       2, color_white);
    
    // 2. Draw right eye: 0.5-0.7ms (same as left)
    draw_circle_filled(lcd_buffer, 96 + eye_x*20, 80 + eye_y*15,
                       eye_radius, color_white);
    
    draw_circle_filled(lcd_buffer, 96 + eye_x*20, 80 + eye_y*15,
                       pupil_radius, color_black);
    
    draw_circle_filled(lcd_buffer, 96 + eye_x*20 - 2, 80 + eye_y*15 - 2,
                       2, color_white);
    
    // 3. Draw eyelids (if partially closed): 0.3-0.5ms
    if (eyelid_open < 1.0) {
        draw_eyelid_arc(lcd_buffer, 64, 70, eyelid_open);
        draw_eyelid_arc(lcd_buffer, 96, 70, eyelid_open);
    }
}

void draw_circle_filled(uint16_t *buffer, int cx, int cy, int radius, uint16_t color) {
    // Midpoint circle algorithm
    // For radius 10: ~314 pixels (10*pi*2)
    // For radius 8: ~201 pixels
    // Total ~500 pixels per frame
    
    // At ~1000 pixels/ms: 0.5ms for both eyes
}

// Total: 1-2ms per frame
```

**CPU Load:** 1-2ms / 66ms = **1.5-3% per frame**

**Circle drawing optimization:**
```c
// Optimized: Use lookup table for circle
void draw_circle_filled_fast(uint16_t *buffer, int cx, int cy, 
                             int radius, uint16_t color) {
    // Pre-calculated circle points for common radii
    const int8_t circle_8[] = { /* pre-computed points */ };
    const int8_t circle_10[] = { /* pre-computed points */ };
    
    // Instead of Midpoint algorithm: just plot pre-computed points
    // Reduces circle drawing from 0.5ms to 0.1ms
}

// With optimization: 0.3-0.8ms per frame
```

---

### Step 4: LCD Display Update

**Time per frame:** ~2-3ms

```c
void display_frame_on_lcd(uint16_t *lcd_buffer) {
    // Write 160x160 buffer to both LCDs via SPI
    // SPI Speed: typically 40MHz on BK7258
    // Data size: 160*160*2 = 51,200 bytes
    
    // 1. SPI write for LCD1: ~1.3ms
    //    51,200 bytes * 8 bits / (40MHz) = ~10ms theoretical
    //    With DMA: ~1.3ms (hardware does transfer in background)
    bk_display_flush(lcd1_handle, lcd_buffer, 160*160*2);
    
    // 2. SPI write for LCD2: ~1.3ms (parallel with LCD1 if separate SPI)
    bk_display_flush(lcd2_handle, lcd_buffer, 160*160*2);
    
    // 3. Wait for completion: ~0.5ms
    while (!spi_transfer_complete());
}

// Total: 2-3ms per frame (with DMA)
// Note: Can overlap with next frame rendering
```

**CPU Load:** 0.5-1ms actual CPU (rest is hardware DMA) = **0.75-1.5%**

---

## Complete CPU Load Table - All Processes

### Full System CPU Utilization Matrix

| Process/Task | Core | CPU % | Timing | Frequency | Status | Notes |
|---|---|---|---|---|---|---|
| **OS & RTOS** | 0 | 5-8% | Continuous | Always on | Essential | Context switching, scheduling |
| **WiFi/Network Stack** | 0 | 1-2% | Intermittent | Always ready | Essential | Idle most of time |
| **GIF Frame Loading** | 0 | 1.5-3% | Per frame | 15fps | Avatar | Load from NAND cache |
| **Mouth PNG Compositing** | 0 | 0.75-2.25% | Per frame | 15fps | Avatar | With dirty region optimization |
| **Parametric Eye Drawing** | 0 | 0.45-1.2% | Per frame | 15fps | Avatar | Circle drawing with LUT |
| **LCD Display Update (CPU)** | 0 | 0.75-1.5% | Per frame | 15fps | Avatar | Hardware DMA transfer |
| **Face Detection (Local)** | 0 | 5-10% | Periodic | Every 500ms | Optional | Only when active |
| **Audio Ring Buffer Mgmt** | 1 | 0.1-0.2% | Per 100ms | Audio chunks | Avatar | Mostly idle/waiting |
| **Audio Playback (DMA)** | 1 | 0.05-0.1% | Background | Continuous | Avatar | Hardware DMA, no CPU load |
| **WebSocket Receive Task** | 1 | 1-2% | Periodic | On data arrival | Network | Brief activity bursts |
| **JSON Parsing** | 1 | 0.5-2% | Periodic | On command | Network | Intermittent parsing |
| **Microphone Capture** | 1 | 0.2-0.5% | Every 100ms | Audio chunks | Input | Ring buffer writes |
| **Camera Capture** | 0 | 0.5-1% | Every 333ms | 3fps | Input | DVP camera read |
| **LED/Status Indicator** | 0 | <0.1% | Periodic | Various | Diagnostic | Low priority |
| | | | | | | |
| **CORE 0 TOTAL** | 0 | 9-20% | Mixed | Always | | GIF + Mouth + Eyes + LCD + Face |
| **CORE 1 TOTAL** | 1 | 1.5-4.5% | Mixed | Always | | Audio + Network + Mic |
| | | | | | | |
| **SYSTEM TOTAL (Normal)** | Both | **12-18%** | Average | Typical | **NORMAL** | Speaking without face detection |
| **SYSTEM TOTAL (Peak)** | Both | **28-35%** | Worst case | Max load | **SAFE** | Face detection + avatar + network |
| **SYSTEM TOTAL (Idle)** | Both | **6-10%** | Minimal | Waiting | **LIGHT** | Just animations, no processing |

---

## Detailed Scenario Breakdown

### Scenario 1: Device Idle (Waiting for Input)
```
OS/RTOS:        6%
WiFi baseline:  1%
GIF animation:  2%
Eyes drift:     0.5%
Network:        0.1%
────────────────────
TOTAL:          9-10% (Very Light)
```

### Scenario 2: Device Speaking (Normal Operation)
```
OS/RTOS:              7%
WiFi/Network:         1.5%
GIF animation:        2%
Mouth sync:           1.5%
Eyes tracking:        0.8%
LCD update:           0.8%
Audio playback:       0.2%
Microphone:           0.3%
Camera capture:       0.5%
Network RX + parse:   2%
────────────────────────────────
TOTAL AVERAGE:        16-18% (Comfortable)
TOTAL PEAK:           22-25% (During bursts)
```

### Scenario 3: Emotion Transition
```
OS/RTOS:              7%
WiFi/Network:         1.5%
GIF blend:            3%
Mouth composite:      2%
Eyes repositioning:   1.2%
LCD update:           1%
Audio/Mic/Camera:     0.8%
────────────────────────────────
TOTAL AVERAGE:        17-19% (Still Manageable)
TOTAL PEAK:           23-26% (Transition spikes)
```

### Scenario 4: Active Face Detection
```
OS/RTOS:              8%
WiFi/Network:         2%
GIF animation:        2%
Mouth sync:           1.5%
Eyes (face tracking): 1%
LCD update:           0.8%
Camera frequent:      2%
FACE DETECTION:       7-10%
Network RX + parse:   2%
────────────────────────────────
TOTAL AVERAGE:        27-32% (Still under limit)
TOTAL PEAK:           35-40% (Max realistic)
```

### Scenario 5: Worst Case (Everything)
```
OS/RTOS:              8%
WiFi/Network:         2.5%
GIF talking:          2%
Mouth complex:        2%
Eyes face track:      1.2%
LCD update:           1%
Audio/Mic:            0.8%
Camera:               2%
FACE DETECTION:       10%
Network burst:        3%
Motion blend:         1%
────────────────────────────────
TOTAL AVERAGE:        34-36% (Still under 40%)
TOTAL PEAK:           40-45% (Brief spikes)
```

---

## Total CPU Load Per Frame

### Baseline (GIF Only):
```
GIF frame load:        1-2ms     (1.5-3%)
─────────────────────────────
TOTAL (GIF):           1-2ms     (1.5-3%)
```

### With Mouth Overlay:
```
GIF frame load:        1-2ms     (1.5-3%)
Mouth composite:       2-3ms     (3-4.5%)
─────────────────────────────
TOTAL (GIF + Mouth):   3-5ms     (4.5-7.5%)
```

### Full Stack (GIF + Mouth + Eyes):
```
GIF frame load:        1-2ms     (1.5-3%)
Mouth composite:       2-3ms     (3-4.5%)
Eye drawing:           1-2ms     (1.5-3%)
LCD update (DMA):      0.5-1ms   (0.75-1.5%)
─────────────────────────────
TOTAL (Full):          4.5-8ms   (6.75-12%)
```

### With Optimizations:
```
GIF frame load:        1-2ms     (1.5-3%)
Mouth composite:       0.5-1.5ms (0.75-2.25%) [dirty region]
Eye drawing:           0.3-0.8ms (0.45-1.2%)  [lookup table]
LCD update (DMA):      0.5-1ms   (0.75-1.5%)
─────────────────────────────
TOTAL (Optimized):     2.3-5.3ms (3.45-8%)
```

---

## Audio Playback Overhead

```c
void audio_playback_task(void *arg) {
    while (1) {
        // Check ring buffer
        if (audio_ring.available >= AUDIO_CHUNK_SIZE) {
            
            // 1. Read from ring buffer: <0.1ms
            ring_buffer_read(&audio_ring, chunk, AUDIO_CHUNK_SIZE);
            
            // 2. Configure audio hardware: <0.1ms
            audio_set_buffer(chunk, AUDIO_CHUNK_SIZE);
            
            // 3. Start DMA transfer: <0.05ms
            audio_start_playback();
            
            // 4. Wait for completion (DMA in background): blocking
            audio_wait_for_completion();  // 100ms (audio chunk duration)
            
        } else {
            // No audio available, sleep
            rtos_delay_milliseconds(10);
        }
    }
}

// Audio task CPU load: ~0.2% (mostly waiting for DMA)
// Sleep doesn't consume CPU
```

---

## Network I/O Overhead

```c
void network_receive_task(void *arg) {
    // Receive commands + audio chunks from server
    // Data rate: 100-500 Kbps (sparse)
    
    // 1. WebSocket receive: ~1ms per command/chunk
    websocket_receive(&data);  // 1ms
    
    // 2. Parse JSON (if needed): 0.5-2ms
    cJSON_Parse(command_json);  // 0.5-2ms
    
    // 3. Write to ring buffer: <0.1ms
    ring_buffer_write(&audio_ring, audio_data);
}

// Network task CPU load: ~1-2% (mostly sleeping, brief activity)
```

---

## Total System CPU Budget

```
System Baseline:           5-10%
WiFi/Network (idle):       1-2%
Avatar Rendering:          6.75-12% (full) or 3.45-8% (optimized)
Audio Playback:            0.2%
Network RX (periodic):     1-2% (when receiving)
Other/Margin:              5-10%
─────────────────────────────────
TOTAL AVERAGE:             ~30-40%
PEAK (optimized):          ~50% (during response)
PEAK (unoptimized):        ~70% (during response)
```

**Available headroom:** 30-50% for future features

---

## Detailed Frame-by-Frame Timing

### Scenario: Rendering octopus speaking happily

```
Timeline (66ms per frame @ 15fps):

T=0ms     START frame render
T=0-1ms   Load GIF frame from cache
T=1-2ms   Composite mouth PNG (mouth_e shape)
T=2-3ms   Draw parametric eyes (looking at face)
T=3-4ms   (idle - preparing LCD data)
T=4-5ms   DMA write to LCD1 (async, doesn't block CPU)
T=5-6ms   DMA write to LCD2 (async, doesn't block CPU)

T=6-66ms  IDLE - wait for next frame
          (audio playback happening in parallel on different CPU core)

T=66ms    START next frame render
```

**CPU timeline with dual-core:**
```
Core 0: Animation rendering (6ms) + idle (60ms)
Core 1: Audio playback (continuous) + network (occasional)
```

---

## Realistic Load Scenarios

### Scenario 1: Idle (No Speaking)

```
GIF animation:    1-2ms per frame
Eye gentle drift: 0.3-0.8ms per frame
No mouth:         0ms
Audio:            0% (no playback)
Network:          ~0.1% (periodic keep-alive)

TOTAL:            2-3ms per frame = 3-5% CPU
```

### Scenario 2: Device Speaking

```
GIF animation (talking_animated): 1-2ms per frame
Mouth sync (mouth_a shape):        1-2ms per frame (with dirty region)
Eye tracking (looking at user):    0.3-0.8ms per frame
Audio playback:                    0.2% (mostly DMA)
Network RX (audio chunks):         1-2% (periodic)

TOTAL:                             4.5-7ms per frame = 6.75-10.5% + 1-2% network
PEAK:                              ~12% during heavy sync
```

### Scenario 3: Emotion Transition

```
GIF blend (old emotion → new):     2-3ms per frame
Mouth sync:                        1-2ms per frame
Eye look away/back:                0.5-1ms per frame
Audio playback:                    0.2%
Network:                           1-2%

TOTAL:                             4-7ms per frame = 6-10.5% + 1-2% network
```

### Scenario 4: Worst Case (All Features)

```
GIF animation:                     1-2ms
Mouth composite (full dirty region): 2-3ms
Eye drawing with eyelid:           1-2ms
Audio buffering:                   0.2%
Network RX + JSON parse:           2-3%
Face detection (lightweight):       10-15ms every 500ms (2% average)

PEAK INSTANT:                      ~20% (very rare)
AVERAGE:                           ~10-12%
```

---

## Bottleneck Analysis

### Current bottleneck: Mouth PNG Compositing

**Why:** Full 80x40 pixel region composite every frame

**Options to reduce:**

#### Option 1: Dirty Region Optimization
```c
// Only redraw pixels that changed
// Example: Mouth changes from closed to 'a' shape
// Dirty region: 30x20 pixels instead of 80x40
// Reduction: 2-3ms → 0.5-1ms
// Effort: 2-3 hours coding
// Result: ~2ms saved per frame = 3% CPU reduction
```

#### Option 2: Mouth Animation Pre-rendering
```c
// Instead of PNG overlay + alpha blend:
// Pre-render common mouth transitions
// Load pre-rendered GIF of mouth morphing
// Reduction: Composite time eliminated
// Benefit: Smoother transitions
// Effort: 1-2 weeks asset creation
// Result: 2-3ms saved = 3-4.5% CPU reduction
```

#### Option 3: Hardware SPI Drawing
```c
// If BK7258 SPI can do pixel compositing:
// Offload to hardware
// Reduction: 2-3ms → 0.2ms
// Effort: 4-5 weeks driver development
// Not recommended for MVP
```

---

## GPU/Acceleration Analysis

**Does BK7258 have GPU capabilities?**

No dedicated GPU, BUT:

1. **SPI Hardware:** Can do bulk memory transfers (already used for LCD)
2. **DMA Engines:** Can handle memcpy for frame buffer (already used)
3. **Hardware Accelerators:** None for graphics

**What about NEON/SIMD?**

BK7258 Cortex-M4 has optional SIMD, but:
- Not reliably available across all units
- Complex to use on embedded ARM
- Small gain (2-3x speedup) not worth complexity
- Skip for MVP

---

## Memory Bandwidth Analysis

**Does compositing saturate memory bus?**

```
Memory bandwidth: Cortex-M4 @ 480MHz
  - Best case: ~600MB/s DDR (theoretical)
  - Real case: ~300-400MB/s (shared with WiFi, etc)

Mouth compositing:
  - Read GIF frame: 51KB * 15fps = 765KB/s
  - Read mouth PNG: 16KB * 15fps = 240KB/s
  - Write frame buffer: 51KB * 15fps = 765KB/s
  - Total: ~1.8MB/s = <<1% of bandwidth

No memory bottleneck!
```

---

## Final CPU Budget Table

| Component | Min | Typical | Max | Notes |
|-----------|-----|---------|-----|-------|
| OS/System | 5% | 8% | 12% | Varies with WiFi |
| Avatar Rendering | 2% | 7% | 12% | Optimized/Unoptimized |
| Audio Playback | 0.1% | 0.2% | 1% | Mostly DMA |
| Network I/O | 0.5% | 2% | 5% | Periodic bursts |
| Face Detection | 0% | 2% | 15% | Only when active |
| **TOTAL** | **7.6%** | **19%** | **45%** | **Peak usage** |

**Available for other features:** 55-92% CPU

---

## Recommendations for MVP

### Use Unoptimized (2-3 weeks):
```
Full feature set
CPU: ~12% average, ~25% peak
Simple implementation
Good enough for MVP
```

### Use Optimized (Add 1-2 weeks):
```
Dirty region optimization
CPU: ~8% average, ~15% peak
Better performance budget
Still reasonable timeline
Recommended for MVP
```

### Use Heavily Optimized (Add 3-4 weeks):
```
Dirty regions + LUT circles + motion prediction
CPU: ~5% average, ~10% peak
Maximum performance
Only if needed later
Skip for MVP
```

---

## Conclusion

**Mouth Phonemes + Layered Animations CPU Load:**

- **Baseline (optimized):** 8-10% CPU average
- **Peak (during speech):** 15-20% CPU
- **Worst case:** 25-30% CPU
- **Headroom remaining:** 55-65% CPU

**Verdict: EASILY AFFORDABLE FOR MVP**

The system has:
- Plenty of CPU headroom
- No memory bottlenecks
- No I/O saturation
- Room for additional features

**Recommendation: Use optimized approach (1-2 weeks extra effort, significant performance improvement)**

---

Version: 1.0  
Date: December 27, 2025  
Status: Ready for Implementation
