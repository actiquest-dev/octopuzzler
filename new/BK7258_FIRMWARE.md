# BK7258 Firmware Development Guide

## Quick Start

BK7258 runs C code. Firmware handles:
- Audio capture (16kHz PCM)
- WebSocket connection
- Animation rendering (GIF + overlays)
- Display output

## Hardware

```
BK7258 SoC
├─ Dual-core ARM CPU
├─ 256MB RAM
├─ 4MB internal flash
├─ WiFi 2.4GHz 802.11
├─ 16-bit ADC microphone
├─ SPI/I2C interfaces
└─ GPIO pins
```

## Code Structure

### 1. main.c - Event Loop

```c
#include "audio_capture.h"
#include "websocket_client.h"
#include "animation_render.h"
#include "display_driver.h"

int main() {
    init_wifi();
    init_audio();
    init_display();
    init_websocket();
    
    while(1) {
        // Get 50ms audio chunk
        uint16_t* audio_chunk = audio_capture_chunk();
        
        // Send via WebSocket (async)
        websocket_send_audio(audio_chunk, 800);
        
        // Check for commands from backend
        animation_command_t cmd = websocket_receive_command();
        if(cmd.valid) {
            render_animation(cmd);
        }
        
        // Small delay
        sleep_ms(50);
    }
    return 0;
}
```

### 2. audio_capture.c - Audio Input

```c
#include "audio_capture.h"
#include <stdint.h>

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256  // Ring buffer chunks
#define CHUNK_SIZE 800   // 50ms @ 16kHz

typedef struct {
    uint16_t samples[CHUNK_SIZE];
    int write_pos;
    int read_pos;
} ring_buffer_t;

ring_buffer_t rb;

void init_audio() {
    // Configure ADC
    configure_adc(SAMPLE_RATE, 16);
    
    // Start continuous capture
    start_adc_dma(&rb.samples[0], sizeof(rb.samples));
    
    rb.write_pos = 0;
    rb.read_pos = 0;
}

uint16_t* audio_capture_chunk() {
    // Wait until chunk ready
    while(rb.write_pos == rb.read_pos) {
        sleep_ms(1);
    }
    
    uint16_t* chunk = &rb.samples[rb.read_pos * CHUNK_SIZE];
    rb.read_pos = (rb.read_pos + 1) % BUFFER_SIZE;
    
    return chunk;
}

// ADC DMA interrupt
void adc_dma_isr() {
    rb.write_pos = (rb.write_pos + 1) % BUFFER_SIZE;
}
```

### 3. websocket_client.c - Network Communication

```c
#include "websocket_client.h"
#include "lwip/api.h"

#define BACKEND_IP "192.168.1.100"
#define BACKEND_PORT 8000
#define DEVICE_ID "device-001"

typedef struct {
    struct netconn* conn;
    char recv_buffer[4096];
    int recv_pos;
} websocket_t;

websocket_t ws;

void init_websocket() {
    // Connect to backend
    ws.conn = netconn_new(NETCONN_TCP);
    netconn_connect(ws.conn, IP_ADDR_ANY, BACKEND_PORT);
    
    // Send WebSocket handshake
    char handshake[512];
    sprintf(handshake,
        "GET /ws/%s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n",
        DEVICE_ID, BACKEND_IP, BACKEND_PORT
    );
    
    netconn_write(ws.conn, handshake, strlen(handshake), NETCONN_COPY);
}

void websocket_send_audio(uint16_t* audio, int size) {
    // Create JSON message
    char json[2048];
    char hex_audio[1600];
    
    // Convert audio to hex
    for(int i = 0; i < size; i++) {
        sprintf(&hex_audio[i*4], "%04x", audio[i]);
    }
    
    sprintf(json,
        "{\"audio\":\"%s\",\"sample_rate\":16000}",
        hex_audio
    );
    
    // Send as WebSocket frame
    send_websocket_frame(ws.conn, json, strlen(json));
}

animation_command_t websocket_receive_command() {
    animation_command_t cmd;
    cmd.valid = 0;
    
    // Check if data available
    struct netbuf* buf = netconn_recv(ws.conn);
    if(!buf) return cmd;
    
    // Parse WebSocket frame
    char* data = buf->p->payload;
    int len = buf->p->len;
    
    // Skip WebSocket header (2 bytes)
    if(len > 2) {
        // Parse JSON command
        // cmd = parse_animation_json(&data[2], len-2);
    }
    
    netbuf_delete(buf);
    return cmd;
}
```

### 4. animation_render.c - Display Output

```c
#include "animation_render.h"
#include "gif_loader.h"
#include "display_driver.h"

#define DISPLAY_WIDTH 240
#define DISPLAY_HEIGHT 240

typedef struct {
    uint8_t* body_gif;
    uint8_t* mouth_png;
    uint8_t* eyes_json;
    int current_frame;
} animation_state_t;

animation_state_t anim;

void init_animation() {
    // Load assets
    anim.body_gif = load_gif("/avatar_body.gif");
    anim.mouth_png = load_png("/mouth_shapes.png");
    anim.eyes_json = load_json("/eyes.json");
    anim.current_frame = 0;
}

void render_animation(animation_command_t cmd) {
    // Get current body frame
    uint8_t* frame = get_gif_frame(anim.body_gif, anim.current_frame);
    
    // Draw body
    display_draw_frame(frame, 0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT);
    
    // Draw mouth overlay
    int mouth_idx = cmd.phoneme_idx;  // 0-7
    uint8_t* mouth = get_mouth_shape(anim.mouth_png, mouth_idx);
    display_draw_sprite(mouth, 100, 140, 40, 40);
    
    // Draw eyes
    int eye_x = cmd.eye_x;
    int eye_y = cmd.eye_y;
    float blink = cmd.blink_amount;  // 0-1
    draw_eyes(eye_x, eye_y, blink);
    
    // Update frame counter
    anim.current_frame = (anim.current_frame + 1) % get_gif_frame_count(anim.body_gif);
    
    // Flip display
    display_flip();
}

void draw_eyes(int x, int y, float blink) {
    // Simple parametric eyes
    int eye_width = 20;
    int eye_height = (int)(20 * blink);
    
    if(eye_height > 0) {
        display_fill_ellipse(100 + x, 80 + y, eye_width, eye_height, COLOR_WHITE);
        display_fill_ellipse(180 + x, 80 + y, eye_width, eye_height, COLOR_WHITE);
    }
}
```

### 5. display_driver.c - Hardware Interface

```c
#include "display_driver.h"
#include "hw_spi.h"

#define DC_PIN 10
#define CS_PIN 11
#define RST_PIN 12

void init_display() {
    // Configure SPI
    spi_init(SPI_MASTER, 10000000);  // 10MHz
    
    // Configure control pins
    gpio_config_output(DC_PIN);
    gpio_config_output(CS_PIN);
    gpio_config_output(RST_PIN);
    
    // Reset display
    gpio_write(RST_PIN, 0);
    sleep_ms(10);
    gpio_write(RST_PIN, 1);
    sleep_ms(10);
    
    // Send initialization commands
    send_command(0x36);  // MADCTL
    send_data(0x00);
    
    send_command(0x3A);  // COLMOD
    send_data(0x05);     // 16-bit color
    
    send_command(0x29);  // DISPON
}

void display_draw_frame(uint8_t* frame, int x, int y, int w, int h) {
    set_window(x, y, x + w - 1, y + h - 1);
    
    gpio_write(DC_PIN, 1);  // Data mode
    gpio_write(CS_PIN, 0);
    
    spi_write(frame, w * h * 2);  // 16-bit color
    
    gpio_write(CS_PIN, 1);
}

void display_flip() {
    // Called after each frame
}

void send_command(uint8_t cmd) {
    gpio_write(DC_PIN, 0);
    gpio_write(CS_PIN, 0);
    spi_write(&cmd, 1);
    gpio_write(CS_PIN, 1);
}

void send_data(uint8_t data) {
    gpio_write(DC_PIN, 1);
    gpio_write(CS_PIN, 0);
    spi_write(&data, 1);
    gpio_write(CS_PIN, 1);
}

void set_window(int x1, int y1, int x2, int y2) {
    // Column address
    send_command(0x2A);
    send_data(x1 >> 8);
    send_data(x1 & 0xFF);
    send_data(x2 >> 8);
    send_data(x2 & 0xFF);
    
    // Row address
    send_command(0x2B);
    send_data(y1 >> 8);
    send_data(y1 & 0xFF);
    send_data(y2 >> 8);
    send_data(y2 & 0xFF);
}
```

## Build

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(bk7258_octopus)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-Wall -O2 -march=armv7-m")

# Source files
file(GLOB SOURCES src/*.c)

# Executable
add_executable(firmware ${SOURCES})

# Link libraries
target_link_libraries(firmware
    lwip
    freertos
    bk7258_hal
    m
)

# Output
add_custom_command(TARGET firmware POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
    COMMAND echo "Firmware size: $(stat -f%z firmware.bin) bytes"
)
```

### Compile

```bash
mkdir build
cd build
cmake ..
make
```

## Flash

```bash
# Using BK7258 flasher
./bk7258_flasher firmware.bin

# Or via serial
python3 bk7258_serial_flash.py --port /dev/ttyUSB0 firmware.bin
```

## Testing

### Local Testing (no hardware)

```python
# device_simulator.py
import asyncio
import websockets
import json
import random

async def simulate_device():
    uri = "ws://localhost:8000/ws/device-001"
    
    async with websockets.connect(uri) as websocket:
        for i in range(100):
            # Simulate audio chunk
            audio = [random.randint(0, 65535) for _ in range(800)]
            audio_hex = ''.join(f'{x:04x}' for x in audio)
            
            msg = json.dumps({
                "audio": audio_hex,
                "sample_rate": 16000
            })
            
            await websocket.send(msg)
            
            # Receive command
            response = await websocket.recv()
            cmd = json.loads(response)
            print(f"Command: {cmd}")
            
            await asyncio.sleep(0.05)

asyncio.run(simulate_device())
```

## Key Metrics

- Audio capture latency: ~20ms
- WebSocket round-trip: 50-100ms
- Display refresh: 66ms (15fps)
- CPU usage: 10-30%
- Memory: 256MB RAM, ~100MB for assets
- Power: 200-250mW

## Common Issues

**WiFi drops**
```c
// Reconnect on disconnect
void wifi_event_handler(wifi_event_t event) {
    if(event == WIFI_EVENT_DISCONNECTED) {
        wifi_reconnect();
    }
}
```

**Audio glitches**
```c
// Use larger ring buffer
#define BUFFER_SIZE 512  // Increased from 256
```

**Display flicker**
```c
// Double buffer
uint8_t buffer1[240*240*2];
uint8_t buffer2[240*240*2];
// Alternate between them
```

## Next Steps

1. Flash firmware to BK7258
2. Connect WiFi
3. Start backend server
4. Test audio streaming
5. Verify animation sync
6. Deploy to production

Total time: ~1 week for full integration testing
