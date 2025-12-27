# Audio Processing Pipeline

## Audio Capture (BK7258)

### Microphone Setup

```c
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define BIT_DEPTH 16
#define CHUNK_SIZE 800  // 50ms @ 16kHz

void init_audio() {
    // Configure ADC
    adc_config_t config = {
        .sample_rate = SAMPLE_RATE,
        .bit_depth = BIT_DEPTH,
        .channels = CHANNELS,
        .buffer_size = CHUNK_SIZE * 4
    };
    
    adc_init(&config);
    adc_start_continuous();
}
```

### Ring Buffer

```c
#define BUFFER_CHUNKS 256
#define TOTAL_SIZE (CHUNK_SIZE * BUFFER_CHUNKS)

typedef struct {
    uint16_t samples[TOTAL_SIZE];
    volatile int write_idx;
    volatile int read_idx;
    int chunk_count;
} ring_buffer_t;

ring_buffer_t rb;

// ADC interrupt - fills buffer
void adc_isr() {
    int chunk_idx = rb.write_idx % BUFFER_CHUNKS;
    uint16_t* chunk = &rb.samples[chunk_idx * CHUNK_SIZE];
    
    // DMA fills chunk
    dma_transfer_adc(chunk, CHUNK_SIZE);
    
    rb.write_idx++;
}

// Main loop - reads from buffer
uint16_t* get_audio_chunk() {
    while(rb.write_idx == rb.read_idx) {
        sleep_ms(1);
    }
    
    int chunk_idx = rb.read_idx % BUFFER_CHUNKS;
    uint16_t* chunk = &rb.samples[chunk_idx * CHUNK_SIZE];
    
    rb.read_idx++;
    return chunk;
}
```

### Preprocessing

Normalize audio before sending:

```c
void normalize_audio(uint16_t* input, float* output, int size) {
    // Find max value
    uint16_t max_val = 0;
    for(int i = 0; i < size; i++) {
        if(input[i] > max_val) max_val = input[i];
    }
    
    // Normalize to [-1, 1]
    float scale = 1.0f / max_val;
    for(int i = 0; i < size; i++) {
        int16_t signed_sample = (int16_t)input[i];
        output[i] = signed_sample * scale / 32768.0f;
    }
}
```

## WebSocket Streaming

### Connection

```c
#define BACKEND_HOST "192.168.1.100"
#define BACKEND_PORT 8000
#define DEVICE_ID "device-001"

void init_websocket() {
    // Create TCP connection
    struct netconn* conn = netconn_new(NETCONN_TCP);
    netconn_connect(conn, 
        (ip_addr_t*)inet_addr(BACKEND_HOST),
        BACKEND_PORT
    );
    
    // Send WebSocket upgrade request
    char request[512];
    sprintf(request,
        "GET /ws/%s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n",
        DEVICE_ID, BACKEND_HOST, BACKEND_PORT
    );
    
    netconn_write(conn, request, strlen(request), NETCONN_COPY);
}
```

### Audio Transmission

Send 50ms audio chunks as JSON:

```python
# Backend receiver
import asyncio
import websockets
import json

async def handle_audio_stream(websocket, path):
    device_id = path.split('/')[-1]
    print(f"Device connected: {device_id}")
    
    async for message in websocket:
        data = json.loads(message)
        
        audio_hex = data['audio']
        sample_rate = data['sample_rate']
        
        # Convert hex to bytes
        audio_bytes = bytes.fromhex(audio_hex)
        
        # Process audio
        result = await process_audio(audio_bytes, sample_rate)
        
        # Send commands back
        response = {
            "phoneme_idx": result.phoneme,
            "eye_x": result.eye_x,
            "eye_y": result.eye_y,
            "blink_amount": result.blink
        }
        
        await websocket.send(json.dumps(response))
```

### Latency Optimization

```python
# Use binary format instead of JSON for lower latency
import struct
import asyncio

async def handle_binary_audio(websocket, path):
    async for message in websocket:
        # Binary format:
        # 4 bytes: sample_rate
        # 2 bytes: chunk_size
        # N bytes: audio data
        
        sample_rate = struct.unpack('>I', message[0:4])[0]
        chunk_size = struct.unpack('>H', message[4:6])[0]
        audio_data = message[6:6+chunk_size*2]
        
        # Process immediately
        result = await processor.process_audio(audio_data)
        
        # Send binary response
        response = struct.pack('>ffff',
            result.phoneme_idx,
            result.eye_x,
            result.eye_y,
            result.blink_amount
        )
        
        await websocket.send(response)
```

## Backend Processing Pipeline

### FastAPI Server

```python
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import asyncio

app = FastAPI()

# Load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load SenseVoice
    global audio_processor
    audio_processor = SenseVoiceProcessor()
    
    # Load LLM
    global llm
    llm = QwenLLM()
    
    # Load TTS
    global tts
    tts = TTSEngine()
    
    # Load animation sync
    global animation_gen
    animation_gen = AnimationCommandGenerator()
    
    yield
    
    # Cleanup on shutdown

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    
    print(f"Device {device_id} connected")
    
    try:
        while True:
            # Receive audio
            message = await websocket.receive_text()
            data = json.loads(message)
            audio_hex = data['audio']
            audio_bytes = bytes.fromhex(audio_hex)
            
            # Process: STT + Emotion + Language
            audio_result = await audio_processor.process_audio(
                audio_bytes,
                sample_rate=16000
            )
            
            # LLM response
            response_text = await llm.generate(
                text=audio_result.text,
                emotion=audio_result.emotion,
                language=audio_result.language
            )
            
            # TTS synthesis
            tts_audio, duration_ms = await tts.synthesize(
                response_text,
                emotion=audio_result.emotion
            )
            
            # Animation commands
            animation_commands = await animation_gen.generate(
                text=response_text,
                duration_ms=duration_ms,
                emotion=audio_result.emotion
            )
            
            # Send back to device
            response = {
                "audio": tts_audio.hex(),
                "animations": animation_commands.dict(),
                "duration_ms": duration_ms
            }
            
            await websocket.send_json(response)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print(f"Device {device_id} disconnected")
```

## Audio Quality Metrics

### Loudness Normalization

```python
import numpy as np

def normalize_loudness(audio, target_loudness_db=-20):
    """Normalize to target loudness (LUFS)"""
    
    # Calculate current loudness
    audio = np.array(audio, dtype=np.float32)
    energy = np.mean(audio ** 2)
    current_loudness = 10 * np.log10(energy + 1e-9)
    
    # Calculate gain
    gain_db = target_loudness_db - current_loudness
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain with clipping
    normalized = np.clip(audio * gain_linear, -1.0, 1.0)
    
    return normalized
```

### Voice Activity Detection (VAD)

```python
def voice_activity_detection(audio, threshold=0.02):
    """Detect if audio contains speech"""
    
    energy = np.mean(audio ** 2)
    
    if energy > threshold:
        return True  # Voice detected
    else:
        return False  # Silence
```

### Noise Reduction

```python
import numpy as np

def reduce_noise(audio, noise_profile=None):
    """Simple spectral subtraction"""
    
    # FFT
    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    
    # Noise gate
    noise_gate = np.mean(magnitude) * 0.5
    magnitude[magnitude < noise_gate] = noise_gate
    
    # Reconstruct
    cleaned = magnitude * np.exp(1j * phase)
    audio_cleaned = np.fft.irfft(cleaned)
    
    return audio_cleaned
```

## Buffering & Flow Control

### Adaptive Buffering

```python
class AdaptiveBuffer:
    """Adjust buffer size based on latency"""
    
    def __init__(self, target_latency_ms=100):
        self.buffer = []
        self.target_latency = target_latency_ms
        self.chunk_time_ms = 50
        self.buffer_size = 2  # Start with 100ms
    
    def add_chunk(self, chunk):
        self.buffer.append(chunk)
    
    def get_chunk(self):
        if len(self.buffer) >= self.buffer_size:
            return self.buffer.pop(0)
        return None
    
    def adjust_buffer_size(self, latency_ms):
        """Increase buffer if latency too low"""
        if latency_ms < self.target_latency * 0.8:
            self.buffer_size = min(self.buffer_size + 1, 5)
        elif latency_ms > self.target_latency * 1.2:
            self.buffer_size = max(self.buffer_size - 1, 1)
```

### Backpressure Handling

```python
async def process_audio_stream(websocket):
    buffer = AdaptiveBuffer()
    
    async def receive_audio():
        """Receive audio from device"""
        async for message in websocket:
            data = json.loads(message)
            chunk = bytes.fromhex(data['audio'])
            buffer.add_chunk(chunk)
    
    async def process_chunks():
        """Process chunks as they arrive"""
        while True:
            chunk = buffer.get_chunk()
            if chunk:
                result = await processor.process_audio(chunk)
                # Send back
                await websocket.send_json({"result": result})
            else:
                await asyncio.sleep(0.01)  # Wait for more data
    
    # Run both concurrently
    receive_task = asyncio.create_task(receive_audio())
    process_task = asyncio.create_task(process_chunks())
    
    await asyncio.gather(receive_task, process_task)
```

## Error Handling

### Corrupt Audio Detection

```python
def is_audio_valid(audio_bytes, expected_size=1600):
    """Check if audio chunk is valid"""
    
    if len(audio_bytes) != expected_size:
        return False
    
    # Check for all zeros (dead microphone)
    if np.all(audio_bytes == 0):
        return False
    
    # Check for all ones (overflow)
    if np.all(audio_bytes == 255):
        return False
    
    return True
```

### Retry Logic

```python
async def process_audio_with_retry(
    audio_bytes,
    max_retries=3,
    backoff_ms=100
):
    """Process with automatic retry"""
    
    for attempt in range(max_retries):
        try:
            result = await processor.process_audio(audio_bytes)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_ms / 1000)
                backoff_ms *= 2
            else:
                return None  # Failed after retries
```

## Performance Optimization

### Batch Processing

```python
async def batch_process_audio(audio_chunks, batch_size=4):
    """Process multiple chunks in batch"""
    
    results = []
    
    for i in range(0, len(audio_chunks), batch_size):
        batch = audio_chunks[i:i+batch_size]
        
        # Stack into batch tensor
        batch_audio = torch.stack([
            torch.from_numpy(chunk) for chunk in batch
        ])
        
        # Process batch
        with torch.no_grad():
            batch_results = model(batch_audio)
        
        results.extend(batch_results)
    
    return results
```

### GPU Memory Management

```python
def optimize_gpu_memory():
    """Minimize GPU memory usage"""
    
    import torch
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Use half precision
    torch.set_default_dtype(torch.float16)
    
    # Enable memory efficient attention
    torch.backends.cuda.matmul.allow_tf32 = True
```

## Monitoring

### Audio Quality Metrics

```python
class AudioMonitor:
    """Track audio quality in real-time"""
    
    def __init__(self):
        self.signal_power = []
        self.noise_floor = []
        self.snr_values = []
    
    def analyze(self, audio):
        signal_power = np.mean(audio ** 2)
        self.signal_power.append(signal_power)
        
        # Estimate noise floor
        sorted_power = np.sort(np.abs(audio))
        noise_estimate = np.mean(sorted_power[:len(sorted_power)//4])
        self.noise_floor.append(noise_estimate)
        
        # Calculate SNR
        snr = 10 * np.log10((signal_power + 1e-9) / (noise_estimate + 1e-9))
        self.snr_values.append(snr)
    
    def get_stats(self):
        return {
            "avg_signal_power": np.mean(self.signal_power),
            "avg_snr_db": np.mean(self.snr_values),
            "min_snr_db": np.min(self.snr_values)
        }
```

## Summary

- **Capture:** 16kHz, 16-bit, continuous ring buffer
- **Transmission:** WebSocket, 50ms chunks, JSON format
- **Processing:** SenseVoiceSmall (STT + Emotion + Language)
- **Buffering:** Adaptive based on latency
- **Quality:** Normalization, VAD, noise reduction
- **Reliability:** Retry logic, error detection
- **Monitoring:** Real-time SNR and quality metrics
