# Debugging Guide

Complete guide for diagnosing and fixing issues in WowCube Octopus Avatar.

## Debugging Levels

```
Level 1: Log Inspection          (5-10 min)
Level 2: Local Testing            (15-30 min)
Level 3: Performance Profiling     (30-60 min)
Level 4: Hardware Testing          (1-2 hours)
Level 5: Nuclear Option - Reset    (depends)
```

## Level 1: Log Inspection

### Enable Debug Logging

```python
# In main.py
import logging

# Set debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Backend started")
```

### View Logs

```bash
# Real-time logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log

# Search for specific module
grep STT logs/app.log

# Last N lines
tail -100 logs/app.log

# Full stack trace
grep -A 20 "Traceback" logs/app.log
```

### Log Format

```
2025-12-27 10:30:45.123 - stt_pipeline - DEBUG - Loading SenseVoiceSmall
2025-12-27 10:30:46.500 - stt_pipeline - INFO - ✓ Model loaded
2025-12-27 10:30:47.100 - stt_pipeline - WARNING - Audio quality: 0.75 (low)
2025-12-27 10:30:48.200 - stt_pipeline - ERROR - STT failed: CUDA out of memory
```

## Common Issues & Solutions

### 1. Audio Input Issues

**Problem:** "Audio not being captured" or "Silent audio"

```python
# Debug script
import pyaudio
import numpy as np

def test_microphone():
    p = pyaudio.PyAudio()
    
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"  {i}: {info['name']}")
    
    # Test default mic
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=800
    )
    
    print("Recording 3 seconds...")
    frames = []
    for _ in range(3 * 16000 // 800):
        data = stream.read(800)
        audio = np.frombuffer(data, dtype=np.float32)
        frames.append(audio)
        
        # Check if audio is silent
        volume = np.max(np.abs(audio))
        print(f"  Volume: {volume:.3f}")
        
        if volume < 0.01:
            print("  ⚠️ WARNING: Very quiet!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Check result
    all_audio = np.concatenate(frames)
    print(f"Total samples: {len(all_audio)}")
    print(f"Max amplitude: {np.max(np.abs(all_audio)):.3f}")
    
    if np.max(np.abs(all_audio)) < 0.1:
        print("❌ Microphone is silent!")
        print("Solutions:")
        print("  1. Check system audio input settings")
        print("  2. Try different audio device: stream = p.open(input_device_index=N)")
        print("  3. Check microphone permissions: sudo usermod -aG audio $USER")
        print("  4. Restart PulseAudio: pulseaudio -k")
    else:
        print("✓ Microphone works!")

test_microphone()
```

**Checklist:**
```
☐ Microphone plugged in?
☐ Microphone not muted?
☐ System volume > 10%?
☐ Input device set correctly?
☐ ALSA/PulseAudio running?
☐ Correct user permissions (audio group)?
```

### 2. STT (Speech-to-Text) Issues

**Problem:** "STT returning empty text" or "Low confidence"

```python
# Debug STT
async def debug_stt():
    from stt_pipeline import STTPipeline
    import numpy as np
    
    pipeline = STTPipeline()
    
    # Test 1: Silence
    print("Test 1: Silence")
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    result = await pipeline.process_chunk(silence)
    print(f"  Text: '{result['text']}'")
    print(f"  Confidence: {max(result.get('emotion_scores', {}).values()):.2f}")
    
    # Test 2: Noise
    print("\nTest 2: White noise")
    noise = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
    result = await pipeline.process_chunk(noise)
    print(f"  Text: '{result['text']}'")
    
    # Test 3: Check model loading
    print("\nTest 3: Model status")
    print(f"  Device: {pipeline.device}")
    print(f"  Model: {pipeline.model_name}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

asyncio.run(debug_stt())
```

**Solutions:**

```
Issue: Empty text output
├─ Check: Audio is not silent (volume > 0.05)
├─ Check: Audio is 16kHz sample rate
├─ Check: Model is loaded (print model info)
├─ Check: CUDA memory available (>1.5GB)
└─ Fix: Reduce sample rate, use CPU instead of GPU

Issue: Low confidence scores
├─ Check: Audio quality (SNR > 20dB)
├─ Check: Speech is clear (no background noise)
├─ Check: Language matches model (English default)
└─ Fix: Use louder, clearer audio

Issue: Wrong language detected
├─ Check: Audio actually is that language
├─ Check: Model supports that language (50+ languages)
└─ Fix: Manually specify language if needed
```

### 3. LLM (Language Model) Issues

**Problem:** "LLM returning garbage" or "Timeout"

```python
# Debug LLM
async def debug_llm():
    from llm_integration import QwenLLM
    
    llm = QwenLLM()
    
    # Test 1: Simple prompt
    print("Test 1: Simple prompt")
    result = await llm.generate_response(
        user_text="Hi",
        emotion="neutral"
    )
    print(f"  Response: {result['text']}")
    print(f"  Success: {result['success']}")
    
    # Test 2: Emotion handling
    print("\nTest 2: Emotion handling")
    for emotion in ["happy", "sad", "angry"]:
        result = await llm.generate_response(
            user_text="I'm sad",
            emotion=emotion
        )
        print(f"  {emotion}: {result['text'][:50]}")
    
    # Test 3: Check VRAM
    print("\nTest 3: Memory status")
    import torch
    print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    print(f"  VRAM cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")

asyncio.run(debug_llm())
```

**Solutions:**

```
Issue: CUDA out of memory
├─ Check: Available VRAM > 16GB
├─ Check: No other processes using GPU (nvidia-smi)
├─ Check: Model size (Qwen-7B needs 14GB)
├─ Fix 1: Use smaller model (Qwen-1.8B)
├─ Fix 2: Quantize model (4-bit)
├─ Fix 3: Use CPU (slower)
└─ Fix 4: Use Modal.com (cloud)

Issue: Slow response (>5 seconds)
├─ Check: Model is on GPU (not CPU)
├─ Check: Batch size = 1
├─ Check: Token count < 100
├─ Fix: Profile with torch.profiler

Issue: Nonsense output
├─ Check: Prompt is correct
├─ Check: System prompt is set
├─ Check: Temperature is reasonable (0.5-0.9)
├─ Fix: Try different model version
└─ Fix: Add constraint tokens
```

### 4. TTS (Text-to-Speech) Issues

**Problem:** "No audio output" or "Bad audio quality"

```python
# Debug TTS
async def debug_tts():
    from tts_synthesis import TTSSynthesis
    import numpy as np
    
    tts = TTSSynthesis()
    
    # Test 1: Basic synthesis
    print("Test 1: Basic synthesis")
    audio, duration = await tts.synthesize("Hello world", emotion="happy")
    print(f"  Audio size: {len(audio)} bytes")
    print(f"  Duration: {duration}ms")
    print(f"  Is silent: {len(audio) == 0}")
    
    # Test 2: All emotions
    print("\nTest 2: Emotions")
    for emotion in ["happy", "sad", "angry", "neutral", "surprised"]:
        audio, duration = await tts.synthesize("Test", emotion=emotion)
        print(f"  {emotion}: {duration}ms")
    
    # Test 3: Audio quality
    print("\nTest 3: Audio analysis")
    audio, _ = await tts.synthesize("The quick brown fox", emotion="happy")
    audio_np = np.frombuffer(audio, dtype=np.int16)
    print(f"  Max amplitude: {np.max(np.abs(audio_np))}")
    print(f"  RMS: {np.sqrt(np.mean(audio_np**2))}")
    print(f"  Is clipping: {np.max(np.abs(audio_np)) > 30000}")

asyncio.run(debug_tts())
```

**Solutions:**

```
Issue: No audio output (0 bytes)
├─ Check: Model loaded (check VRAM)
├─ Check: Input text not empty
├─ Check: CUDA/GPU working
└─ Fix: Restart backend, clear cache

Issue: Audio too quiet
├─ Check: Amplitude < 1000
├─ Fix: Increase TTS volume scale
└─ Fix: Use audio amplification

Issue: Audio clipping/distorted
├─ Check: Max amplitude > 30000
├─ Fix: Reduce scale factor
└─ Fix: Normalize output

Issue: Audio has glitches/artifacts
├─ Check: Model not crashing
├─ Check: Memory not running out
├─ Fix: Try vocoder update
└─ Fix: Use simpler TTS model
```

### 5. Animation Sync Issues

**Problem:** "Mouth not syncing" or "Eyes not moving"

```python
# Debug animation
async def debug_animation():
    from animation_sync_service import AnimationCommandGenerator
    
    gen = AnimationCommandGenerator()
    
    # Test 1: Mouth sync
    print("Test 1: Mouth phoneme extraction")
    text = "Hello world"
    phonemes = gen.phoneme_extractor.extract_phonemes(text, 2000)
    print(f"  Text: {text}")
    print(f"  Phonemes: {len(phonemes)}")
    for p in phonemes[:5]:
        print(f"    {p.phoneme}: mouth={p.mouth_open:.2f}")
    
    # Test 2: Gaze planning
    print("\nTest 2: Gaze planning")
    for emotion in ["happy", "sad", "angry"]:
        gaze = gen.gaze_planner.plan_gaze(emotion)
        print(f"  {emotion}: x={gaze[0].x:.2f}, y={gaze[0].y:.2f}")
    
    # Test 3: Full animation generation
    print("\nTest 3: Full animation")
    from animation_sync_service import GeminiResponse
    response = GeminiResponse(
        text="That's wonderful!",
        audio_duration_ms=2000,
        emotion_detected="happy"
    )
    commands = await gen.generate_commands(response)
    print(f"  Duration: {commands.duration_ms}ms")
    print(f"  Phoneme timings: {len(commands.phonemes)}")
    print(f"  Gaze targets: {len(commands.gaze)}")
    print(f"  Sync markers: {len(commands.sync_markers)}")

asyncio.run(debug_animation())
```

**Solutions:**

```
Issue: Mouth not moving at all
├─ Check: Phoneme extraction working
├─ Check: PNG overlays exist (assets/mouth_*.png)
├─ Check: Animation service running
└─ Fix: Check if device received commands

Issue: Mouth out of sync
├─ Check: Phoneme timing correct
├─ Check: Audio duration matches animation
├─ Check: Frame rate consistent (15fps)
└─ Fix: Adjust timing offset

Issue: Eyes not moving
├─ Check: Gaze planner running
├─ Check: Face detection active (if tracking)
├─ Check: Eye rendering code works
└─ Fix: Check device firmware

Issue: Animation jittery/stuttering
├─ Check: Frame rate stable
├─ Check: Timing not drifting
├─ Check: No CPU spikes
└─ Fix: Reduce animation complexity
```

### 6. Network/WebSocket Issues

**Problem:** "WebSocket connection failing" or "Commands not arriving"

```python
# Debug WebSocket
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/test-device"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected")
            
            # Test 1: Send simple message
            print("\nTest 1: Send message")
            message = {
                "type": "test",
                "data": "hello"
            }
            await websocket.send(json.dumps(message))
            print(f"  Sent: {message}")
            
            # Test 2: Receive response
            print("\nTest 2: Receive response")
            try:
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5.0
                )
                print(f"  Received: {response}")
            except asyncio.TimeoutError:
                print("  ✗ No response (timeout)")
            
            # Test 3: Send audio
            print("\nTest 3: Send audio chunk")
            import numpy as np
            audio = np.zeros(1600, dtype=np.int16).tobytes()
            message = {
                "audio": audio.hex(),
                "sample_rate": 16000
            }
            await websocket.send(json.dumps(message))
            print(f"  Audio sent ({len(audio)} bytes)")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nChecklist:")
        print("  ☐ Backend running? (python backend/main.py)")
        print("  ☐ Port correct? (8000)")
        print("  ☐ Firewall allows? (sudo ufw allow 8000)")
        print("  ☐ CORS enabled?")

asyncio.run(test_websocket())
```

**Solutions:**

```
Issue: Connection refused
├─ Check: Backend running (ps aux | grep main.py)
├─ Check: Port 8000 open (netstat -an | grep 8000)
├─ Check: Host correct (localhost vs 127.0.0.1)
└─ Fix: Start backend, wait for "Server ready"

Issue: Connection timeout
├─ Check: Firewall rules (sudo ufw status)
├─ Check: Network connectivity (ping localhost)
└─ Fix: ufw allow 8000

Issue: Commands not arriving
├─ Check: WebSocket stays connected
├─ Check: No disconnect/reconnect spam
├─ Check: No error messages in logs
└─ Fix: Check network buffering

Issue: Messages corrupted
├─ Check: Audio hex encoding correct
├─ Check: JSON valid
└─ Fix: Use text protocol instead of hex
```

### 7. Device Simulator Issues

**Problem:** "Device not responding" or "Simulator crashes"

```python
# Debug device simulator
import sys
sys.path.insert(0, 'device/simulator')
from device_simulator import DeviceSimulator

async def debug_device():
    device = DeviceSimulator(device_id="debug-device")
    
    # Test 1: Audio capture
    print("Test 1: Audio capture")
    try:
        await device.capture_audio(duration_ms=100)
        print("  ✓ Audio captured")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: WebSocket connection
    print("\nTest 2: WebSocket")
    try:
        await device.connect()
        print("  ✓ Connected to backend")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 3: Send audio
    print("\nTest 3: Send audio")
    try:
        await device.send_audio()
        print("  ✓ Audio sent")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Receive response
    print("\nTest 4: Receive response")
    try:
        response = await device.receive_response(timeout=5)
        print(f"  ✓ Got response: {len(response)} bytes")
    except Exception as e:
        print(f"  ✗ Timeout: {e}")

asyncio.run(debug_device())
```

## Level 2: Local Testing

### Unit Test Template

```python
# tests/test_module.py
import pytest
import asyncio

class TestSTTPipeline:
    """Test STT module"""
    
    @pytest.fixture
    def pipeline(self):
        from stt_pipeline import STTPipeline
        return STTPipeline()
    
    @pytest.mark.asyncio
    async def test_empty_audio(self, pipeline):
        """Test with silence"""
        import numpy as np
        silence = np.zeros(1600, dtype=np.int16).tobytes()
        result = await pipeline.process_chunk(silence)
        
        assert result['success'] == False  # Silence should fail
        assert result['text'] == ""
    
    @pytest.mark.asyncio
    async def test_noise_audio(self, pipeline):
        """Test with noise"""
        import numpy as np
        noise = np.random.randint(-1000, 1000, 1600, dtype=np.int16).tobytes()
        result = await pipeline.process_chunk(noise)
        
        assert 'latency_ms' in result
        assert result['latency_ms'] < 200  # Should be fast
    
    def test_model_loaded(self, pipeline):
        """Check model is loaded"""
        assert pipeline.model is not None
        assert pipeline.device in ['cuda', 'cpu']
```

Run tests:
```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

## Level 3: Performance Profiling

### CPU Profiling

```python
# profile_cpu.py
import cProfile
import pstats
import asyncio
from backend.stt_pipeline import STTPipeline

async def profile_stt():
    pipeline = STTPipeline()
    
    import numpy as np
    audio = np.zeros(1600, dtype=np.int16).tobytes()
    
    # Warm up
    await pipeline.process_chunk(audio)
    
    # Profile 10 iterations
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(10):
        await pipeline.process_chunk(audio)
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime')
    stats.print_stats(20)

asyncio.run(profile_stt())
```

### Memory Profiling

```python
# profile_memory.py
from memory_profiler import profile

@profile
def test_stt():
    from stt_pipeline import STTPipeline
    import numpy as np
    
    pipeline = STTPipeline()
    audio = np.zeros(1600, dtype=np.int16).tobytes()
    
    result = pipeline.process_chunk(audio)
    return result

test_stt()
```

Run:
```bash
python -m memory_profiler profile_memory.py
```

## Level 4: Hardware Testing

### GPIO/LED Testing (BK7258)

```c
// test_gpio.c
#include <stdio.h>
#include "gpio.h"

void test_led() {
    // Test LED pin
    gpio_init(GPIO_LED);
    gpio_set_direction(GPIO_LED, GPIO_OUTPUT);
    
    printf("Testing LED (5 blinks)...\n");
    for (int i = 0; i < 5; i++) {
        gpio_set_level(GPIO_LED, 1);
        sleep_ms(500);
        printf("  LED ON\n");
        
        gpio_set_level(GPIO_LED, 0);
        sleep_ms(500);
        printf("  LED OFF\n");
    }
    
    printf("LED test passed!\n");
}
```

### Display Testing

```c
// test_display.c
void test_display() {
    display_init();
    
    // Test 1: Fill with color
    printf("Test 1: Colors\n");
    display_fill(0xFFFF);  // White
    sleep_ms(500);
    display_fill(0xF800);  // Red
    sleep_ms(500);
    
    // Test 2: Draw shapes
    printf("Test 2: Shapes\n");
    display_clear();
    display_draw_circle(80, 80, 50, 0xFFFF);
    sleep_ms(500);
    
    // Test 3: Text
    printf("Test 3: Text\n");
    display_clear();
    display_draw_text(10, 10, "Hello World");
    sleep_ms(500);
    
    printf("Display tests passed!\n");
}
```

### Audio Testing

```c
// test_audio.c
void test_microphone() {
    adc_init();
    
    printf("Recording 1 second...\n");
    uint16_t max_val = 0;
    
    for (int i = 0; i < 16000; i++) {
        uint16_t sample = adc_read();
        if (sample > max_val) {
            max_val = sample;
        }
    }
    
    printf("Max amplitude: %d\n", max_val);
    
    if (max_val < 100) {
        printf("✗ Microphone is silent!\n");
    } else {
        printf("✓ Microphone works (max=%d)\n", max_val);
    }
}
```

## Level 5: Nuclear Options

### Clear Cache & Restart

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Clear pip cache
pip cache purge

# Clear CUDA cache
rm -rf ~/.cache/torch/hub

# Restart everything
pkill python
docker-compose down
docker-compose up -d

# Check status
docker-compose logs -f
```

### Reset to Factory State

```bash
# Full reset
rm -rf venv models logs *.log
rm -rf ~/.cache/huggingface
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models fresh
python download_models.py
```

### Device Firmware Reset (BK7258)

```bash
# Erase entire flash
bk_flasher --erase

# Flash bootloader
bk_flasher --flash bootloader.bin 0x0

# Flash firmware
bk_flasher --flash firmware.bin 0x1000

# Verify
bk_flasher --read 0x0 256 > verify.bin
hexdump -C verify.bin | head
```

## Monitoring in Production

### Set up Monitoring

```python
# monitoring.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
stt_requests = Counter('stt_requests_total', 'Total STT requests')
stt_latency = Histogram('stt_latency_ms', 'STT latency in ms')
stt_errors = Counter('stt_errors_total', 'STT errors')

def instrument_stt():
    @wraps(stt_function)
    async def wrapper(*args, **kwargs):
        stt_requests.inc()
        start = time.time()
        try:
            result = await stt_function(*args, **kwargs)
            return result
        except Exception as e:
            stt_errors.inc()
            raise
        finally:
            latency = (time.time() - start) * 1000
            stt_latency.observe(latency)
    return wrapper

# Start metrics server
start_http_server(8001)
```

Access metrics:
```
http://localhost:8001/metrics
```

## Checklist: Before Production

```
☐ All unit tests passing
☐ Integration tests passing
☐ Performance benchmarks met
☐ Memory usage < 20GB
☐ CPU load < 30% average
☐ Audio quality verified
☐ Latency < 500ms E2E
☐ No memory leaks
☐ Graceful error handling
☐ Logging configured
☐ Monitoring in place
☐ Documentation updated
```

---

Version: 1.0
Date: December 27, 2025
Status: Ready for Debugging
