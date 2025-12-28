# Testing Guide

Complete testing strategy for WowCube Octopus Avatar system.

## Testing Pyramid

```
                    ╱╲
                   ╱  ╲       E2E Tests (10%)
                  ╱    ╲      10-15 min each
                 ╱______╲
                ╱        ╲
               ╱          ╲    Integration Tests (30%)
              ╱            ╲   1-5 min each
             ╱______________╲
            ╱                ╲
           ╱                  ╲ Unit Tests (60%)
          ╱____________________╲ 10-100ms each
```

## Unit Tests (60%)

### Test Structure

```python
# tests/test_stt_pipeline.py
import pytest
import asyncio
import numpy as np
from backend.stt_pipeline import STTPipeline

class TestSTTPipeline:
    """Test STT module in isolation"""
    
    @pytest.fixture
    def pipeline(self):
        """Create fresh pipeline for each test"""
        return STTPipeline()
    
    @pytest.fixture
    def silence_audio(self):
        """Generate silence (1 second @ 16kHz)"""
        return np.zeros(16000, dtype=np.int16).tobytes()
    
    @pytest.fixture
    def noise_audio(self):
        """Generate white noise"""
        noise = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        return noise.astype(np.int16).tobytes()
    
    # ============================================================
    # BASIC FUNCTIONALITY TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_model_loads(self, pipeline):
        """Verify model loads correctly"""
        assert pipeline.model is not None
        assert pipeline.device in ['cuda', 'cpu']
    
    @pytest.mark.asyncio
    async def test_process_silence(self, pipeline, silence_audio):
        """STT should return empty for silence"""
        result = await pipeline.process_chunk(silence_audio)
        
        assert result['success'] == False
        assert result['text'] == ""
        assert result['confidence'] < 0.5
    
    @pytest.mark.asyncio
    async def test_process_noise(self, pipeline, noise_audio):
        """STT should handle noise gracefully"""
        result = await pipeline.process_chunk(noise_audio)
        
        assert 'latency_ms' in result
        assert result['latency_ms'] < 200
    
    # ============================================================
    # OUTPUT FORMAT TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_output_structure(self, pipeline, silence_audio):
        """Verify output has required fields"""
        result = await pipeline.process_chunk(silence_audio)
        
        required_fields = [
            'text', 'emotion', 'language', 'confidence',
            'emotion_scores', 'latency_ms', 'success'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_emotion_scores_sum_to_one(self, pipeline, noise_audio):
        """Emotion scores should sum to ~1.0"""
        result = await pipeline.process_chunk(noise_audio)
        
        total = sum(result['emotion_scores'].values())
        assert 0.9 < total <= 1.1  # Allow small floating point error
    
    @pytest.mark.asyncio
    async def test_confidence_in_range(self, pipeline, noise_audio):
        """Confidence should be 0-1"""
        result = await pipeline.process_chunk(noise_audio)
        
        assert 0.0 <= result['confidence'] <= 1.0
    
    # ============================================================
    # PERFORMANCE TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_latency_under_budget(self, pipeline, silence_audio):
        """STT should complete within 200ms"""
        result = await pipeline.process_chunk(silence_audio)
        
        assert result['latency_ms'] < 200, \
            f"STT took {result['latency_ms']}ms (budget: 200ms)"
    
    @pytest.mark.asyncio
    async def test_consistent_performance(self, pipeline, silence_audio):
        """Performance should be consistent across runs"""
        latencies = []
        
        for _ in range(5):
            result = await pipeline.process_chunk(silence_audio)
            latencies.append(result['latency_ms'])
        
        # Check variance is small
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        assert std_latency / mean_latency < 0.2, \
            "Performance too inconsistent"
    
    # ============================================================
    # ERROR HANDLING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_empty_audio(self, pipeline):
        """Handle empty audio gracefully"""
        result = await pipeline.process_chunk(b"")
        
        assert 'error' not in result or result['success'] == False
    
    @pytest.mark.asyncio
    async def test_corrupted_audio(self, pipeline):
        """Handle corrupted audio gracefully"""
        corrupted = b"\x00\xFF\x00\xFF" * 100
        result = await pipeline.process_chunk(corrupted)
        
        # Should not crash
        assert 'latency_ms' in result
```

### Run Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_stt_pipeline.py::TestSTTPipeline::test_model_loads -v

# Run with coverage
pytest tests/unit/ --cov=backend --cov-report=html

# Run in parallel (faster)
pytest tests/unit/ -n auto

# Run with detailed output
pytest tests/unit/ -vv -s
```

## Integration Tests (30%)

### Test Backend Integration

```python
# tests/test_integration.py
import pytest
import asyncio
import numpy as np
from backend.stt_pipeline import STTPipeline
from backend.llm_integration import QwenLLM
from backend.tts_synthesis import TTSSynthesis

class TestBackendIntegration:
    """Test components working together"""
    
    @pytest.fixture
    def services(self):
        """Initialize all services"""
        return {
            'stt': STTPipeline(),
            'llm': QwenLLM(),
            'tts': TTSSynthesis()
        }
    
    # ============================================================
    # PIPELINE INTEGRATION
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_stt_to_llm(self, services):
        """STT output should work with LLM"""
        # Create fake STT result
        stt_result = {
            'text': 'Hello',
            'emotion': 'happy',
            'language': 'en'
        }
        
        # Pass to LLM
        llm_result = await services['llm'].generate_response(
            user_text=stt_result['text'],
            emotion=stt_result['emotion']
        )
        
        assert llm_result['text'] != ""
        assert len(llm_result['text']) > 5
    
    @pytest.mark.asyncio
    async def test_llm_to_tts(self, services):
        """LLM output should work with TTS"""
        # Create fake LLM result
        llm_result = {
            'text': 'That is wonderful!',
            'emotion': 'happy'
        }
        
        # Pass to TTS
        audio, duration = await services['tts'].synthesize(
            text=llm_result['text'],
            emotion=llm_result['emotion']
        )
        
        assert len(audio) > 0
        assert duration > 0
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, services):
        """Test STT → LLM → TTS flow"""
        # This tests the complete flow
        
        # 1. Fake STT result
        stt_text = "Hello"
        emotion = "happy"
        
        # 2. Get LLM response
        llm_result = await services['llm'].generate_response(
            user_text=stt_text,
            emotion=emotion
        )
        
        # 3. Synthesize TTS
        audio, duration = await services['tts'].synthesize(
            text=llm_result['text'],
            emotion=llm_result['emotion']
        )
        
        # Verify end-to-end
        assert llm_result['text'] != ""
        assert len(audio) > 0
        assert duration > 0
    
    # ============================================================
    # EMOTION HANDLING
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_all_emotions(self, services):
        """Test system works for all emotions"""
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised']
        
        for emotion in emotions:
            # Generate response
            llm_result = await services['llm'].generate_response(
                user_text="Test",
                emotion=emotion
            )
            
            # Synthesize
            audio, _ = await services['tts'].synthesize(
                text=llm_result['text'],
                emotion=emotion
            )
            
            assert llm_result['text'] != ""
            assert len(audio) > 0
    
    # ============================================================
    # LATENCY TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_pipeline_latency(self, services):
        """Full pipeline should complete in <1 second"""
        import time
        
        start = time.time()
        
        # STT (simulated)
        llm_result = await services['llm'].generate_response(
            user_text="Hello",
            emotion="happy"
        )
        
        # TTS
        await services['tts'].synthesize(
            text=llm_result['text'],
            emotion="happy"
        )
        
        elapsed = (time.time() - start) * 1000
        
        assert elapsed < 1000, \
            f"Pipeline took {elapsed}ms (budget: 1000ms)"
```

### Test WebSocket Integration

```python
# tests/test_websocket.py
import pytest
import asyncio
import json
import websockets
import numpy as np

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket can connect"""
    uri = "ws://localhost:8000/ws/test-device"
    
    async with websockets.connect(uri) as websocket:
        # Connection successful
        assert websocket.open
        await websocket.close()

@pytest.mark.asyncio
async def test_send_audio_receive_response():
    """Test audio message round-trip"""
    uri = "ws://localhost:8000/ws/test-device"
    
    async with websockets.connect(uri) as websocket:
        # Send audio chunk
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        message = {
            "type": "audio",
            "data": audio.hex(),
            "sample_rate": 16000
        }
        
        await websocket.send(json.dumps(message))
        
        # Receive response
        response = await asyncio.wait_for(
            websocket.recv(),
            timeout=5.0
        )
        
        data = json.loads(response)
        assert 'type' in data
        assert 'animation' in data or 'status' in data

@pytest.mark.asyncio
async def test_multiple_messages():
    """Test sending multiple messages"""
    uri = "ws://localhost:8000/ws/test-device"
    
    async with websockets.connect(uri) as websocket:
        for i in range(5):
            audio = np.zeros(1600, dtype=np.int16).tobytes()
            message = {
                "sequence": i,
                "data": audio.hex()
            }
            
            await websocket.send(json.dumps(message))
            
            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=5.0
            )
            
            assert response != ""
```

### Run Integration Tests

```bash
# Start backend first
python backend/main.py &

# Run integration tests
pytest tests/integration/ -v

# Stop backend
pkill python
```

## End-to-End Tests (10%)

### Test Complete System

```python
# tests/test_e2e.py
import pytest
import asyncio
import json
import websockets
import numpy as np
import time

class TestEndToEnd:
    """Test complete system working together"""
    
    @pytest.fixture
    async def device_connection(self):
        """Connect to running backend"""
        uri = "ws://localhost:8000/ws/e2e-test-device"
        websocket = await websockets.connect(uri)
        yield websocket
        await websocket.close()
    
    @pytest.mark.asyncio
    async def test_complete_flow(self, device_connection):
        """
        Complete user interaction:
        1. User speaks
        2. Backend processes
        3. Animation streams back
        4. Device renders
        """
        
        # Simulate 2 seconds of speech
        print("Sending audio...")
        audio_chunks = []
        for i in range(20):  # 20 * 100ms = 2 seconds
            audio = np.zeros(1600, dtype=np.int16).tobytes()
            audio_chunks.append(audio)
            
            message = {
                "type": "audio",
                "chunk": i,
                "total_chunks": 20,
                "data": audio.hex()
            }
            
            await device_connection.send(json.dumps(message))
        
        # Collect responses
        print("Receiving responses...")
        responses = []
        
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        try:
            while time.time() - start_time < timeout:
                response = await asyncio.wait_for(
                    device_connection.recv(),
                    timeout=1.0
                )
                
                data = json.loads(response)
                responses.append(data)
                
                # Check for completion marker
                if data.get('type') == 'complete':
                    print(f"Received {len(responses)} response messages")
                    break
        
        except asyncio.TimeoutError:
            pass
        
        # Verify responses
        assert len(responses) > 0, "No responses received"
        
        # Check for animation data
        animation_responses = [
            r for r in responses 
            if 'animation' in r or 'phonemes' in r
        ]
        
        assert len(animation_responses) > 0, \
            "No animation data received"
        
        # Verify animation structure
        for resp in animation_responses:
            if 'phonemes' in resp:
                assert 'mouth_shape' in resp['phonemes'][0]
            if 'gaze' in resp:
                assert 'x' in resp['gaze'][0]
                assert 'y' in resp['gaze'][0]
    
    @pytest.mark.asyncio
    async def test_emotion_detection(self, device_connection):
        """Test system detects emotions"""
        
        emotions_detected = []
        
        for emotion in ['happy', 'sad', 'angry']:
            # Send audio chunk
            audio = np.zeros(1600, dtype=np.int16).tobytes()
            message = {
                "type": "audio",
                "emotion_hint": emotion,
                "data": audio.hex()
            }
            
            await device_connection.send(json.dumps(message))
            
            # Receive response
            response = await asyncio.wait_for(
                device_connection.recv(),
                timeout=5.0
            )
            
            data = json.loads(response)
            if 'emotion' in data:
                emotions_detected.append(data['emotion'])
        
        # Verify emotion was detected
        assert len(emotions_detected) > 0
    
    @pytest.mark.asyncio
    async def test_eye_tracking(self, device_connection):
        """Test eye tracking commands are sent"""
        
        # Enable face detection
        message = {"type": "enable_tracking"}
        await device_connection.send(json.dumps(message))
        
        # Send audio with face tracking enabled
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        message = {
            "type": "audio",
            "tracking_enabled": True,
            "data": audio.hex()
        }
        
        await device_connection.send(json.dumps(message))
        
        # Receive response
        response = await asyncio.wait_for(
            device_connection.recv(),
            timeout=5.0
        )
        
        data = json.loads(response)
        
        # Check for eye gaze data
        assert 'gaze' in data or 'eyes' in data, \
            "No gaze data in response"

# Pytest markers for running subsets
pytestmark = pytest.mark.e2e
```

### Run E2E Tests

```bash
# Start backend
python backend/main.py &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 5

# Run E2E tests
pytest tests/e2e/ -v -s

# Stop backend
kill $BACKEND_PID
```

## Performance Tests

### Benchmark Suite

```python
# tests/test_performance.py
import pytest
import asyncio
import time
import numpy as np
from backend.stt_pipeline import STTPipeline

class TestPerformance:
    """Test performance meets budgets"""
    
    @pytest.fixture
    def pipeline(self):
        return STTPipeline()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_stt_throughput(self, pipeline):
        """Test STT can handle 100 requests/second"""
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        
        start = time.time()
        completed = 0
        
        # Run for 5 seconds
        while time.time() - start < 5:
            result = await pipeline.process_chunk(audio)
            completed += 1
        
        elapsed = time.time() - start
        throughput = completed / elapsed
        
        assert throughput > 10, \
            f"STT throughput {throughput} < 10 req/sec"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_stability(self, pipeline):
        """Test memory doesn't leak"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1e9
        
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        
        # Process 1000 chunks
        for _ in range(1000):
            await pipeline.process_chunk(audio)
        
        # Check memory after
        final_memory = process.memory_info().rss / 1e9
        
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 1.0, \
            f"Memory grew by {memory_growth}GB (leak suspected)"

# Run performance tests
# pytest tests/test_performance.py -m performance -v
```

## Stress Tests

```python
# tests/test_stress.py
import pytest
import asyncio
import numpy as np

class TestStress:
    """Test system under stress"""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test 100 concurrent requests"""
        from backend.stt_pipeline import STTPipeline
        
        pipeline = STTPipeline()
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        
        # Create 100 concurrent tasks
        tasks = [
            pipeline.process_chunk(audio)
            for _ in range(100)
        ]
        
        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check all succeeded
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"{len(errors)} requests failed"
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_long_session(self):
        """Test 1 hour of continuous use"""
        # Simulate 1 hour of conversation
        # This would normally take 1 hour, but we compress time
        
        interactions = 100  # 100 interactions simulating 1 hour
        
        for i in range(interactions):
            # Simulate STT → LLM → TTS
            pass  # Real implementation would do full pipeline
        
        # Verify system is still responsive
        # (no crashes, memory leaks, etc)
```

## Coverage Report

```bash
# Generate coverage
pytest tests/ --cov=backend --cov-report=html

# View report
open htmlcov/index.html

# Target coverage: >80%
# Command to check:
coverage report | grep -E "TOTAL|^(backend|tests)"
```

## Test Configuration

### pytest.ini

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    stress: Stress tests
    slow: Tests that take >10s

# Logging
log_cli = true
log_cli_level = INFO

# Timeouts
timeout = 300

# Parallelization
addopts = -n auto
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
      - name: Generate coverage
        run: pytest tests/ --cov=backend --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Data

### Fixture Audio Files

```python
# tests/fixtures/audio.py
import numpy as np

@pytest.fixture
def audio_silence():
    """1 second of silence"""
    return np.zeros(16000, dtype=np.int16).tobytes()

@pytest.fixture
def audio_noise():
    """1 second of white noise"""
    noise = np.random.randint(-10000, 10000, 16000)
    return noise.astype(np.int16).tobytes()

@pytest.fixture
def audio_sine_wave():
    """1 second of 440Hz sine wave"""
    t = np.linspace(0, 1, 16000)
    wave = np.sin(2 * np.pi * 440 * t)
    scaled = (wave * 32767).astype(np.int16)
    return scaled.tobytes()
```

## Test Execution Strategy

### Pre-commit (5 min)

```bash
# Run before committing
pytest tests/unit/ -q
black . && flake8 .
```

### Pre-push (15 min)

```bash
# Run before pushing
pytest tests/unit/ tests/integration/ -v
```

### Pre-release (1 hour)

```bash
# Full test suite
pytest tests/ -v --cov=backend
pytest tests/ -m performance
pytest tests/ -m stress
```

## Checklist: Before Production

```
Unit Tests:
   >80% code coverage
   All unit tests passing
   No flaky tests (pass 100% of time)
  
Integration Tests:
   All modules work together
   WebSocket integration working
   Error handling tested
  
E2E Tests:
   Complete flows work
   All emotions handled
   Performance budgets met
  
Performance:
   Latency < 500ms E2E
   Memory < 20GB
   CPU < 30% average
   No memory leaks
  
Stress:
   100 concurrent requests work
   Long sessions (1hr) stable
   Graceful degradation under load
  
CI/CD:
   All tests pass on CI
   Coverage reports generated
   Automated deployment configured
```

---

Version: 1.0
Date: December 27, 2025
Status: Ready for Implementation
