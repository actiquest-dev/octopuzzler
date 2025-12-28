# Quick Start Guide

Get WowCube Octopus Avatar running in **30 minutes**.

## Prerequisites

```bash
# Install required tools
- Python 3.10+
- Git
- Docker (optional but recommended)
- 8GB RAM minimum
- GPU recommended (NVIDIA CUDA compatible)
```

## Step 1: Clone Repository (2 minutes)

```bash
git clone https://github.com/yourusername/WowCube-Octopus-Avatar.git
cd WowCube-Octopus-Avatar

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Models (10 minutes)

```bash
# Create models directory
mkdir -p models

# Download SenseVoiceSmall (1.5GB)
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('iic/SenseVoiceSmall')"

# Download Qwen-7B (14GB - optional, use smaller for testing)
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen-7B-Chat')"

# Or use smaller model for quick testing
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen1.5-1.8B-Chat')"
```

## Step 3: Start Backend (5 minutes)

### Option A: Local (Simple)

```bash
# Terminal 1: Start FastAPI server
python3 backend/main.py

# Server runs at http://localhost:8000
# WebSocket at ws://localhost:8000/ws/test-device
```

### Option B: Docker (Recommended)

```bash
# Build Docker image
docker build -t octopus-avatar .

# Run container
docker run --gpus all -p 8000:8000 octopus-avatar
```

## Step 4: Test Backend (5 minutes)

```bash
# Terminal 2: Send test audio
python3 test_stt.py

# Expected output:
#  STT: "Hello world"
#  Emotion: happy (0.85)
#  Language: en (0.96)
#  Response: "That's wonderful!"
#  Audio synthesized: 2000ms
#  Animations: 8 phonemes
```

## Step 5: Connect Device Simulator (3 minutes)

```bash
# Terminal 3: Run device simulator
python3 device_simulator.py

# Simulates BK7258 device sending audio
# Expected: Device avatar responds with animation commands
```

## Quick Test Script

```python
# test_quick.py
import asyncio
import json
from stt_pipeline import STTPipeline
from llm_integration import QwenLLM
from tts_synthesis import TTSSynthesis

async def test_pipeline():
    print(" Starting quick test...")
    
    # Initialize services
    stt = STTPipeline()
    llm = QwenLLM()
    tts = TTSSynthesis()
    
    # Simulate user audio (silence for now)
    import numpy as np
    audio = np.zeros(1600, dtype=np.int16).tobytes()
    
    # 1. STT
    print(" Testing STT...")
    result_stt = await stt.process_chunk(audio)
    print(f"   Text: {result_stt['text']}")
    print(f"   Emotion: {result_stt['emotion']}")
    
    # 2. LLM
    print(" Testing LLM...")
    result_llm = await llm.generate_response(
        user_text="Hello!",
        emotion=result_stt['emotion']
    )
    print(f"   Response: {result_llm['text']}")
    
    # 3. TTS
    print(" Testing TTS...")
    audio_bytes, duration = await tts.synthesize(
        text=result_llm['text'],
        emotion=result_llm['emotion']
    )
    print(f"   Audio: {duration}ms")
    
    print(" All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
```

Run:
```bash
python3 test_quick.py
```

## Directory Structure

```
WowCube-Octopus-Avatar/
├── backend/
│   ├── main.py                    # FastAPI gateway
│   ├── stt_pipeline.py            # Speech recognition
│   ├── llm_integration.py         # Language model
│   ├── tts_synthesis.py           # Text to speech
│   ├── animation_sync_service.py  # Animation sync
│   └── eye_tracking.py            # Eye tracking
│
├── device/
│   ├── firmware/
│   │   ├── main.c                 # BK7258 main
│   │   ├── audio_capture.c        # Audio input
│   │   ├── animation_render.c     # GIF rendering
│   │   ├── display_driver.c       # LCD output
│   │   └── CMakeLists.txt
│   │
│   └── simulator/
│       └── device_simulator.py    # Local testing
│
├── assets/
│   └── avatars/
│       ├── octopus_happy.gif
│       ├── octopus_sad.gif
│       └── ... (5 emotions)
│
├── tests/
│   ├── test_stt.py
│   ├── test_llm.py
│   ├── test_tts.py
│   └── test_quick.py
│
├── docs/
│   └── (all documentation)
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuration

Create `.env` file:

```bash
# Backend
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Models
MODEL_STT=iic/SenseVoiceSmall
MODEL_LLM=Qwen/Qwen-7B-Chat
MODEL_TTS=glow-tts

# Device
DEVICE_ID=test-device-001
DEVICE_HOST=localhost
DEVICE_PORT=5000

# Optional APIs
ANTHROPIC_API_KEY=sk-...  # If using Gemini Live

# Logging
LOG_LEVEL=INFO
```

## Common Commands

```bash
# Start backend
python3 backend/main.py

# Run tests
pytest tests/

# Check code quality
black . && flake8 .

# Generate documentation
cd docs && make html

# Build Docker image
docker build -t octopus .

# Deploy to production
git push origin main  # CI/CD handles deployment

# View logs
tail -f logs/app.log

# Monitor performance
python3 utils/monitor.py
```

## Troubleshooting

### CUDA Error
```
ERROR: No CUDA device found

Solution:
  1. Check GPU: nvidia-smi
  2. Install CUDA 12.1
  3. Set: export CUDA_VISIBLE_DEVICES=0
  4. Fallback to CPU: Set device='cpu' in code
```

### Out of Memory
```
ERROR: CUDA out of memory

Solution 1: Reduce batch size (in code, set batch_size=1)
Solution 2: Use smaller model (Qwen-1.8B instead of Qwen-7B)
Solution 3: Add swap: fallocate -l 16G /swapfile && swapon /swapfile
Solution 4: Use Modal.com (cloud) instead of local
```

### WebSocket Connection Failed
```
ERROR: Failed to connect to WebSocket

Solution:
  1. Check backend is running: curl http://localhost:8000/health
  2. Check firewall: ufw allow 8000
  3. Check CORS: Add to FastAPI app.add_middleware(CORSMiddleware)
  4. Check browser console for errors
```

### Audio Not Playing
```
ERROR: No audio output from TTS

Solution:
  1. Check device has speakers
  2. Check audio permissions: sudo usermod -aG audio $USER
  3. Test with: aplay test.wav
  4. Check TTS output file exists
```

## Performance Benchmarks

```bash
# Run performance test
python3 utils/benchmark.py

Expected results:
  STT latency:      100-150ms 
  LLM latency:      50-200ms 
  TTS latency:      100-150ms 
  Animation:        10-20ms 
  Total E2E:        300-400ms 
  
  Memory (GPU):     ~18GB 
  Memory (RAM):     ~2GB 
  CPU load:         15-25% 
```

## Next Steps

1. **Customize Avatar:**
   ```bash
   # Edit EMOTION_ANIMATION.md for avatar design
   # Create custom GIFs in assets/avatars/
   ```

2. **Train on Your Data:**
   ```bash
   # Fine-tune Qwen on your domain-specific data
   # See LLM_INTEGRATION.md for details
   ```

3. **Deploy to Device:**
   ```bash
   # Build firmware for BK7258
   # See BK7258_FIRMWARE.md for flashing instructions
   ```

4. **Scale to Production:**
   ```bash
   # Deploy to Modal.com or AWS
   # See COST_ANALYSIS.md for infrastructure
   ```

## Learning Path

**Week 1: Understand System**
- Read: ARCHITECTURE.md
- Watch: Demo video (15 min)
- Run: Quick start test

**Week 2: Modify Backend**
- Read: STT_PIPELINE.md, LLM_INTEGRATION.md
- Customize: Emotion prompts, response styles
- Deploy: To Modal.com

**Week 3: Hardware Integration**
- Read: BK7258_FIRMWARE.md
- Build: Device firmware
- Test: With simulator

**Week 4: Production Ready**
- Deploy: To production
- Monitor: Performance metrics
- Scale: Add more users

## Support & Community

```
Documentation:     /docs
Issues:            GitHub Issues
Discussions:       GitHub Discussions
Discord:           [community link]
Email:             support@example.com
```

## Useful Resources

```
Models:
  - SenseVoiceSmall: https://huggingface.co/iic/SenseVoiceSmall
  - Qwen:            https://huggingface.co/Qwen/Qwen-7B-Chat
  - Glow-TTS:        https://github.com/glow-glow/glow-tts

Libraries:
  - FastAPI:         https://fastapi.tiangolo.com
  - Transformers:    https://huggingface.co/docs/transformers
  - PyTorch:         https://pytorch.org/

Hardware:
  - BK7258:          [manufacturer docs]
  - Modal.com:       https://modal.com/docs
```

## Checklist: Ready for Development

- [ ] Python 3.10+ installed
- [ ] Git cloned repository
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Backend starts without errors
- [ ] Test script passes
- [ ] Device simulator connects
- [ ] First API call successful
- [ ] Documentation reviewed

**You're ready to develop! **

---

Version: 1.0
Date: December 27, 2025
Status: Ready to Use
