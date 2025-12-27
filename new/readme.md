# WowCube Octopus Avatar - Project Structure & Documentation Map

## Quick Navigation

- 📖 **New to this project?** Start with [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md)
- 🏗️ **Want system architecture?** Read [ARCHITECTURE.md](ARCHITECTURE.md)
- 🚀 **Ready to deploy?** Check [DEPLOYMENT.md](DEPLOYMENT.md)
- 💻 **Let's code!** See [examples/full_pipeline.py](examples/full_pipeline.py)

---

## Project Structure Overview

### 📚 **Documentation** (`docs/`)

Complete guides organized by topic:

#### Core Architecture
- **[00_OVERVIEW.md](docs/00_OVERVIEW.md)** - Project goals, features, timeline
- **[01_SYSTEM_ARCHITECTURE.md](docs/01_SYSTEM_ARCHITECTURE.md)** - Complete system design

#### Hardware (`docs/HARDWARE/`)
- **BK7258_SPECS.md** - Microcontroller specifications & capabilities
- **BK7258_FIRMWARE.md** - Firmware development guide
- **WowCube_Integration.md** - Hardware integration details

#### Audio Processing (`docs/AUDIO/`)
- **SENSEVOICE_UNIFIED.md** ⭐ - Main: STT + Emotion + Language in one model
- **AUDIO_PROCESSING.md** - Audio capture & WebSocket streaming
- **EMOTION_RECOGNITION.md** - Speech emotion analysis details

#### Backend Services (`docs/BACKEND/`)
- **ARCHITECTURE.md** - Backend system design
- **STT_PIPELINE.md** - Speech-to-text implementation
- **LLM_INTEGRATION.md** - Qwen LLM integration
- **TTS_SYNTHESIS.md** - Text-to-speech + emotion-aware synthesis
- **DEPLOYMENT_OPTIONS.md** - Cloud vs on-premise comparison

#### Animation System (`docs/ANIMATION/`)
- **SYNC_SERVICE.md** ⭐ - Animation synchronization service
- **PHONEME_MAPPING.md** - Phoneme to mouth shape mapping
- **EMOTION_ANIMATION.md** - Emotion-based GIF animations
- **AVATAR_DESIGN.md** - Avatar design guidelines

#### Performance & Optimization (`docs/PERFORMANCE/`)
- **LATENCY_ANALYSIS.md** - End-to-end latency breakdown
- **CPU_LOAD_ANALYSIS.md** - CPU/GPU resource analysis
- **COST_ANALYSIS.md** - Infrastructure cost breakdown

#### Deployment Guides (`docs/DEPLOYMENT/`)
- **DOCKER.md** - Docker containerization
- **MODAL.md** - Modal.com cloud deployment
- **ON_PREMISE.md** - On-premise GPU setup
- **MONITORING.md** - Monitoring & logging setup

#### Development (`docs/DEVELOPMENT/`)
- **QUICK_START.md** - 5-minute setup guide
- **TESTING.md** - Testing strategies & tools
- **DEBUGGING.md** - Debugging guide
- **CONTRIBUTING.md** - How to contribute

---

### 💻 **Source Code** (`src/`)

#### Device Firmware (`src/device/`)
```
firmware/           # BK7258 C firmware
  ├─ main.c         # Main application
  ├─ audio_capture.c    # Audio capture routines
  ├─ websocket_client.c # WebSocket client
  ├─ animation_render.c # Animation rendering
  ├─ gif_loader.c       # GIF loading
  ├─ display_driver.c   # Display control
  └─ CMakeLists.txt

graphics/           # Avatar assets
  ├─ avatar_body.gif    # Body animations
  ├─ mouth_shapes.png   # 8 phoneme shapes
  ├─ eyes.json          # Eye parametrics
  └─ animations.json    # Animation definitions
```

#### Backend Services (`src/backend/`)
```
sensevoice_server.py    # SenseVoiceSmall (STT + Emotion + Language)
lm_server.py            # Qwen LLM server
tts_server.py           # TTS synthesis server
animation_server.py     # Animation sync service
main_gateway.py         # Main API gateway

models/                 # Models (auto-downloaded)
config/                 # Configuration files
```

#### Shared Utilities (`src/shared/`)
```
audio_utils.py          # Audio processing utilities
emotion_utils.py        # Emotion analysis helpers
animation_utils.py      # Animation helpers
data_structures.py      # Shared data classes
constants.py            # Constants & enums
```

#### Tests (`src/tests/`)
```
test_sensevoice.py      # SenseVoiceSmall tests
test_lm.py              # LLM tests
test_tts.py             # TTS tests
test_animation.py       # Animation tests
test_integration.py     # End-to-end tests
fixtures/               # Test data & fixtures
conftest.py             # Pytest configuration
```

---

### 🐳 **Deployment** (`docker/`, `kubernetes/`)

#### Docker Files
```
docker/
  ├─ Dockerfile.sensevoice    # SenseVoice container
  ├─ Dockerfile.lm            # LLM container
  ├─ Dockerfile.tts           # TTS container
  ├─ Dockerfile.animation     # Animation container
  ├─ Dockerfile.gateway       # API gateway container
  └─ docker-compose.yml       # Full stack orchestration
```

#### Kubernetes (Optional)
```
kubernetes/
  ├─ namespace.yaml
  ├─ sensevoice-deployment.yaml
  ├─ lm-deployment.yaml
  ├─ tts-deployment.yaml
  ├─ animation-deployment.yaml
  ├─ gateway-deployment.yaml
  ├─ configmap.yaml
  ├─ service.yaml
  └─ ingress.yaml
```

---

### 📝 **Examples** (`examples/`)

Learn by doing:

- **basic_example.py** - Simple usage example
- **streaming_example.py** - Real-time audio streaming
- **emotion_detection.py** - Emotion detection example
- **full_pipeline.py** ⭐ - Complete end-to-end example
- **device_simulator.py** - BK7258 simulator for testing
- **websocket_client.html** - Web client for testing

---

### ⚙️ **Configuration** (`configs/`)

Environment-specific configs:

```
configs/
  ├─ local.yaml          # Local development
  ├─ modal.yaml          # Modal.com deployment
  ├─ aws.yaml            # AWS deployment
  ├─ gcp.yaml            # Google Cloud deployment
  └─ on_premise.yaml     # On-premise setup
```

---

### 🛠️ **Scripts** (`scripts/`)

Utility & deployment scripts:

```
setup_environment.sh     # Environment setup
download_models.py       # Download ML models
run_local.sh             # Run locally
run_docker.sh            # Run with Docker
deploy_modal.py          # Deploy to Modal
benchmark.py             # Performance benchmarking
test_latency.py          # Latency measurement
monitor.py               # Real-time monitoring
```

---

### 📦 **Dependencies** (`requirements/`)

```
base.txt                 # Core dependencies
backend.txt              # Backend-specific
dev.txt                  # Development tools
test.txt                 # Testing tools
prod.txt                 # Production dependencies
```

---

### 🤖 **CI/CD** (`.github/`)

```
.github/
  ├─ workflows/
  │   ├─ ci.yml          # Continuous integration
  │   ├─ tests.yml       # Run tests
  │   ├─ build.yml       # Build Docker images
  │   └─ deploy.yml      # Auto-deploy
  │
  ├─ ISSUE_TEMPLATE/
  │   ├─ bug_report.md
  │   ├─ feature_request.md
  │   └─ question.md
  │
  └─ PULL_REQUEST_TEMPLATE.md
```

---

## Getting Started (5 Minutes)

### Option 1: Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/WowCube-Octopus-Avatar.git
cd WowCube-Octopus-Avatar

# Install dependencies
pip install -r requirements/base.txt

# Download models
python scripts/download_models.py

# Run locally
make run-local
```

### Option 2: Docker
```bash
# Build and run
docker-compose up

# Open browser to http://localhost:8000
```

### Option 3: Modal.com
```bash
# Deploy to Modal
python scripts/deploy_modal.py

# Check deployment status
modal.com dashboard
```

---

## Architecture Quick Overview

```
BK7258 Device
    │ Audio (50ms chunks)
    ▼
SenseVoiceSmall
    │ STT + Emotion + Language
    ├─→ Text ─────────────────┐
    ├─→ Emotion ──────────────┤
    └─→ Language ─────────────┤
                              ▼
                        Qwen LLM
                              │
                              ▼
                        TTS Synthesis
                        + Animation Sync
                              │
                              ▼
                        BK7258 Render
                        (GIF + Mouth + Eyes + Audio)
```

---

## MVP Development Timeline

### Week 1: Audio Processing
- [ ] Install & test SenseVoiceSmall
- [ ] Implement audio capture pipeline
- [ ] Set up WebSocket server
- [ ] Deploy to Modal

**Key files:** `docs/AUDIO/SENSEVOICE_UNIFIED.md`, `src/backend/sensevoice_server.py`

### Week 2: LLM + TTS
- [ ] Integrate Qwen LLM
- [ ] Set up TTS synthesis
- [ ] Implement emotion-aware voice
- [ ] Create main API gateway

**Key files:** `docs/BACKEND/LLM_INTEGRATION.md`, `src/backend/lm_server.py`

### Week 3: Animation + Device
- [ ] Implement animation sync service
- [ ] Create BK7258 firmware
- [ ] Load avatar assets
- [ ] WebSocket device integration

**Key files:** `docs/ANIMATION/SYNC_SERVICE.md`, `src/device/firmware/`

### Week 4: Integration + Deployment
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Docker containerization
- [ ] Production deployment

**Key files:** `docs/DEPLOYMENT/`, `docker-compose.yml`

---

## Directory Size Guide

| Directory | Size | Purpose |
|-----------|------|---------|
| `docs/` | ~15MB | Complete documentation |
| `src/` | ~20MB | Source code |
| `examples/` | ~5MB | Example code |
| `configs/` | ~1MB | Configuration files |
| `scripts/` | ~2MB | Utility scripts |
| **Total (repo)** | **~50MB** | Without models |

Models (auto-downloaded, not in repo):
- SenseVoiceSmall: ~370MB
- Qwen-7B: ~15GB
- TTS models: ~500MB

---

## Key Technologies

- **Audio:** SenseVoiceSmall (ASR + Emotion + Language)
- **LLM:** Qwen 7B
- **TTS:** Glow-TTS + Vocoder
- **Animation:** GIF + PNG overlays + Parametric eyes
- **Device:** BK7258 microcontroller
- **Backend:** FastAPI + WebSocket
- **Deployment:** Docker + Modal.com / On-premise GPU

---

## Performance Targets

- **Latency:** 100-150ms per 50ms audio chunk
- **Cost:** $85/month (100 users)
- **Accuracy:** 85-90% (STT), 80-85% (Emotion), 95%+ (Language)
- **Scalability:** 100-1000+ concurrent users
- **VRAM:** 1.5GB (SenseVoiceSmall unified model)

---

## Important Files Checklist

Essential files for MVP:

- ✅ `README.md` - You are here!
- ✅ `ARCHITECTURE.md` - System design
- ✅ `SETUP.md` - Installation guide
- ✅ `DEPLOYMENT.md` - Deployment instructions
- ✅ `docs/00_OVERVIEW.md` - Project overview
- ✅ `docs/AUDIO/SENSEVOICE_UNIFIED.md` - Audio processing
- ✅ `docs/BACKEND/ARCHITECTURE.md` - Backend design
- ✅ `docs/ANIMATION/SYNC_SERVICE.md` - Animation sync
- ✅ `examples/full_pipeline.py` - Complete example
- ✅ `docker-compose.yml` - Full stack
- ✅ `Makefile` - Common commands

---

## Contributing

See [docs/DEVELOPMENT/CONTRIBUTING.md](docs/DEVELOPMENT/CONTRIBUTING.md) 

---

## Support

- 📖 Documentation: See `docs/` folder
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: [contact info]

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and timeline.

---

## Status

🟢 **MVP Development**
