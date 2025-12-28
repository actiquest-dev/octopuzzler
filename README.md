# Octopus AI - Project Overview

## Overview

Octopus AI is a hybrid device and backend system for a real-time avatar assistant.
The device handles session triggers, face detection, eye tracking, and rendering.
The backend handles face recognition, speech processing, and animation commands.

Key goals:
- Fast greeting after user presence detection.
- Low-latency local eye tracking.
- Streaming audio and animation back to the device.

Full architecture details are in `docs/OCTOPUS_AI_ARCHITECTURE_V1.0.md`.

---

## Current Architecture (Summary)

### Device (BK7258)
- Session triggers: wake word, accelerometer, button.
- Face detection: BlazeFace (device-side).
- Eye tracking: MediaPipe (device-side).
- Rendering: ThorVG-based avatar and lip sync.
- Transport: RTC/WebSocket.

### Backend (GPU)
- Face recognition: InsightFace + FAISS.
- Speech pipeline: SenseVoice (STT + audio emotion) -> Qwen (LLM) -> DIA2 (TTS).
- Animation sync: phoneme timing, gaze, blink markers.
- Event graph storage: FalkorDB.

---

## Data Storage (FalkorDB)

LightRAG is no longer used. We store user history and behavior in FalkorDB as an
event graph. Each user has a graph of actions and events with timestamps.

Example event nodes:
- SessionStart
- FaceRecognized
- AudioChunk
- EmotionDetected
- LLMResponse
- TTSAudio
- AnimationCommand

Example edges:
- USER -> SessionStart
- SessionStart -> AudioChunk
- AudioChunk -> EmotionDetected
- EmotionDetected -> LLMResponse
- LLMResponse -> TTSAudio
- TTSAudio -> AnimationCommand

This graph supports fast history lookup, personalization, and analytics.

---

## Repository Structure

```
docs/
  OCTOPUS_AI_ARCHITECTURE_V1.0.md
  ADDENDUM_QWEN3VL_V1.1.md
  API_REFERENCE.md
  DEPLOYMENT_GUIDE.md
  TROUBLESHOOTING.md

code/
  backend/
  device/

scripts/
tests/
```

---

## Key Docs

- Architecture: `docs/OCTOPUS_AI_ARCHITECTURE_V1.0.md`
- Addendum: `docs/ADDENDUM_QWEN3VL_V1.1.md`
- API: `docs/API_REFERENCE.md`
- Deployment: `docs/DEPLOYMENT_GUIDE.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

---

## Notes

This README reflects the new hybrid architecture and storage model.
If you update core components, update the architecture doc first.
