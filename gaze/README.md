# Gaze Tracking (Octopuzzler)

This folder documents the gaze tracking pipeline used by the octopus avatar.
It is built around a lightweight BlazeFace detector and a native macOS worker
that streams gaze over WebSocket.

## Overview

Data flow:

1. Browser captures camera frames (JPEG, base64).
2. Backend forwards frames to the native gaze worker.
3. Native worker runs BlazeFace and computes gaze x/y in [-1..1].
4. Backend sends gaze into timeline_sync.
5. Frontend animator applies gaze to the pupils.

This keeps CPU usage low and avoids FaceMesh custom ops.

## Components

- Native worker (macOS): `code/tools/gaze_capture_macos.mm`
  - Captures camera via AVFoundation
  - Runs BlazeFace (TFLite)
  - Sends `{"type":"gaze","gaze":{"x":...,"y":...,"blink":...}}` to backend
- BlazeFace detector: `code/device/tracking/blazeface_detector.cpp`
- Eye tracking logic: `code/device/tracking/eye_tracking_optimized.cpp`
  - FaceMesh disabled, BlazeFace-only gaze
- Models:
  - `models/blazeface.tflite`
- Backend integration (in octopus-thorvg):
  - `backend/app.py` (gaze_frame forwarding, worker auto-start)
  - `backend/timeline_sync.py` (gaze packets)
- Frontend integration (in octopus-thorvg):
  - `frontend/index.html` (gaze_frame sender)
  - `frontend/octopus_animator.js` (pupil movement)

## Build (macOS)

From repo root:

```
bash build_gaze.sh
```

This builds `code/tools/gaze_capture_macos`.

## Current State

- `code/tools/gaze_capture_macos` (macOS binary) is included as proof-of-life.
- BlazeFace-only gaze is enabled (FaceMesh disabled).
- Target: recompile the same MM code for Beken (7254).

## Run

1. Ensure the model exists:
   - `models/blazeface.tflite`
2. Start the backend (octopus-thorvg):
   - `python3 backend/run.py`
3. The backend auto-starts the gaze worker if the binary exists.

## Environment Variables

- `OCTO_GAZE_WORKER_BIN` (optional)
  - Path to `gaze_capture_macos` binary.
  - Default (from octopus-thorvg): `code/tools/gaze_capture_macos`
- `OCTO_GAZE_INPUT_WS` (optional)
  - WebSocket URL for gaze input, default: `ws://localhost:8080/gaze`
- `OCTO_MODELS_DIR` (optional)
  - Directory for models, default: `models`

## Message Formats

### Browser -> Backend

```
{
  "gaze_frame": {
    "image_b64": "...",
    "mime": "image/jpeg",
    "w": 640,
    "h": 480,
    "ts_ms": 1712345678901
  }
}
```

### Worker -> Backend

```
{
  "type": "gaze",
  "gaze": { "x": 0.12, "y": -0.08, "blink": false }
}
```

### Backend -> Frontend

```
{
  "cmd": "gaze",
  "gaze": { "x": 0.12, "y": -0.08, "blink": false }
}
```

## Notes

- BlazeFace-only gaze uses face center as a proxy for eye direction.
- For low latency, increase camera FPS in the browser (10 FPS works well).
- If you see `Landmarks2TransformMatrix` errors, FaceMesh is still enabled.
  Keep FaceMesh disabled for CPU-only mode.
