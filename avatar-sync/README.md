# Avatar Sync (Gemini Live) Notes

This folder documents the recent work that wires Gemini Live audio to the
octopus avatar with server-side timeline sync and lipsync.

## Why We Did This

We needed **tight, predictable sync** between Gemini Live audio and avatar
animation. The goal is:

- **Ideal lip-sync timing** (jaw + visemes aligned to the audio playhead).
- A **stacked animation model** that layers jaw, lipsync, emotions, and actions.
- Reliable delivery of animation packets even during long responses.
- Gemini Live serves as a **test environment** for fast sync iteration, so we
  can debug timing without burning budget on backend model runs.

## What Was Built

### 1) WebSocket Proxy + Auth
- Browser connects to a local proxy (`backend/app.py`).
- Proxy connects to Gemini Live with OAuth service account.
- Browser messages are routed; `service_url` is handled locally.

### 2) Audio Buffering and Timeline Sync
- Gemini streams `audio/pcm` chunks.
- Server buffers ~300ms (configurable) and sends consolidated chunks to
  `backend/timeline_sync.py`.
- Timeline service computes animation packets aligned to audio playback.

### 3) Heartbeat and Playback Alignment
- Browser sends heartbeat with `playback_delay_s`.
- Timeline sync smooths the delay to keep a stable playhead reference.

### 4) Animation Layers Produced
- **Jaw**: RMS-based jaw movement (16ms steps, EMA smoothing).
- **Lipsync (proxy visemes)**: RMS + ZCR (50ms steps).
- **Emotion tags**: from model output, forwarded as emotion events.
- **Action tags**: `sing`, `laugh`, `whisper`, `shout`, `sfx_*` events.

## Key Files

- `backend/app.py`: proxy, audio buffering, tag parsing, timeline events.
- `backend/timeline_sync.py`: builds `audio_sync` packets, heartbeat, emotion/action.
- `frontend/index.html`: system prompt + tag filtering in chat.
- `frontend/mediaUtils.js`: microphone capture + AudioWorklet streaming.
- `frontend/geminilive.js`: Gemini Live client (setup, realtime_input, transcription).
- `frontend/octopus_animator.js`: consumes `audio_sync` and renders lipsync.

## Code Layout (Files to Move)

Copy the working pieces into `octopuzzler/code/avatar-sync/` using this layout:

```
code/
  avatar-sync/
    backend/
      app.py
      timeline_sync.py
      run.py
    frontend/
      index.html
      geminilive.js
      mediaUtils.js
      octopus_animator.js
      audio-processors/
        capture.worklet.js
        playback.worklet.js
```

Notes:
- Do not copy `credentials.json` into source control.
- Keep `audio-processors/` together with the frontend assets.

## Message Flow (High Level)

1) Browser -> proxy: `service_url`, `setup`, `realtime_input`, `client_content`.
2) Proxy -> Gemini Live: forwards setup and audio.
3) Gemini Live -> proxy: `modelTurn` audio + transcriptions.
4) Proxy -> timeline-sync: `audio_chunk`, `emotion`, `action`.
5) Timeline-sync -> proxy -> browser: `audio_sync` packets for lipsync.

## Notes

- Action tags are emitted as events; actual animations are implemented client-side.
- Proxy visemes are a lightweight fallback; true visemes can be injected later.
