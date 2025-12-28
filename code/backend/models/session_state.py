@dataclass
class SessionState:
    device_id: str
    user_profile: Optional[dict]
    initial_emotion: str
    last_vision_emotion: dict
    last_audio_emotion: dict
    fused_emotion: dict
    conversation_history: list
    started_at: datetime
    last_activity: datetime
    
    def update_emotion(self, emotion: dict):
        self.fused_emotion = emotion
        self.last_activity = datetime.now()
```

---

# 4. DEVICE ARCHITECTURE

## 4.1 Software Stack
```
┌────────────────────────────────────────────┐
│         BK7258 SOFTWARE LAYERS             │
├────────────────────────────────────────────┤
│                                            │
│  Application Layer:                        │
│  ├─ Avatar Application                    │
│  ├─ Session Manager                       │
│  └─ RTC Client                            │
│                                            │
│  AI/Graphics Layer:                        │
│  ├─ ThorVG (vector rendering)             │
│  ├─ TensorFlow Lite Micro                 │
│  │   ├─ BlazeFace (face detection)        │
│  │   ├─ MediaPipe (eye tracking)          │
│  │   └─ Wake word detector                │
│  └─ JPEG encoder                          │
│                                            │
│  Middleware Layer:                         │
│  ├─ VolcEngine RTC SDK                    │
│  ├─ Camera driver                         │
│  ├─ Display driver (dual LCD)             │
│  ├─ Audio driver (I2S)                    │
│  └─ IMU driver (I2C)                      │
│                                            │
│  OS Layer:                                 │
│  └─ FreeRTOS                              │
└────────────────────────────────────────────┘
```

## 4.2 Task Architecture
```
CPU0 Tasks (41% load):
├─ Network Task (Priority: 5)
│   ├─ WiFi management
│   ├─ RTC connection handling
│   └─ Data channel processing
│
├─ Audio Task (Priority: 6)
│   ├─ Microphone capture
│   ├─ Speaker playback
│   └─ Audio encoding (G711A)
│
├─ Session Task (Priority: 4)
│   ├─ Wake word detection
│   ├─ Accelerometer monitoring
│   └─ Session lifecycle management
│
└─ Display Task (Priority: 3)
    ├─ Framebuffer → LCD transfer
    └─ Dual LCD synchronization

CPU1 Tasks (58% load):
├─ Camera Task (Priority: 5)
│   ├─ Frame capture (15 FPS)
│   └─ Buffer management
│
├─ Face Detection Task (Priority: 4)
│   ├─ BlazeFace inference (2.5 FPS)
│   └─ Face ROI extraction
│
├─ Eye Tracking Task (Priority: 4)
│   ├─ MediaPipe inference (5 FPS)
│   ├─ Gaze calculation
│   └─ Smooth interpolation
│
├─ ThorVG Render Task (Priority: 6)
│   ├─ Avatar generation (30 FPS)
│   ├─ Mouth shape rendering
│   ├─ Eye overlay
│   └─ RGBA → RGB565 conversion
│
└─ JPEG Encode Task (Priority: 3)
    └─ Face crop encoding (on session start)
```

## 4.3 Memory Management
```
Flash Memory Map (128MB):
├─ 0x000000: Bootloader (128KB)
├─ 0x020000: Application (5MB)
├─ 0x520000: TensorFlow Models (32.5MB)
│   ├─ BlazeFace: 2.5MB
│   ├─ MediaPipe: 30MB
│   └─ Wake word: 1MB (TBD)
├─ 0x2520000: ThorVG library (95KB)
├─ 0x2537000: Assets (reserved, unused)
└─ 0x2537000-END: Available (89MB)

RAM Allocation (512MB):
├─ Static (230KB):
│   ├─ Application: 50KB
│   ├─ ThorVG context: 100KB
│   └─ RTC buffers: 80KB
│
├─ Dynamic Heap (456MB):
│   ├─ Camera buffers: 10MB
│   ├─ Audio buffers: 64MB
│   ├─ Display framebuffer: 200KB
│   ├─ TFLite tensors: 30MB
│   ├─ Network buffers: 40MB
│   ├─ ThorVG shapes: 50KB
│   └─ Application heap: 312MB
│
└─ Reserved: 56MB (11%)