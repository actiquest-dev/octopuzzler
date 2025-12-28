"""
RTC Gateway - Main Orchestrator

WebRTC gateway that orchestrates all backend services:
- Manages WebRTC connections with devices
- Routes audio/video streams to appropriate services
- Coordinates session lifecycle
- Handles face recognition and user management
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
import json
import base64
import aiohttp

from services.model_manager import DynamicModelManager
from services.emotion_fusion_service import EmotionFusionService
from services.animation_sync_service import AnimationSyncService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Octopus AI RTC Gateway",
    version="2.0",
    description="WebRTC Gateway for Octopus Avatar System"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Service Endpoints Configuration
# ============================================================

class ServiceEndpoints:
    """External service endpoints"""
    SENSEVOICE_URL = "http://sensevoice:8081"
    EMOVIT_URL = "http://emovit:8083"
    INSIGHTFACE_URL = "http://insightface:8084"
    
    @staticmethod
    def from_env():
        """Load from environment variables"""
        import os
        ServiceEndpoints.SENSEVOICE_URL = os.getenv(
            "SENSEVOICE_URL", 
            ServiceEndpoints.SENSEVOICE_URL
        )
        ServiceEndpoints.EMOVIT_URL = os.getenv(
            "EMOVIT_URL",
            ServiceEndpoints.EMOVIT_URL
        )
        ServiceEndpoints.INSIGHTFACE_URL = os.getenv(
            "INSIGHTFACE_URL",
            ServiceEndpoints.INSIGHTFACE_URL
        )


# ============================================================
# Session Manager
# ============================================================

class SessionManager:
    """
    Manages active sessions
    
    Tracks:
    - Active WebRTC connections
    - User profiles
    - Session state
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, 'Session'] = {}
        self.model_manager = DynamicModelManager()
        self.emotion_fusion = EmotionFusionService()
        self.animation_sync = AnimationSyncService()
        
        # HTTP client for external services
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("SessionManager initialized")
    
    async def initialize(self):
        """Initialize async components"""
        self.http_session = aiohttp.ClientSession()
        logger.info("✓ HTTP session created")
    
    async def shutdown(self):
        """Cleanup async components"""
        if self.http_session:
            await self.http_session.close()
        logger.info("✓ HTTP session closed")
    
    async def create_session(self, device_id: str, websocket: WebSocket) -> 'Session':
        """Create new session"""
        session = Session(
            device_id=device_id,
            websocket=websocket,
            model_manager=self.model_manager,
            emotion_fusion=self.emotion_fusion,
            animation_sync=self.animation_sync,
            http_session=self.http_session
        )
        
        self.active_sessions[device_id] = session
        logger.info(f"Created session for device: {device_id}")
        
        return session
    
    async def get_session(self, device_id: str) -> Optional['Session']:
        """Get existing session"""
        return self.active_sessions.get(device_id)
    
    async def end_session(self, device_id: str):
        """End session"""
        if device_id in self.active_sessions:
            session = self.active_sessions[device_id]
            await session.cleanup()
            del self.active_sessions[device_id]
            logger.info(f"Ended session for device: {device_id}")
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return {
            "active_sessions": len(self.active_sessions),
            "sessions_list": [
                {
                    "device_id": s.device_id,
                    "session_id": s.session_id,
                    "user_name": s.user_profile.get("name") if s.user_profile else None,
                    "duration_seconds": (datetime.now() - s.start_time).total_seconds()
                }
                for s in self.active_sessions.values()
            ],
            "model_manager": self.model_manager.get_vram_status()
        }


# ============================================================
# Session Class
# ============================================================

class Session:
    """
    Individual session state
    
    Handles:
    - User profile
    - Audio/video streams
    - Emotion tracking
    - Conversation context
    """
    
    def __init__(
        self,
        device_id: str,
        websocket: WebSocket,
        model_manager: DynamicModelManager,
        emotion_fusion: EmotionFusionService,
        animation_sync: AnimationSyncService,
        http_session: aiohttp.ClientSession
    ):
        self.device_id = device_id
        self.session_id = f"session_{device_id}_{int(datetime.now().timestamp())}"
        self.websocket = websocket
        
        # Services
        self.model_manager = model_manager
        self.emotion_fusion = emotion_fusion
        self.animation_sync = animation_sync
        self.http_session = http_session
        
        # State
        self.user_profile: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        self.current_emotion = "neutral"
        self.current_vision_emotion = {"emotion": "neutral", "intensity": 0.5}
        self.start_time = datetime.now()
        
        # Conversation topics
        self.conversation_topics: List[str] = []
        
        logger.info(f"Session created: {self.session_id}")
    
    async def handle_session_start(self, data: Dict) -> Dict:
        """
        Handle session start event
        
        Args:
            data: {
                "trigger_type": "wake_word" | "button",
                "face_image": base64 JPEG (200×200)
            }
        
        Returns:
            {
                "session_id": str,
                "user_data": dict (if recognized)
            }
        """
        logger.info(f"Session start: trigger={data.get('trigger_type')}")
        
        # Recognize user from face
        if "face_image" in data:
            try:
                face_image_b64 = data["face_image"]
                
                # Call InsightFace service
                recognition_result = await self._call_insightface(face_image_b64)
                
                if recognition_result.get("is_known"):
                    self.user_profile = recognition_result
                    logger.info(f"User recognized: {recognition_result.get('user_name')}")
                else:
                    logger.info("Unknown user (no face recognition match)")
            
            except Exception as e:
                logger.error(f"Face recognition error: {e}", exc_info=True)
        
        return {
            "session_id": self.session_id,
            "user_data": self.user_profile,
            "status": "started"
        }
    
    async def handle_audio_chunk(self, audio_data: bytes) -> Dict:
        """
        Process audio chunk
        
        Pipeline:
        1. SenseVoice: STT + Audio Emotion
        2. Emotion Fusion: Audio + Vision Emotion
        3. Qwen3-VL: Generate response
        4. DIA2: TTS synthesis
        5. Animation Sync: Generate markers
        
        Args:
            audio_data: G711A encoded audio
        
        Returns:
            {
                "text_response": str,
                "audio_response": str (base64 G711A),
                "animation": dict,
                "emotion": str
            }
        """
        try:
            # Stage 1: SenseVoice (STT + Audio Emotion)
            stt_result = await self._call_sensevoice(audio_data)
            
            text = stt_result.get("text", "")
            audio_emotion = stt_result.get("audio_emotion", {})
            
            logger.info(f"STT result: '{text}' (emotion={audio_emotion.get('emotion')})")
            
            if not text.strip():
                logger.warning("Empty transcription, skipping response generation")
                return {
                    "text_response": "",
                    "audio_response": "",
                    "animation": {},
                    "emotion": self.current_emotion
                }
            
            # Stage 2: Emotion Fusion
            fused_emotion = await self.emotion_fusion.fuse(
                audio_emotion=audio_emotion,
                vision_emotion=self.current_vision_emotion,
                audio_chunk=None  # VAD handled by SenseVoice
            )
            
            self.current_emotion = fused_emotion.emotion
            
            logger.info(f"Fused emotion: {fused_emotion.emotion} (conf={fused_emotion.confidence:.2f})")
            
            # Stage 3: Qwen3-VL (LLM Response)
            llm_response = await self.model_manager.generate_llm_response(
                text=text,
                image=None,  # No vision context in audio-only mode
                user_profile=self.user_profile,
                emotion=fused_emotion.emotion,
                conversation_topics=self.conversation_topics
            )
            
            response_text = llm_response["response"]
            
            logger.info(f"LLM response: '{response_text[:50]}...'")
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": text,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Stage 4: DIA2 (TTS)
            tts_result = await self.model_manager.generate_tts(
                text=response_text,
                emotion=fused_emotion.emotion,
                intensity=fused_emotion.intensity
            )
            
            audio_b64 = tts_result["audio_base64"]
            word_timestamps = tts_result.get("word_timestamps", [])
            
            # Stage 5: Animation Sync
            animation = await self.animation_sync.generate_animation(
                text=response_text,
                word_timestamps=word_timestamps,
                emotion=fused_emotion.emotion,
                intensity=fused_emotion.intensity
            )
            
            logger.info(f"Pipeline complete: {len(word_timestamps)} words, {len(animation['sync_markers'])} markers")
            
            return {
                "text_response": response_text,
                "audio_response": audio_b64,
                "animation": animation,
                "emotion": fused_emotion.emotion
            }
        
        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)
            
            # Return error response
            return {
                "text_response": "Sorry, I had trouble processing that.",
                "audio_response": "",
                "animation": {},
                "emotion": "neutral",
                "error": str(e)
            }
    
    async def handle_face_update(self, face_image_b64: str):
        """
        Update vision emotion from face image
        
        Args:
            face_image_b64: Base64 JPEG (200×200)
        """
        try:
            # Call EmoVIT service
            emotion_result = await self._call_emovit(face_image_b64)
            
            if emotion_result.get("face_detected"):
                emotion_data = emotion_result.get("emotion", {})
                self.current_vision_emotion = {
                    "emotion": emotion_data.get("emotion", "neutral"),
                    "intensity": emotion_data.get("intensity", 0.5)
                }
                
                logger.debug(f"Vision emotion updated: {self.current_vision_emotion}")
        
        except Exception as e:
            logger.error(f"Face emotion update error: {e}")
    
    async def _call_sensevoice(self, audio_data: bytes) -> Dict:
        """Call SenseVoice service for STT + emotion"""
        try:
            # Encode audio to base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            url = f"{ServiceEndpoints.SENSEVOICE_URL}/v1/transcribe"
            payload = {
                "audio_data": audio_b64,
                "sample_rate": 16000,
                "language": "auto"
            }
            
            async with self.http_session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"SenseVoice error: {response.status} - {error_text}")
                    return {"text": "", "audio_emotion": {"emotion": "neutral", "confidence": 0.5}}
        
        except Exception as e:
            logger.error(f"SenseVoice call failed: {e}")
            return {"text": "", "audio_emotion": {"emotion": "neutral", "confidence": 0.5}}
    
    async def _call_emovit(self, face_image_b64: str) -> Dict:
        """Call EmoVIT service for vision emotion"""
        try:
            url = f"{ServiceEndpoints.EMOVIT_URL}/v1/detect"
            payload = {
                "image_data": face_image_b64,
                "return_all_emotions": False
            }
            
            async with self.http_session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"EmoVIT error: {response.status} - {error_text}")
                    return {"face_detected": False}
        
        except Exception as e:
            logger.error(f"EmoVIT call failed: {e}")
            return {"face_detected": False}
    
    async def _call_insightface(self, face_image_b64: str) -> Dict:
        """Call InsightFace service for face recognition"""
        try:
            # Decode base64 to bytes
            face_image_bytes = base64.b64decode(face_image_b64)
            
            url = f"{ServiceEndpoints.INSIGHTFACE_URL}/v1/recognize"
            
            # Send as multipart form data
            data = aiohttp.FormData()
            data.add_field('file', face_image_bytes, filename='face.jpg', content_type='image/jpeg')
            
            async with self.http_session.post(url, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"InsightFace error: {response.status} - {error_text}")
                    return {"is_known": False}
        
        except Exception as e:
            logger.error(f"InsightFace call failed: {e}")
            return {"is_known": False}
    
    async def cleanup(self):
        """Cleanup session resources"""
        # Save conversation history (TODO: to database)
        logger.info(f"Session cleanup: {len(self.conversation_history)} messages in history")
        
        # Update user analytics if user was recognized
        if self.user_profile:
            # TODO: Update analytics in database
            pass
        
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Session {self.session_id} ended (duration: {duration:.1f}s)")


# ============================================================
# Global Session Manager
# ============================================================

session_manager = SessionManager()


# ============================================================
# REST API Endpoints
# ============================================================

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "Octopus AI RTC Gateway",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "stats": "/v1/stats",
            "websocket": "/rtc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = session_manager.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": stats["active_sessions"],
        "model_status": stats["model_manager"]
    }


@app.get("/v1/stats")
async def get_stats():
    """Get detailed statistics"""
    return session_manager.get_stats()


@app.post("/v1/session/{device_id}/end")
async def end_session(device_id: str):
    """End session manually"""
    await session_manager.end_session(device_id)
    return {"status": "ended", "device_id": device_id}


# ============================================================
# WebSocket Endpoint (WebRTC Data Channel)
# ============================================================

@app.websocket("/rtc")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebRTC data channel endpoint
    
    Message Types:
    --------------
    Client → Server:
    - session_start: Initialize session
    - audio_chunk: Process audio
    - face_update: Update face emotion
    - session_end: End session
    
    Server → Client:
    - session_initialized: Session ready
    - audio_response: TTS + animation
    - error: Error message
    """
    await websocket.accept()
    
    device_id = None
    session = None
    
    try:
        logger.info("WebSocket connection established")
        
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                continue
            
            msg_type = data.get("type")
            logger.debug(f"Received: {msg_type}")
            
            # Handle message types
            if msg_type == "session_start":
                device_id = data.get("device_id")
                
                if not device_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "device_id required"
                    })
                    continue
                
                # Create session
                session = await session_manager.create_session(device_id, websocket)
                
                # Process session start
                response = await session.handle_session_start(data)
                
                await websocket.send_json({
                    "type": "session_initialized",
                    **response
                })
            
            elif msg_type == "audio_chunk":
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active session"
                    })
                    continue
                
                # Decode audio
                audio_b64 = data.get("audio_data")
                if not audio_b64:
                    continue
                
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    logger.error(f"Audio decode error: {e}")
                    continue
                
                # Process audio
                response = await session.handle_audio_chunk(audio_bytes)
                
                # Send response
                await websocket.send_json({
                    "type": "audio_response",
                    **response
                })
            
            elif msg_type == "face_update":
                if session:
                    face_image_b64 = data.get("face_image")
                    if face_image_b64:
                        await session.handle_face_update(face_image_b64)
            
            elif msg_type == "session_end":
                if session and device_id:
                    await session_manager.end_session(device_id)
                
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session.session_id if session else None
                })
                
                break
            
            elif msg_type == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {device_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if device_id:
            await session_manager.end_session(device_id)
        
        logger.info("WebSocket connection closed")


# ============================================================
# Startup/Shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("RTC Gateway starting...")
    
    # Load service endpoints from environment
    ServiceEndpoints.from_env()
    
    # Initialize session manager
    await session_manager.initialize()
    
    logger.info("✓ RTC Gateway ready")
    logger.info(f"  SenseVoice: {ServiceEndpoints.SENSEVOICE_URL}")
    logger.info(f"  EmoVIT: {ServiceEndpoints.EMOVIT_URL}")
    logger.info(f"  InsightFace: {ServiceEndpoints.INSIGHTFACE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("RTC Gateway shutting down...")
    
    # End all active sessions
    for device_id in list(session_manager.active_sessions.keys()):
        await session_manager.end_session(device_id)
    
    # Shutdown session manager
    await session_manager.shutdown()
    
    logger.info("✓ RTC Gateway stopped")


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
```