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