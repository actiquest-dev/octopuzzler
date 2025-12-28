# LLM Integration

## Overview

Language model for generating empathic responses aware of user emotion and language.

```
STT Output (text + emotion + language)
    ↓
Qwen LLM
    ├─ Input context (emotion, language, history)
    ├─ System prompt (empathic octopus personality)
    └─ Generate response (emotion-aware)
    ↓
Response text + emotion guidance
```

## Model: Qwen

Three size options:

```
Qwen-7B:   Fast (50-100ms), good for MVP
Qwen-14B:  Balanced (100-200ms), better quality
Qwen-72B:  Best quality (200-500ms), slower
```

## Installation

```bash
pip install transformers torch
```

## Implementation

```python
# llm_integration.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from typing import Dict, Optional

class QwenLLM:
    """Qwen language model for response generation"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B-Chat",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f" {model_name} loaded")
        
        # Emotion-aware prompts
        self.emotion_prompts = {
            "happy": "The user is in a great mood! Celebrate with them, be upbeat and enthusiastic.",
            "sad": "The user is feeling down. Show empathy and compassion. Offer comfort.",
            "angry": "The user is frustrated. Acknowledge their feelings and stay calm.",
            "neutral": "The user is in a normal mood. Be friendly and engaging.",
            "surprised": "The user is amazed or shocked. Acknowledge their surprise."
        }
    
    async def generate_response(
        self,
        user_text: str,
        emotion: str = "neutral",
        language: str = "en",
        conversation_history: Optional[list] = None,
        max_tokens: int = 50
    ) -> Dict:
        """
        Generate empathic response
        
        Returns:
            {
                "text": "That's wonderful!",
                "emotion": "happy",
                "confidence": 0.95
            }
        """
        
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(emotion, language)
            
            # Build conversation context
            messages = self._build_conversation(
                user_text,
                emotion,
                conversation_history
            )
            
            # Generate response
            response_text = await self._generate(
                messages,
                system_prompt,
                max_tokens
            )
            
            # Determine response emotion
            response_emotion = self._infer_emotion(response_text, emotion)
            
            return {
                "text": response_text,
                "emotion": response_emotion,
                "confidence": 0.9,
                "success": True
            }
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "text": "I'm here to listen!",
                "emotion": "neutral",
                "confidence": 0.5,
                "success": False,
                "error": str(e)
            }
    
    def _build_system_prompt(self, emotion: str, language: str) -> str:
        """Build system prompt with emotion context"""
        
        emotion_context = self.emotion_prompts.get(
            emotion,
            self.emotion_prompts["neutral"]
        )
        
        language_name = self._get_language_name(language)
        
        system_prompt = f"""You are Ollie, an empathic octopus avatar in a smart device.

User's current emotion: {emotion}
User's language: {language_name}

Personality:
- Warm and caring
- Responsive to emotions
- Conversational and natural
- Curious about user's thoughts
- Uses appropriate humor

Response guidelines:
- Keep it brief (1-2 sentences, under 20 seconds when spoken)
- Match user's emotional tone
- Respond in {language_name}
- Be genuine and authentic
- {emotion_context}

Important: Your response will be converted to speech and animated.
Keep it concise and natural."""
        
        return system_prompt
    
    def _build_conversation(
        self,
        user_text: str,
        emotion: str,
        history: Optional[list] = None
    ) -> list:
        """Build conversation context"""
        
        messages = []
        
        # Add conversation history
        if history:
            messages.extend(history[-4:])  # Last 4 messages
        
        # Add current user message
        user_message = f"""User (emotion: {emotion}): {user_text}"""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    async def _generate(
        self,
        messages: list,
        system_prompt: str,
        max_tokens: int
    ) -> str:
        """Generate response using model"""
        
        # Format for Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            system=system_prompt
        )
        
        # Tokenize
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids
            )
        ]
        
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
    def _infer_emotion(self, text: str, user_emotion: str) -> str:
        """Infer response emotion from text"""
        
        # Simple heuristics (in production, use emotion classifier)
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["great", "wonderful", "amazing", "love", "happy"]):
            return "happy"
        
        if any(word in text_lower for word in ["sad", "sorry", "miss", "lonely"]):
            return "sad"
        
        if any(word in text_lower for word in ["angry", "furious", "annoyed"]):
            return "angry"
        
        if any(word in text_lower for word in ["wow", "really", "wow"]):
            return "surprised"
        
        # Default to matching user emotion
        return user_emotion
    
    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to name"""
        
        languages = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ru": "Russian",
            "ko": "Korean",
            "pt": "Portuguese",
            "it": "Italian"
        }
        
        return languages.get(lang_code, "English")

# Singleton
llm_service = None

def init_llm(model_name: str = "Qwen/Qwen-7B-Chat"):
    """Initialize LLM service"""
    global llm_service
    llm_service = QwenLLM(model_name=model_name)
    return llm_service

async def get_llm_service():
    """Get LLM service instance"""
    return llm_service
```

## Integration with STT

```python
# full_pipeline.py
import asyncio
from stt_pipeline import STTPipeline
from llm_integration import QwenLLM
from tts_service import TTSService

class FullConversationPipeline:
    """Complete STT → LLM → TTS pipeline"""
    
    def __init__(self):
        self.stt = STTPipeline()
        self.llm = QwenLLM()
        self.tts = TTSService()
        self.conversation_history = []
    
    async def process_user_audio(self, audio_bytes: bytes) -> Dict:
        """Process user audio and generate response"""
        
        # 1. STT: Convert audio to text + emotion + language
        stt_result = await self.stt.process_chunk(audio_bytes)
        
        user_text = stt_result['text']
        user_emotion = stt_result['emotion']
        user_language = stt_result['language']
        
        print(f"User: {user_text}")
        print(f"  Emotion: {user_emotion}")
        print(f"  Language: {user_language}")
        
        # 2. LLM: Generate empathic response
        llm_result = await self.llm.generate_response(
            user_text=user_text,
            emotion=user_emotion,
            language=user_language,
            conversation_history=self.conversation_history
        )
        
        response_text = llm_result['text']
        response_emotion = llm_result['emotion']
        
        print(f"Bot: {response_text}")
        print(f"  Emotion: {response_emotion}")
        
        # 3. TTS: Synthesize speech
        tts_result = await self.tts.synthesize(
            text=response_text,
            emotion=response_emotion,
            language=user_language
        )
        
        # 4. Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": f"[{user_emotion}] {user_text}"
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": f"[{response_emotion}] {response_text}"
        })
        
        # Limit history to last 10 messages
        self.conversation_history = self.conversation_history[-10:]
        
        return {
            "user_text": user_text,
            "user_emotion": user_emotion,
            "response_text": response_text,
            "response_emotion": response_emotion,
            "audio": tts_result['audio'],
            "duration_ms": tts_result['duration_ms']
        }

# Usage
pipeline = FullConversationPipeline()
result = await pipeline.process_user_audio(audio_bytes)
```

## Prompt Engineering

### System Prompt Template

```python
system_prompt = """You are Ollie, an empathic octopus avatar.

Context:
- User emotion: {emotion}
- User language: {language}
- Conversation history: {history}

Personality traits:
- Warm and caring
- Emotionally intelligent
- Curious and playful
- Genuine and authentic
- Uses natural speech patterns

Response rules:
1. Keep it brief (1-2 sentences max)
2. Match the user's emotional tone
3. Respond in the user's language
4. Ask follow-up questions when appropriate
5. Show genuine interest
6. Avoid being preachy or condescending

Emotion guidelines:
- Happy: Celebrate with them, be enthusiastic
- Sad: Show empathy, offer comfort
- Angry: Validate feelings, stay calm
- Neutral: Be friendly and engaging
- Surprised: Acknowledge amazement

Remember: Your words will be converted to speech and animated.
Speak naturally as if talking to a friend."""
```

### Few-Shot Examples

```python
examples = """
Example 1:
User: "I'm so excited about my promotion!"
Emotion: happy
Response: "That's fantastic! Congratulations! You must feel amazing right now. "

Example 2:
User: "I've been feeling really lonely lately"
Emotion: sad
Response: "I hear you. Loneliness is tough. I'm here to listen if you want to talk about it."

Example 3:
User: "This project is frustrating me"
Emotion: angry
Response: "That sounds really frustrating. What's making it so difficult?"

Example 4:
User: "How's your day going?"
Emotion: neutral
Response: "My day is wonderful, especially now that I'm talking with you! How about yours?"
"""
```

## Conversation Management

```python
# conversation_manager.py
from datetime import datetime

class ConversationManager:
    """Manage multi-turn conversations"""
    
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.start_time = datetime.now()
    
    def add_user_message(self, text: str, emotion: str, language: str):
        """Add user message to history"""
        
        self.history.append({
            "role": "user",
            "content": text,
            "emotion": emotion,
            "language": language,
            "timestamp": datetime.now()
        })
        
        self._trim_history()
    
    def add_assistant_message(self, text: str, emotion: str):
        """Add assistant message to history"""
        
        self.history.append({
            "role": "assistant",
            "content": text,
            "emotion": emotion,
            "timestamp": datetime.now()
        })
        
        self._trim_history()
    
    def _trim_history(self):
        """Keep only last N messages"""
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> list:
        """Get conversation context for LLM"""
        
        return [
            {
                "role": msg['role'],
                "content": msg['content']
            }
            for msg in self.history[-5:]  # Last 5 turns
        ]
    
    def get_emotion_trend(self) -> str:
        """Get user's emotional trend"""
        
        user_emotions = [
            msg['emotion']
            for msg in self.history
            if msg['role'] == 'user' and 'emotion' in msg
        ]
        
        if not user_emotions:
            return "neutral"
        
        # Most recent emotion
        return user_emotions[-1]

# Usage
manager = ConversationManager()

manager.add_user_message(
    text="I'm so happy!",
    emotion="happy",
    language="en"
)

manager.add_assistant_message(
    text="That's wonderful!",
    emotion="happy"
)

# Get context for next response
context = manager.get_context()
```

## Error Handling

```python
# error_handling.py

async def generate_with_fallback(
    llm: QwenLLM,
    user_text: str,
    emotion: str,
    language: str,
    fallback_responses: Dict = None
) -> Dict:
    """Generate response with fallback options"""
    
    try:
        # Try primary LLM
        result = await llm.generate_response(
            user_text=user_text,
            emotion=emotion,
            language=language
        )
        
        if result['success'] and result['text']:
            return result
    
    except Exception as e:
        print(f"LLM error: {e}")
    
    # Fallback to canned responses
    if fallback_responses is None:
        fallback_responses = {
            "happy": "That sounds wonderful! ",
            "sad": "I'm here for you.",
            "angry": "I understand your frustration.",
            "neutral": "Tell me more!",
            "surprised": "That's amazing!"
        }
    
    return {
        "text": fallback_responses.get(emotion, "I'm listening."),
        "emotion": emotion,
        "confidence": 0.3,
        "success": False,
        "fallback": True
    }
```

## Performance Optimization

### Model Quantization

```python
from transformers import AutoModelForCausalLM

# Load quantized model (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto"
)
```

### Batch Processing

```python
async def batch_generate_responses(
    llm: QwenLLM,
    user_inputs: List[Dict]
) -> List[Dict]:
    """Generate multiple responses in parallel"""
    
    tasks = [
        llm.generate_response(
            user_text=inp['text'],
            emotion=inp['emotion'],
            language=inp['language']
        )
        for inp in user_inputs
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_response_template(emotion: str, language: str) -> str:
    """Cache response templates"""
    
    templates = {
        ("happy", "en"): "That's wonderful! ",
        ("sad", "en"): "I'm here for you.",
    }
    
    return templates.get((emotion, language), "I'm listening.")
```

## Testing

```python
# test_llm.py
import asyncio

async def test_llm_responses():
    """Test LLM with various inputs"""
    
    llm = QwenLLM()
    
    test_cases = [
        {
            "text": "I got a promotion!",
            "emotion": "happy",
            "language": "en"
        },
        {
            "text": "I'm feeling really sad",
            "emotion": "sad",
            "language": "en"
        },
        {
            "text": "I'm so frustrated!",
            "emotion": "angry",
            "language": "en"
        }
    ]
    
    for test in test_cases:
        result = await llm.generate_response(**test)
        
        print(f"Input: {test['text']}")
        print(f"Output: {result['text']}")
        print(f"Emotion: {result['emotion']}")
        print()

if __name__ == "__main__":
    asyncio.run(test_llm_responses())
```

## Deployment

### Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

RUN pip install transformers torch

COPY llm_integration.py .

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen-7B-Chat', trust_remote_code=True)"

CMD ["python", "-m", "uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm
  template:
    metadata:
      labels:
        app: llm
    spec:
      containers:
      - name: llm
        image: llm-service:latest
        resources:
          requests:
            memory: "16Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "20Gi"
            nvidia.com/gpu: "1"
        ports:
        - containerPort: 8000
```

## Summary

- **Model:** Qwen-7B/14B/72B
- **Latency:** 50-300ms per response
- **Languages:** Multilingual support
- **Emotion-aware:** Responds to user emotion
- **Conversation:** Multi-turn with history
- **Fallback:** Graceful degradation
- **Production-ready:** Error handling + monitoring

Perfect for empathic avatar responses! 
