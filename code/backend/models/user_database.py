"""
User Database Models

Data models and schemas for user profiles, sessions, and analytics.
Provides structured access to user data with validation.

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ResponseStyle(Enum):
    """Response style preferences"""
    CONCISE = "concise"      # 2-3 sentences
    BALANCED = "balanced"    # 3-5 sentences (default)
    DETAILED = "detailed"    # Comprehensive with examples


class SkillLevel(Enum):
    """User skill level for recommendations"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class UserPreferences:
    """
    User preferences for avatar behavior
    
    Attributes:
        avatar_emotion_intensity: How strongly avatar expresses emotions (0.0-1.0)
        response_style: Verbosity level for responses
        language: Preferred language ("auto" for detection)
        voice_speed: TTS playback speed (0.5-2.0, 1.0 = normal)
        enable_vision: Allow avatar to see via camera
        enable_games: Enable game suggestions
    """
    avatar_emotion_intensity: float = 0.8
    response_style: ResponseStyle = ResponseStyle.BALANCED
    language: str = "auto"
    voice_speed: float = 1.0
    enable_vision: bool = True
    enable_games: bool = True
    
    def __post_init__(self):
        """Validate preferences"""
        if not 0.0 <= self.avatar_emotion_intensity <= 1.0:
            raise ValueError("avatar_emotion_intensity must be between 0.0 and 1.0")
        
        if not 0.5 <= self.voice_speed <= 2.0:
            raise ValueError("voice_speed must be between 0.5 and 2.0")
        
        if isinstance(self.response_style, str):
            self.response_style = ResponseStyle(self.response_style)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "avatar_emotion_intensity": self.avatar_emotion_intensity,
            "response_style": self.response_style.value,
            "language": self.language,
            "voice_speed": self.voice_speed,
            "enable_vision": self.enable_vision,
            "enable_games": self.enable_games
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreferences':
        """Create from dictionary"""
        return cls(
            avatar_emotion_intensity=data.get("avatar_emotion_intensity", 0.8),
            response_style=data.get("response_style", ResponseStyle.BALANCED),
            language=data.get("language", "auto"),
            voice_speed=data.get("voice_speed", 1.0),
            enable_vision=data.get("enable_vision", True),
            enable_games=data.get("enable_games", True)
        )


@dataclass
class SessionAnalytics:
    """
    Analytics for user sessions
    
    Tracks usage patterns and engagement metrics.
    """
    total_sessions: int = 0
    total_time_seconds: int = 0
    sessions_per_day: Dict[str, int] = field(default_factory=dict)
    favorite_topics: List[str] = field(default_factory=list)
    avg_session_duration: float = 0.0
    most_common_emotion: str = "neutral"
    vision_queries_count: int = 0
    
    def add_session(
        self,
        duration_seconds: int,
        topics: List[str],
        emotions: List[str],
        vision_used: bool = False
    ):
        """
        Add session data
        
        Args:
            duration_seconds: Session duration
            topics: Topics discussed
            emotions: Emotions detected during session
            vision_used: Whether vision was used
        """
        self.total_sessions += 1
        self.total_time_seconds += duration_seconds
        
        # Update daily count
        today = datetime.now().strftime("%Y-%m-%d")
        self.sessions_per_day[today] = self.sessions_per_day.get(today, 0) + 1
        
        # Update favorite topics
        for topic in topics:
            if topic and topic not in self.favorite_topics:
                self.favorite_topics.append(topic)
        
        # Keep only top 10 topics
        if len(self.favorite_topics) > 10:
            self.favorite_topics = self.favorite_topics[-10:]
        
        # Update average duration
        if self.total_sessions > 0:
            self.avg_session_duration = self.total_time_seconds / self.total_sessions
        
        # Update most common emotion
        if emotions:
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            self.most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Track vision usage
        if vision_used:
            self.vision_queries_count += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_sessions": self.total_sessions,
            "total_time_seconds": self.total_time_seconds,
            "sessions_per_day": self.sessions_per_day,
            "favorite_topics": self.favorite_topics,
            "avg_session_duration": self.avg_session_duration,
            "most_common_emotion": self.most_common_emotion,
            "vision_queries_count": self.vision_queries_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionAnalytics':
        """Create from dictionary"""
        return cls(
            total_sessions=data.get("total_sessions", 0),
            total_time_seconds=data.get("total_time_seconds", 0),
            sessions_per_day=data.get("sessions_per_day", {}),
            favorite_topics=data.get("favorite_topics", []),
            avg_session_duration=data.get("avg_session_duration", 0.0),
            most_common_emotion=data.get("most_common_emotion", "neutral"),
            vision_queries_count=data.get("vision_queries_count", 0)
        )


@dataclass
class GameStats:
    """
    Statistics for a specific game
    
    Attributes:
        game_name: Name of the game
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        total_played: Total games played
        win_rate: Win percentage
        highest_score: Best score achieved (if applicable)
        achievements: List of achievements unlocked
    """
    game_name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_played: int = 0
    win_rate: float = 0.0
    highest_score: Optional[int] = None
    achievements: List[str] = field(default_factory=list)
    
    def add_result(self, result: str, score: Optional[int] = None):
        """
        Add game result
        
        Args:
            result: "win", "loss", or "draw"
            score: Optional score value
        """
        self.total_played += 1
        
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        elif result == "draw":
            self.draws += 1
        
        # Update win rate
        if self.total_played > 0:
            self.win_rate = self.wins / self.total_played
        
        # Update highest score
        if score is not None:
            if self.highest_score is None or score > self.highest_score:
                self.highest_score = score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "game_name": self.game_name,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_played": self.total_played,
            "win_rate": self.win_rate,
            "highest_score": self.highest_score,
            "achievements": self.achievements
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameStats':
        """Create from dictionary"""
        return cls(
            game_name=data.get("game_name", ""),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            total_played=data.get("total_played", 0),
            win_rate=data.get("win_rate", 0.0),
            highest_score=data.get("highest_score"),
            achievements=data.get("achievements", [])
        )


@dataclass
class Recommendations:
    """
    Personalized recommendations for user
    
    Based on conversation history, interests, and skill level.
    """
    interests: List[str] = field(default_factory=list)
    skill_level: SkillLevel = SkillLevel.BEGINNER
    suggested_topics: List[str] = field(default_factory=list)
    suggested_games: List[str] = field(default_factory=list)
    learning_path: List[str] = field(default_factory=list)
    
    def update_interests(self, new_interests: List[str]):
        """Add new interests"""
        for interest in new_interests:
            if interest and interest not in self.interests:
                self.interests.append(interest)
        
        # Keep only top 15 interests
        if len(self.interests) > 15:
            self.interests = self.interests[-15:]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "interests": self.interests,
            "skill_level": self.skill_level.value if isinstance(self.skill_level, SkillLevel) else self.skill_level,
            "suggested_topics": self.suggested_topics,
            "suggested_games": self.suggested_games,
            "learning_path": self.learning_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Recommendations':
        """Create from dictionary"""
        return cls(
            interests=data.get("interests", []),
            skill_level=SkillLevel(data.get("skill_level", "beginner")),
            suggested_topics=data.get("suggested_topics", []),
            suggested_games=data.get("suggested_games", []),
            learning_path=data.get("learning_path", [])
        )


@dataclass
class UserProfile:
    """
    Complete user profile
    
    Contains all user data including:
    - Basic info (ID, name, registration date)
    - Face embedding (for recognition)
    - Preferences
    - Conversation history
    - Analytics
    - Game statistics
    - Recommendations
    
    Attributes:
        user_id: Unique user identifier
        name: User's display name
        embedding: Face embedding (512-dim vector)
        registered_at: ISO timestamp of registration
        last_seen: ISO timestamp of last interaction
        preferences: User preferences
        conversation_topics: Recent conversation topics (max 20)
        conversation_history: Recent messages (max 50)
        analytics: Session analytics
        games: Game statistics by game name
        recommendations: Personalized recommendations
    """
    user_id: str
    name: str
    embedding: List[float]
    registered_at: str = ""
    last_seen: str = ""
    preferences: UserPreferences = field(default_factory=UserPreferences)
    conversation_topics: List[str] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    analytics: SessionAnalytics = field(default_factory=SessionAnalytics)
    games: Dict[str, GameStats] = field(default_factory=dict)
    recommendations: Recommendations = field(default_factory=Recommendations)
    
    def __post_init__(self):
        """Initialize timestamps if not set"""
        if not self.registered_at:
            self.registered_at = datetime.now().isoformat()
        
        if not self.last_seen:
            self.last_seen = datetime.now().isoformat()
        
        # Convert dict preferences to UserPreferences if needed
        if isinstance(self.preferences, dict):
            self.preferences = UserPreferences.from_dict(self.preferences)
        
        # Convert dict analytics to SessionAnalytics if needed
        if isinstance(self.analytics, dict):
            self.analytics = SessionAnalytics.from_dict(self.analytics)
        
        # Convert dict games to GameStats if needed
        if self.games:
            new_games = {}
            for game_name, stats in self.games.items():
                if isinstance(stats, dict):
                    new_games[game_name] = GameStats.from_dict(stats)
                else:
                    new_games[game_name] = stats
            self.games = new_games
        
        # Convert dict recommendations if needed
        if isinstance(self.recommendations, dict):
            self.recommendations = Recommendations.from_dict(self.recommendations)
    
    def add_conversation_topic(self, topic: str):
        """Add conversation topic (max 20)"""
        if topic and topic not in self.conversation_topics:
            self.conversation_topics.append(topic)
            
            # Keep only last 20 topics
            if len(self.conversation_topics) > 20:
                self.conversation_topics = self.conversation_topics[-20:]
    
    def add_conversation_message(self, role: str, content: str):
        """Add message to conversation history (max 50)"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def add_game_result(
        self,
        game_name: str,
        result: str,
        score: Optional[int] = None
    ):
        """
        Add game result
        
        Args:
            game_name: Name of the game
            result: "win", "loss", or "draw"
            score: Optional score
        """
        if game_name not in self.games:
            self.games[game_name] = GameStats(game_name=game_name)
        
        self.games[game_name].add_result(result, score)
    
    def update_last_seen(self):
        """Update last seen timestamp"""
        self.last_seen = datetime.now().isoformat()
    
    def to_dict(self, include_embedding: bool = False) -> Dict:
        """
        Convert to dictionary
        
        Args:
            include_embedding: Include face embedding (default: False for privacy)
        
        Returns:
            Dictionary representation
        """
        data = {
            "user_id": self.user_id,
            "name": self.name,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "preferences": self.preferences.to_dict(),
            "conversation_topics": self.conversation_topics,
            "conversation_history": self.conversation_history,
            "analytics": self.analytics.to_dict(),
            "games": {
                game_name: stats.to_dict()
                for game_name, stats in self.games.items()
            },
            "recommendations": self.recommendations.to_dict()
        }
        
        if include_embedding:
            data["embedding"] = self.embedding
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create from dictionary"""
        return cls(
            user_id=data["user_id"],
            name=data["name"],
            embedding=data.get("embedding", []),
            registered_at=data.get("registered_at", ""),
            last_seen=data.get("last_seen", ""),
            preferences=UserPreferences.from_dict(data.get("preferences", {})),
            conversation_topics=data.get("conversation_topics", []),
            conversation_history=data.get("conversation_history", []),
            analytics=SessionAnalytics.from_dict(data.get("analytics", {})),
            games={
                game_name: GameStats.from_dict(stats)
                for game_name, stats in data.get("games", {}).items()
            },
            recommendations=Recommendations.from_dict(data.get("recommendations", {}))
        )
    
    def to_json(self, include_embedding: bool = False, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(include_embedding), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserProfile':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# Example usage and tests
if __name__ == "__main__":
    print("=== User Database Models Test ===")
    
    # Test 1: Create new user profile
    print("\n=== Test 1: Create User Profile ===")
    
    profile = UserProfile(
        user_id="user_0001",
        name="Michael",
        embedding=[0.1] * 512  # Dummy embedding
    )
    
    print(f"Created user: {profile.name} ({profile.user_id})")
    print(f"Registered: {profile.registered_at}")
    
    # Test 2: Update preferences
    print("\n=== Test 2: Update Preferences ===")
    
    profile.preferences.response_style = ResponseStyle.DETAILED
    profile.preferences.voice_speed = 1.2
    
    print(f"Response style: {profile.preferences.response_style.value}")
    print(f"Voice speed: {profile.preferences.voice_speed}")
    
    # Test 3: Add conversation topics
    print("\n=== Test 3: Add Topics ===")
    
    topics = ["AI", "Python", "Machine Learning", "TensorFlow", "PyTorch"]
    for topic in topics:
        profile.add_conversation_topic(topic)
    
    print(f"Topics: {profile.conversation_topics}")
    
    # Test 4: Add conversation messages
    print("\n=== Test 4: Add Messages ===")
    
    profile.add_conversation_message("user", "Hello!")
    profile.add_conversation_message("assistant", "Hi! How can I help you?")
    profile.add_conversation_message("user", "Tell me about AI")
    
    print(f"Conversation history: {len(profile.conversation_history)} messages")
    for msg in profile.conversation_history:
        print(f"  [{msg['role']}] {msg['content']}")
    
    # Test 5: Add session analytics
    print("\n=== Test 5: Session Analytics ===")
    
    profile.analytics.add_session(
        duration_seconds=180,
        topics=["AI", "Machine Learning"],
        emotions=["happy", "curious", "happy"],
        vision_used=True
    )
    
    profile.analytics.add_session(
        duration_seconds=240,
        topics=["Python", "Programming"],
        emotions=["neutral", "curious"],
        vision_used=False
    )
    
    print(f"Total sessions: {profile.analytics.total_sessions}")
    print(f"Total time: {profile.analytics.total_time_seconds}s")
    print(f"Avg duration: {profile.analytics.avg_session_duration:.1f}s")
    print(f"Most common emotion: {profile.analytics.most_common_emotion}")
    print(f"Vision queries: {profile.analytics.vision_queries_count}")
    
    # Test 6: Add game results
    print("\n=== Test 6: Game Results ===")
    
    profile.add_game_result("rock_paper_scissors", "win")
    profile.add_game_result("rock_paper_scissors", "loss")
    profile.add_game_result("rock_paper_scissors", "win")
    profile.add_game_result("trivia", "win", score=850)
    profile.add_game_result("trivia", "win", score=920)
    
    for game_name, stats in profile.games.items():
        print(f"{game_name}:")
        print(f"  Played: {stats.total_played}")
        print(f"  Win rate: {stats.win_rate:.1%}")
        if stats.highest_score:
            print(f"  Highest score: {stats.highest_score}")
    
    # Test 7: Recommendations
    print("\n=== Test 7: Recommendations ===")
    
    profile.recommendations.update_interests(["AI", "Robotics", "IoT"])
    profile.recommendations.skill_level = SkillLevel.INTERMEDIATE
    profile.recommendations.suggested_topics = ["Computer Vision", "NLP"]
    profile.recommendations.suggested_games = ["chess", "sudoku"]
    
    print(f"Interests: {profile.recommendations.interests}")
    print(f"Skill level: {profile.recommendations.skill_level.value}")
    print(f"Suggested topics: {profile.recommendations.suggested_topics}")
    
    # Test 8: Serialization
    print("\n=== Test 8: Serialization ===")
    
    # To dict (without embedding for privacy)
    profile_dict = profile.to_dict(include_embedding=False)
    print(f"Dict keys: {list(profile_dict.keys())}")
    print(f"Has embedding: {'embedding' in profile_dict}")
    
    # To JSON
    json_str = profile.to_json(include_embedding=False, indent=None)
    print(f"JSON length: {len(json_str)} chars")
    
    # From JSON
    profile_restored = UserProfile.from_json(json_str)
    print(f"Restored user: {profile_restored.name}")
    print(f"Restored topics: {profile_restored.conversation_topics}")
    
    # Test 9: Full dict with embedding
    print("\n=== Test 9: Full Export ===")
    
    full_dict = profile.to_dict(include_embedding=True)
    print(f"Has embedding: {'embedding' in full_dict}")
    print(f"Embedding size: {len(full_dict['embedding'])} dims")
    
    print("\n✓ All tests passed!")