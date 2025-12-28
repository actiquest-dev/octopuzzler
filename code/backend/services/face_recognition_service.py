"""
Face Recognition Service

Uses InsightFace (ArcFace) for face recognition and FAISS for similarity search.
Stores only embeddings (512-dim vectors) - cannot reverse to face images (privacy-safe).

Features:
- Face detection (RetinaFace)
- Embedding extraction (ArcFace, 512-dim)
- Similarity search (FAISS, L2 distance)
- User database management
- Privacy-preserving (embeddings only)

Author: Octopus AI Team
Date: December 28, 2025
Version: 1.1
"""

import insightface
import numpy as np
import faiss
import pickle
import os
import logging
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """
    Face Recognition Service using InsightFace + FAISS
    
    Architecture:
    ------------
    1. RetinaFace: Detect faces in image
    2. ArcFace: Extract 512-dim embedding
    3. FAISS: Search for similar embeddings
    4. User DB: Store user profiles
    
    Privacy:
    --------
    - Only embeddings stored (512 floats)
    - Cannot reverse to original face image
    - No raw face images saved
    
    Performance:
    -----------
    - Face detection: ~20ms
    - Embedding extraction: ~25ms
    - FAISS search (1K users): <1ms
    - Total: ~50ms per recognition
    
    Usage:
    ------
    service = FaceRecognitionService()
    
    # Register user
    result = await service.register_user("Michael", face_image_bytes)
    
    # Recognize user
    result = await service.recognize(face_image_bytes)
    if result['is_known']:
        print(f"Hello {result['user_name']}!")
    """
    
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        database_path: str = "data/users.db",
        faiss_index_path: str = "data/users.faiss",
        recognition_threshold: float = 0.6
    ):
        """
        Initialize face recognition service
        
        Args:
            model_pack: InsightFace model pack (buffalo_l recommended)
            database_path: Path to user database file
            faiss_index_path: Path to FAISS index file
            recognition_threshold: Distance threshold for recognition (lower = stricter)
        """
        logger.info("Initializing Face Recognition Service...")
        
        self.database_path = database_path
        self.faiss_index_path = faiss_index_path
        self.recognition_threshold = recognition_threshold
        
        # Initialize InsightFace
        logger.info(f"Loading InsightFace model: {model_pack}")
        self.app = insightface.app.FaceAnalysis(
            name=model_pack,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✓ InsightFace loaded")
        
        # Initialize FAISS index
        self.dimension = 512  # ArcFace embedding dimension
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        
        # User database
        self.users: Dict[str, Dict] = {}
        self.user_id_to_faiss_idx: Dict[str, int] = {}
        self.faiss_idx_to_user_id: Dict[int, str] = {}
        
        # Load existing database
        self._load_database()
        
        logger.info(
            f"✓ Face Recognition Service ready "
            f"(threshold={recognition_threshold}, users={len(self.users)})"
        )
    
    async def recognize(self, face_image: bytes) -> Dict:
        """
        Recognize face from image
        
        Args:
            face_image: JPEG image bytes
        
        Returns:
            {
                "is_known": bool,
                "user_id": str (if known),
                "user_name": str (if known),
                "confidence": float (0.0-1.0),
                "distance": float,
                "preferences": dict,
                "conversation_topics": list,
                "analytics": dict,
                "games": dict
            }
        """
        try:
            # Decode image
            img_array = self._decode_image(face_image)
            
            # Detect faces
            faces = self.app.get(img_array)
            
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return {
                    "is_known": False,
                    "error": "No face detected"
                }
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using first")
            
            # Get embedding (512-dim vector)
            embedding = faces[0].embedding
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            
            # Search in FAISS
            if self.faiss_index.ntotal == 0:
                logger.info("No users registered yet")
                return {"is_known": False}
            
            distances, indices = self.faiss_index.search(
                embedding.reshape(1, -1).astype('float32'),
                k=1
            )
            
            distance = float(distances[0][0])
            faiss_idx = int(indices[0][0])
            
            logger.info(f"Recognition: distance={distance:.3f}, threshold={self.recognition_threshold}")
            
            # Check threshold
            if distance < self.recognition_threshold:
                # Match found
                user_id = self.faiss_idx_to_user_id[faiss_idx]
                profile = self.users[user_id]
                
                # Update last seen
                profile['last_seen'] = datetime.now().isoformat()
                self._save_database()
                
                # Calculate confidence (0.0 to 1.0)
                # Lower distance = higher confidence
                confidence = max(0.0, 1.0 - (distance / self.recognition_threshold))
                
                logger.info(f"✓ Recognized user: {profile['name']} (confidence={confidence:.2f})")
                
                return {
                    "is_known": True,
                    "user_id": user_id,
                    "user_name": profile['name'],
                    "confidence": confidence,
                    "distance": distance,
                    "preferences": profile.get('preferences', {}),
                    "conversation_topics": profile.get('conversation_topics', []),
                    "analytics": profile.get('analytics', {}),
                    "games": profile.get('games', {})
                }
            else:
                # No match
                logger.info(f"Unknown user (distance={distance:.3f} > threshold)")
                return {
                    "is_known": False,
                    "distance": distance
                }
        
        except Exception as e:
            logger.error(f"Recognition failed: {e}", exc_info=True)
            return {
                "is_known": False,
                "error": str(e)
            }
    
    async def register_user(self, name: str, face_image: bytes) -> Dict:
        """
        Register new user
        
        Args:
            name: User's name
            face_image: JPEG image bytes
        
        Returns:
            {
                "success": bool,
                "user_id": str,
                "message": str
            }
        """
        try:
            # Decode image
            img_array = self._decode_image(face_image)
            
            # Detect faces
            faces = self.app.get(img_array)
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "message": "No face detected in image"
                }
            
            if len(faces) > 1:
                return {
                    "success": False,
                    "message": "Multiple faces detected. Please provide image with single face."
                }
            
            # Get embedding
            embedding = faces[0].embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Check if user already exists (similar embedding)
            if self.faiss_index.ntotal > 0:
                distances, indices = self.faiss_index.search(
                    embedding.reshape(1, -1).astype('float32'),
                    k=1
                )
                
                if distances[0][0] < 0.3:  # Very similar (likely same person)
                    existing_user_id = self.faiss_idx_to_user_id[indices[0][0]]
                    existing_name = self.users[existing_user_id]['name']
                    return {
                        "success": False,
                        "message": f"User already registered as '{existing_name}'"
                    }
            
            # Generate user ID
            user_id = f"user_{len(self.users) + 1:04d}"
            
            # Add to FAISS index
            faiss_idx = self.faiss_index.ntotal
            self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
            
            # Update mappings
            self.user_id_to_faiss_idx[user_id] = faiss_idx
            self.faiss_idx_to_user_id[faiss_idx] = user_id
            
            # Create user profile
            profile = {
                "user_id": user_id,
                "name": name,
                "embedding": embedding.tolist(),  # Store for backup
                "registered_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                
                # Personalization
                "preferences": {
                    "avatar_emotion_intensity": 0.8,
                    "response_style": "balanced",  # "concise" | "balanced" | "detailed"
                    "language": "auto",
                    "voice_speed": 1.0
                },
                
                # Social
                "conversation_topics": [],
                "conversation_history": [],
                
                # Analytics
                "analytics": {
                    "total_sessions": 0,
                    "total_time_seconds": 0,
                    "sessions_per_day": {},
                    "favorite_topics": [],
                    "avg_session_duration": 0
                },
                
                # Games
                "games": {},
                
                # Recommendations
                "recommendations": {
                    "interests": [],
                    "skill_level": "beginner",
                    "suggested_topics": [],
                    "suggested_games": []
                }
            }
            
            self.users[user_id] = profile
            
            # Save to disk
            self._save_database()
            
            logger.info(f"✓ Registered new user: {user_id} - {name}")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": f"Successfully registered {name}"
            }
        
        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}"
            }
    
    async def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """
        Update user profile
        
        Args:
            user_id: User ID
            updates: Dictionary of fields to update
        
        Returns:
            Success status
        """
        if user_id not in self.users:
            logger.warning(f"User not found: {user_id}")
            return False
        
        profile = self.users[user_id]
        
        # Update allowed fields
        allowed_fields = [
            'preferences', 'conversation_topics', 'conversation_history',
            'analytics', 'games', 'recommendations'
        ]
        
        for field in allowed_fields:
            if field in updates:
                if isinstance(updates[field], dict) and isinstance(profile.get(field), dict):
                    # Deep merge for dict fields
                    profile[field].update(updates[field])
                else:
                    profile[field] = updates[field]
        
        self._save_database()
        logger.info(f"Updated profile for user: {user_id}")
        return True
    
    async def add_conversation_topic(self, user_id: str, topic: str) -> bool:
        """
        Add conversation topic to user profile
        
        Args:
            user_id: User ID
            topic: Topic to add
        
        Returns:
            Success status
        """
        if user_id not in self.users:
            return False
        
        profile = self.users[user_id]
        
        if topic not in profile['conversation_topics']:
            profile['conversation_topics'].append(topic)
            
            # Keep only last 20 topics
            if len(profile['conversation_topics']) > 20:
                profile['conversation_topics'] = profile['conversation_topics'][-20:]
            
            self._save_database()
            logger.debug(f"Added topic '{topic}' for user {user_id}")
        
        return True
    
    async def update_analytics(
        self,
        user_id: str,
        session_duration: int,
        topics: List[str]
    ) -> bool:
        """
        Update user analytics after session
        
        Args:
            user_id: User ID
            session_duration: Session duration in seconds
            topics: Topics discussed in session
        
        Returns:
            Success status
        """
        if user_id not in self.users:
            return False
        
        profile = self.users[user_id]
        analytics = profile['analytics']
        
        # Update session count
        analytics['total_sessions'] += 1
        analytics['total_time_seconds'] += session_duration
        
        # Update daily count
        today = datetime.now().strftime("%Y-%m-%d")
        sessions_per_day = analytics.get('sessions_per_day', {})
        sessions_per_day[today] = sessions_per_day.get(today, 0) + 1
        analytics['sessions_per_day'] = sessions_per_day
        
        # Update favorite topics
        for topic in topics:
            if topic not in analytics['favorite_topics']:
                analytics['favorite_topics'].append(topic)
        
        # Keep only top 10 favorite topics
        if len(analytics['favorite_topics']) > 10:
            analytics['favorite_topics'] = analytics['favorite_topics'][-10:]
        
        # Update average session duration
        if analytics['total_sessions'] > 0:
            analytics['avg_session_duration'] = (
                analytics['total_time_seconds'] / analytics['total_sessions']
            )
        
        self._save_database()
        logger.debug(f"Updated analytics for user {user_id}")
        return True
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        profile = self.users.get(user_id)
        
        if profile:
            # Return copy without embedding (privacy)
            profile_copy = profile.copy()
            profile_copy.pop('embedding', None)
            return profile_copy
        
        return None
    
    async def list_users(self) -> List[Dict]:
        """List all registered users (without embeddings)"""
        users_list = []
        
        for user_id, profile in self.users.items():
            users_list.append({
                "user_id": user_id,
                "name": profile['name'],
                "registered_at": profile['registered_at'],
                "last_seen": profile['last_seen'],
                "total_sessions": profile['analytics']['total_sessions']
            })
        
        return users_list
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete user (admin only)
        
        Args:
            user_id: User ID to delete
        
        Returns:
            Success status
        """
        if user_id not in self.users:
            return False
        
        # Remove from FAISS index
        # Note: FAISS doesn't support deletion, so we rebuild index
        logger.info(f"Rebuilding FAISS index (removing user {user_id})...")
        
        new_index = faiss.IndexFlatL2(self.dimension)
        new_mappings_id_to_idx = {}
        new_mappings_idx_to_id = {}
        
        for uid, profile in self.users.items():
            if uid != user_id:
                embedding = np.array(profile['embedding']).astype('float32')
                new_idx = new_index.ntotal
                new_index.add(embedding.reshape(1, -1))
                new_mappings_id_to_idx[uid] = new_idx
                new_mappings_idx_to_id[new_idx] = uid
        
        self.faiss_index = new_index
        self.user_id_to_faiss_idx = new_mappings_id_to_idx
        self.faiss_idx_to_user_id = new_mappings_idx_to_id
        
        # Remove from users dict
        del self.users[user_id]
        
        # Save
        self._save_database()
        
        logger.info(f"✓ Deleted user: {user_id}")
        return True
    
    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode JPEG to numpy array"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _save_database(self):
        """Save user database to disk"""
        # Create directory if needed
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        db_data = {
            'users': self.users,
            'user_id_to_faiss_idx': self.user_id_to_faiss_idx,
            'faiss_idx_to_user_id': self.faiss_idx_to_user_id
        }
        
        # Save pickle
        with open(self.database_path, 'wb') as f:
            pickle.dump(db_data, f)
        
        # Save FAISS index separately
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        
        logger.debug(f"Database saved ({len(self.users)} users)")
    
    def _load_database(self):
        """Load user database from disk"""
        if not os.path.exists(self.database_path):
            logger.info("No existing user database found. Starting fresh.")
            return
        
        try:
            # Load pickle
            with open(self.database_path, 'rb') as f:
                db_data = pickle.load(f)
            
            self.users = db_data['users']
            self.user_id_to_faiss_idx = db_data['user_id_to_faiss_idx']
            self.faiss_idx_to_user_id = db_data['faiss_idx_to_user_id']
            
            # Load FAISS index
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
            
            logger.info(f"✓ Loaded {len(self.users)} users from database")
        
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            logger.info("Starting with empty database")
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "total_users": len(self.users),
            "faiss_index_size": self.faiss_index.ntotal,
            "recognition_threshold": self.recognition_threshold,
            "embedding_dimension": self.dimension
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_face_recognition():
        """Test face recognition service"""
        
        # Initialize service
        service = FaceRecognitionService()
        
        print("\n=== Face Recognition Service Test ===")
        print(f"Stats: {service.get_stats()}")
        
        # Test 1: Register user (requires test image)
        print("\n=== Test 1: Register User ===")
        try:
            with open("test_face.jpg", "rb") as f:
                face_image = f.read()
            
            result = await service.register_user("Test User", face_image)
            print(f"Registration: {result}")
            
            if result['success']:
                user_id = result['user_id']
                
                # Test 2: Recognize same user
                print("\n=== Test 2: Recognize User ===")
                result = await service.recognize(face_image)
                print(f"Recognition: {result}")
                
                if result['is_known']:
                    print(f"✓ Recognized: {result['user_name']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    print(f"  Distance: {result['distance']:.3f}")
                
                # Test 3: Update profile
                print("\n=== Test 3: Update Profile ===")
                success = await service.update_user_profile(
                    user_id,
                    {
                        "preferences": {
                            "response_style": "detailed"
                        }
                    }
                )
                print(f"Update: {success}")
                
                # Test 4: Add conversation topic
                print("\n=== Test 4: Add Topics ===")
                await service.add_conversation_topic(user_id, "AI")
                await service.add_conversation_topic(user_id, "Technology")
                print("Topics added")
                
                # Test 5: Update analytics
                print("\n=== Test 5: Update Analytics ===")
                await service.update_analytics(
                    user_id,
                    session_duration=180,
                    topics=["AI", "Programming"]
                )
                print("Analytics updated")
                
                # Test 6: Get profile
                print("\n=== Test 6: Get Profile ===")
                profile = await service.get_user_profile(user_id)
                print(f"Profile:")
                print(f"  Name: {profile['name']}")
                print(f"  Topics: {profile['conversation_topics']}")
                print(f"  Sessions: {profile['analytics']['total_sessions']}")
                
                # Test 7: List users
                print("\n=== Test 7: List Users ===")
                users = await service.list_users()
                for user in users:
                    print(f"  {user['user_id']}: {user['name']} ({user['total_sessions']} sessions)")
        
        except FileNotFoundError:
            print("No test image found (test_face.jpg)")
            print("Skipping registration/recognition tests")
        
        # Stats
        print("\n=== Final Stats ===")
        print(service.get_stats())
    
    asyncio.run(test_face_recognition())