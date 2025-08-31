// avatar-system.js - TalkingHead integration
// Import TalkingHead from local file
import { TalkingHead } from './talkinghead.mjs';

let talkingHead = null;

// Initialize TalkingHead
async function initTalkingHead() {
  try {
    const avatarContainer = document.getElementById('avatarContainer');
    if (!avatarContainer) {
      console.error('[TalkingHead] Avatar container not found');
      window.dispatchEvent(new CustomEvent('avatar:error', { 
        detail: { message: 'Avatar container not found' } 
      }));
      return false;
    }
    
    // Check if TalkingHead class is available
    if (typeof TalkingHead === 'undefined') {
      console.error('[TalkingHead] TalkingHead class not available');
      window.dispatchEvent(new CustomEvent('avatar:error', { 
        detail: { message: 'TalkingHead class not loaded' } 
      }));
      return false;
    }
    
    // Initialize TalkingHead with correct parameters
    talkingHead = new TalkingHead(avatarContainer, {
      ttsEndpoint: null, // We use Hume for TTS
      lipsyncLang: 'en',
      avatarMood: 'neutral'
    });

    // Load avatar
    const avatar = {
      url: './avatar.glb', // Make sure this file exists
      body: 'M', // Male avatar
      lipsyncLang: 'en'
    };

    // Show avatar with progress callback
    await talkingHead.showAvatar(avatar, (progress) => {
      const percent = Math.round(progress.loaded / progress.total * 100);
      console.log(`[TalkingHead] Loading avatar: ${percent}%`);
    });

    console.log('[TalkingHead] Avatar loaded successfully');
    
    // Start the avatar animation loop
    talkingHead.start();
    
    // Export to global scope
    window.talkingHead = talkingHead;
    
    return true;
    
  } catch (error) {
    console.error('[TalkingHead] Error loading avatar:', error);
    window.dispatchEvent(new CustomEvent('avatar:error', { 
      detail: { message: error.message } 
    }));
    return false;
  }
}

// Apply emotion to avatar
function applyEmotion(emotionName, intensity = 1.0) {
  if (!talkingHead) {
    console.warn('[TalkingHead] Not initialized');
    return;
  }
  
  console.log(`[TalkingHead] Applying emotion: ${emotionName} (${intensity.toFixed(3)})`);
  
  // Map Hume emotions to TalkingHead moods
  const emotionMoodMap = {
    joy: 'happy',
    happiness: 'happy',
    amusement: 'happy',
    excitement: 'happy',
    surprise: 'surprised',
    anger: 'angry',
    annoyance: 'angry',
    irritation: 'angry',
    sadness: 'sad',
    disappointment: 'sad',
    grief: 'sad',
    fear: 'fear',
    anxiety: 'fear',
    disgust: 'disgust',
    contempt: 'disgust',
    embarrassment: 'neutral',
    pride: 'happy',
    confidence: 'happy',
    neutral: 'neutral'
  };
  
  const mood = emotionMoodMap[emotionName] || 'neutral';
  
  try {
    // Set mood if intensity is significant
    if (intensity > 0.2) {
      talkingHead.setMood(mood);
      console.log(`[TalkingHead] Set mood to: ${mood}`);
    }
    
    // Apply specific morph targets for stronger emotions
    if (intensity > 0.3) {
      switch(emotionName) {
        case 'joy':
        case 'happiness':
        case 'amusement':
          talkingHead.setMorphTargetValue('mouthSmile', Math.min(0.8, intensity));
          break;
          
        case 'sadness':
        case 'disappointment':
          talkingHead.setMorphTargetValue('mouthFrown', Math.min(0.7, intensity));
          break;
          
        case 'anger':
        case 'annoyance':
          talkingHead.setMorphTargetValue('browDown', Math.min(0.8, intensity));
          break;
          
        case 'surprise':
          talkingHead.setMorphTargetValue('eyeWide', Math.min(0.7, intensity));
          talkingHead.setMorphTargetValue('mouthOpen', Math.min(0.4, intensity * 0.5));
          break;
      }
    }
  } catch (error) {
    console.error('[TalkingHead] Error applying emotion:', error);
  }
}

// Handle lipsync from audio analysis
function handleLipsync(event) {
  if (!talkingHead) return;
  
  const level = event.detail.level || 0;
  
  try {
    // TalkingHead has built-in lipsync, but we can enhance it
    if (level > 0.1) {
      // Use different visemes based on audio level
      const visemeIntensity = Math.min(1.0, level * 1.5);
      
      if (level > 0.6) {
        talkingHead.setMorphTargetValue('viseme_aa', visemeIntensity);
      } else if (level > 0.3) {
        talkingHead.setMorphTargetValue('viseme_E', visemeIntensity);
      } else {
        talkingHead.setMorphTargetValue('viseme_I', visemeIntensity * 0.8);
      }
    } else {
      // Reset visemes when not speaking
      talkingHead.setMorphTargetValue('viseme_aa', 0);
      talkingHead.setMorphTargetValue('viseme_E', 0);
      talkingHead.setMorphTargetValue('viseme_I', 0);
    }
  } catch (error) {
    console.error('[TalkingHead] Lipsync error:', error);
  }
}

// Handle facial expressions and head pose
function handleExpression(event) {
  if (!talkingHead) return;
  
  const data = event.detail;
  
  try {
    if (data.facePose) {
      const { yaw, pitch, roll } = data.facePose;
      
      // Convert angles to suitable range for TalkingHead
      const headRotateY = Math.max(-0.8, Math.min(0.8, yaw / 30));
      const headRotateX = Math.max(-0.6, Math.min(0.6, -pitch / 30));
      const headRotateZ = Math.max(-0.4, Math.min(0.4, roll / 60));
      
      // Apply head rotations smoothly
      talkingHead.setRotation('head', headRotateX, headRotateY, headRotateZ, 300);
    }
  } catch (error) {
    console.error('[TalkingHead] Expression error:', error);
  }
}

// Play audio with lipsync using TalkingHead's built-in system
function playAudioWithLipsync(audioData, words = [], wtimes = [], wdurations = []) {
  if (!talkingHead) {
    console.warn('[TalkingHead] Not initialized');
    return;
  }
  
  try {
    // TalkingHead expects specific audio format
    const audioObj = {
      audio: audioData, // Should be AudioBuffer or audio URL
      words: words,
      wtimes: wtimes,
      wdurations: wdurations
    };
    
    talkingHead.speakAudio(audioObj, { 
      lipsyncLang: 'en',
      smoothing: true
    });
    
    console.log(`[TalkingHead] Playing audio with ${words.length} words`);
  } catch (error) {
    console.error('[TalkingHead] Audio playback error:', error);
  }
}

// Stop speaking
function stopSpeaking() {
  if (!talkingHead) return;
  
  try {
    talkingHead.stopSpeaking();
    console.log('[TalkingHead] Stopped speaking');
  } catch (error) {
    console.error('[TalkingHead] Stop speaking error:', error);
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  console.log('[TalkingHead] Initializing system...');
  
  // Wait for all scripts to load
  setTimeout(async () => {
    const success = await initTalkingHead();
    
    if (success) {
      // Export functions globally for compatibility with app.js
      window.applyEmotion = applyEmotion;
      window.playAudioWithLipsync = playAudioWithLipsync;
      window.stopSpeaking = stopSpeaking;
      
      // Add event listeners
      window.addEventListener('octo:lipsync', handleLipsync);
      window.addEventListener('octo:expression', handleExpression);
      
      console.log('[TalkingHead] System ready!');
      
      // Dispatch ready event
      window.dispatchEvent(new CustomEvent('avatar:ready'));
      
    } else {
      console.error('[TalkingHead] Failed to initialize system');
      window.dispatchEvent(new CustomEvent('avatar:error'));
    }
  }, 1000);
});

// Export functions for use as ES6 module
export { 
  initTalkingHead, 
  applyEmotion, 
  playAudioWithLipsync, 
  stopSpeaking,
  handleLipsync,
  handleExpression 
};