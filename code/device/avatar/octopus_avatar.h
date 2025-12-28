/**
 * Octopus Avatar Header
 * 
 * Public interface for octopus avatar rendering.
 */

#ifndef OCTOPUS_AVATAR_H
#define OCTOPUS_AVATAR_H

#include <cstdint>

// Forward declaration
namespace tvg {
    class SwCanvas;
}

/**
 * Emotion states
 */
enum Emotion {
    EMOTION_HAPPY = 0,
    EMOTION_SAD = 1,
    EMOTION_ANGRY = 2,
    EMOTION_NEUTRAL = 3,
    EMOTION_SURPRISED = 4,
    EMOTION_CURIOUS = 5,
    EMOTION_FEARFUL = 6
};

/**
 * RGB color structure
 */
struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

/**
 * Emotion color scheme
 */
struct EmotionColors {
    Color body_primary;
    Color body_secondary;
    Color eyes;
    Color pupils;
};

/**
 * Octopus Avatar Renderer
 * 
 * Renders SpongeBob-style flat vector octopus using ThorVG.
 * 
 * Usage:
 * ------
 * OctopusAvatar avatar(320, 160);
 * 
 * // Set emotion
 * avatar.set_emotion(EMOTION_HAPPY, 0.8f);
 * 
 * // Update animation
 * avatar.update(delta_time);
 * 
 * // Render frame
 * avatar.render();
 * 
 * // Get framebuffer (RGBA8888)
 * const uint8_t* fb = avatar.get_framebuffer();
 * 
 * // Or convert to RGB565 for LCD
 * uint16_t rgb565_buffer[320 * 160];
 * avatar.convert_to_rgb565(rgb565_buffer);
 */
class OctopusAvatar {
public:
    /**
     * Constructor
     * 
     * @param width Canvas width (pixels)
     * @param height Canvas height (pixels)
     */
    OctopusAvatar(uint32_t width, uint32_t height);
    
    /**
     * Destructor
     */
    ~OctopusAvatar();
    
    /**
     * Set target emotion
     * 
     * Smoothly transitions to new emotion over 0.5 seconds.
     * 
     * @param emotion Target emotion
     * @param intensity Emotion intensity (0.0 to 1.0)
     */
    void set_emotion(Emotion emotion, float intensity = 1.0f);
    
    /**
     * Set gaze direction
     * 
     * @param x Horizontal gaze (-1.0 = left, 1.0 = right)
     * @param y Vertical gaze (-1.0 = up, 1.0 = down)
     */
    void set_gaze(float x, float y);
    
    /**
     * Set blink state
     * 
     * @param blinking True if eyes should be closed
     */
    void set_blink(bool blinking);
    
    /**
     * Update animation state
     * 
     * Call every frame with time delta.
     * 
     * @param delta_time Time since last update (seconds)
     */
    void update(float delta_time);
    
    /**
     * Render current frame
     * 
     * Draws to internal framebuffer.
     */
    void render();
    
    /**
     * Get framebuffer (RGBA8888)
     * 
     * @return Pointer to framebuffer (width × height × 4 bytes)
     */
    const uint8_t* get_framebuffer() const;
    
    /**
     * Convert framebuffer to RGB565
     * 
     * @param output Output buffer (width × height × 2 bytes)
     */
    void convert_to_rgb565(uint16_t* output) const;
    
    /**
     * Get current emotion
     */
    Emotion get_current_emotion() const;
    
    /**
     * Get emotion intensity
     */
    float get_emotion_intensity() const;

private:
    // Canvas dimensions
    uint32_t width_;
    uint32_t height_;
    
    // Center point
    float center_x_;
    float center_y_;
    
    // Emotion state
    Emotion current_emotion_;
    Emotion target_emotion_;
    float emotion_blend_;       // 0.0 to 1.0 (transition progress)
    float emotion_intensity_;
    
    // Animation
    float animation_time_;
    
    // Gaze
    float gaze_x_;
    float gaze_y_;
    
    // Blink
    bool is_blinking_;
    
    // ThorVG
    tvg::SwCanvas* canvas_;
    uint8_t* framebuffer_;
    
    // Helper methods
    EmotionColors get_blended_colors() const;
    void render_body(const EmotionColors& colors);
    void render_tentacle(int tentacle_index, const EmotionColors& colors);
    void render_eyes(const EmotionColors& colors);
    void render_single_eye(
        float x, float y,
        float eye_radius, float pupil_radius,
        const EmotionColors& colors
    );
};

#endif // OCTOPUS_AVATAR_H