/**
 * Dual-Mode Eyes Controller Header
 */

#ifndef DUAL_MODE_EYES_H
#define DUAL_MODE_EYES_H

#include <cstdint>

// Forward declaration
class EyeTrackingSystem;

/**
 * Eye control mode
 */
enum EyeMode {
    MODE_LOCAL,      // Using on-device eye tracking
    MODE_BACKEND,    // Using backend gaze commands
    MODE_HYBRID      // Blending between modes (unused)
};

/**
 * Dual-mode eyes statistics
 */
struct DualModeEyesStats {
    EyeMode current_mode;
    float local_gaze_x;
    float local_gaze_y;
    float backend_gaze_x;
    float backend_gaze_y;
    float output_gaze_x;
    float output_gaze_y;
    float blend_factor;
    uint32_t time_since_backend_command;
};

/**
 * Dual-Mode Eyes Controller
 * 
 * Manages eye gaze with automatic mode switching:
 * - Local mode: Uses MediaPipe eye tracking
 * - Backend mode: Uses gaze commands from backend (overrides local)
 * - Smooth transitions between modes
 * 
 * Behavior:
 * - Backend commands activate backend mode for 2 seconds
 * - After 2 seconds without commands, falls back to local mode
 * - Smooth interpolation during transitions
 * 
 * Usage:
 * ------
 * DualModeEyes eyes;
 * eyes.initialize();
 * 
 * // Game loop
 * while (true) {
 *     // Process camera
 *     eyes.process_camera_frame(camera_frame, 640, 480);
 *     
 *     // Update (blending)
 *     eyes.update(delta_time);
 *     
 *     // Get output gaze
 *     float gaze_x, gaze_y;
 *     eyes.get_gaze(gaze_x, gaze_y);
 *     
 *     avatar.set_gaze(gaze_x, gaze_y);
 * }
 * 
 * // Backend commands (from WebRTC)
 * void on_backend_gaze_command(float x, float y) {
 *     eyes.set_backend_gaze(x, y);
 * }
 */
class DualModeEyes {
public:
    /**
     * Constructor
     */
    DualModeEyes();
    
    /**
     * Destructor
     */
    ~DualModeEyes();
    
    /**
     * Initialize dual-mode eyes
     * 
     * Initializes local eye tracking system.
     * 
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Update eye state
     * 
     * Call every frame. Handles:
     * - Mode timeout checking
     * - Gaze blending
     * 
     * @param delta_time Time since last update (seconds)
     */
    void update(float delta_time);
    
    /**
     * Process camera frame for local tracking
     * 
     * Call at camera rate (e.g., 15 FPS).
     * Updates local gaze from eye tracking.
     * 
     * @param frame RGB888 frame data
     * @param width Frame width
     * @param height Frame height
     */
    void process_camera_frame(const uint8_t* frame, int width, int height);
    
    /**
     * Set backend gaze command
     * 
     * Call when receiving gaze command from backend.
     * Switches to backend mode and resets timeout.
     * 
     * @param gaze_x Horizontal gaze (-1.0 to 1.0)
     * @param gaze_y Vertical gaze (-1.0 to 1.0)
     */
    void set_backend_gaze(float gaze_x, float gaze_y);
    
    /**
     * Get current output gaze (blended)
     * 
     * Returns smoothly blended gaze between local and backend.
     * 
     * @param gaze_x Output horizontal gaze
     * @param gaze_y Output vertical gaze
     */
    void get_gaze(float& gaze_x, float& gaze_y);
    
    /**
     * Get current mode
     * 
     * @return Current eye mode (LOCAL or BACKEND)
     */
    EyeMode get_current_mode() const;
    
    /**
     * Force local mode
     * 
     * Immediately switch to local tracking mode.
     * Use when ending session or on error.
     */
    void force_local_mode();
    
    /**
     * Get statistics
     * 
     * @return Current state statistics
     */
    DualModeEyesStats get_stats() const;

private:
    // Eye tracking system
    EyeTrackingSystem* eye_tracker_;
    
    // Current mode
    EyeMode current_mode_;
    
    // Local tracking state
    float local_gaze_x_;
    float local_gaze_y_;
    
    // Backend command state
    float backend_gaze_x_;
    float backend_gaze_y_;
    uint32_t last_backend_command_time_;
    
    // Output state (blended)
    float output_gaze_x_;
    float output_gaze_y_;
    float blend_factor_;  // 0.0 = current, 1.0 = target
    
    // Helper methods
    void update_local_tracking();
    void check_mode_timeout();
    void blend_gaze(float delta_time);
    float clamp(float value, float min, float max) const;
    uint32_t get_current_time_ms() const;
};

#endif // DUAL_MODE_EYES_H