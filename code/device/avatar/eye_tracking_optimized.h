/**
 * Optimized Eye Tracking System Header
 */

#ifndef EYE_TRACKING_OPTIMIZED_H
#define EYE_TRACKING_OPTIMIZED_H

#include <cstdint>

// Forward declarations
class BlazeFaceDetector;
class MediaPipeFaceMesh;
class GazeCalculator;

/**
 * Eye tracking statistics
 */
struct EyeTrackingStats {
    uint32_t total_frames;
    uint32_t face_detections;
    uint32_t eye_tracks;
    bool has_valid_face;
    float current_gaze_x;
    float current_gaze_y;
    bool is_blinking;
};

/**
 * Optimized Eye Tracking System
 * 
 * Two-stage pipeline:
 * 1. Face detection (BlazeFace) @ 2.5 FPS
 * 2. Eye tracking (MediaPipe) @ 5 FPS
 * 
 * Output interpolated to 30 FPS for smooth display.
 * 
 * Usage:
 * ------
 * EyeTrackingSystem tracker;
 * tracker.initialize();
 * 
 * // Camera loop (15 FPS)
 * while (true) {
 *     uint8_t* frame = camera.capture();
 *     tracker.process_frame(frame, 640, 480);
 * }
 * 
 * // Display loop (30 FPS)
 * while (true) {
 *     float alpha = 0.5f;  // Interpolation factor
 *     tracker.interpolate(alpha);
 *     
 *     float gaze_x, gaze_y;
 *     tracker.get_gaze(gaze_x, gaze_y);
 *     
 *     avatar.set_gaze(gaze_x, gaze_y);
 *     avatar.render();
 * }
 */
class EyeTrackingSystem {
public:
    /**
     * Constructor
     */
    EyeTrackingSystem();
    
    /**
     * Destructor
     */
    ~EyeTrackingSystem();
    
    /**
     * Initialize eye tracking system
     * 
     * Loads TFLite models:
     * - BlazeFace: 2.5MB
     * - MediaPipe Face Mesh Lite: 30MB
     * 
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Process camera frame
     * 
     * Call at camera rate (e.g., 15 FPS).
     * Automatically skips frames for optimization.
     * 
     * @param frame_data RGB888 frame data
     * @param width Frame width
     * @param height Frame height
     * @return true if face detected and tracked
     */
    bool process_frame(const uint8_t* frame_data, int width, int height);
    
    /**
     * Get current gaze direction (smoothed)
     * 
     * @param gaze_x Output horizontal gaze (-1.0 to 1.0)
     * @param gaze_y Output vertical gaze (-1.0 to 1.0)
     */
    void get_gaze(float& gaze_x, float& gaze_y);
    
    /**
     * Check if user is blinking
     * 
     * @return true if eyes are closed
     */
    bool is_blinking();
    
    /**
     * Interpolate gaze for smooth animation
     * 
     * Call at display rate (e.g., 30 FPS).
     * 
     * @param alpha Interpolation factor (0.0 to 1.0)
     */
    void interpolate(float alpha);
    
    /**
     * Reset tracking state
     */
    void reset();
    
    /**
     * Get tracking statistics
     */
    EyeTrackingStats get_stats() const;

private:
    // Components
    BlazeFaceDetector* blazeface_;
    MediaPipeFaceMesh* facemesh_;
    GazeCalculator* gaze_calc_;
    
    // Frame counter
    uint32_t frame_count_;
    
    // Face detection state
    bool has_valid_face_;
    int face_roi_x_;
    int face_roi_y_;
    int face_roi_width_;
    int face_roi_height_;
    
    // Working buffers
    uint8_t* face_roi_;  // 200×200×3 RGB
    
    // Gaze state
    float current_gaze_x_;
    float current_gaze_y_;
    float target_gaze_x_;
    float target_gaze_y_;
    float interpolation_factor_;
    
    // Blink state
    bool is_blinking_;
    
    // Smoothing history (5 frames)
    static constexpr int GAZE_HISTORY_SIZE = 5;
    float gaze_history_x_[GAZE_HISTORY_SIZE];
    float gaze_history_y_[GAZE_HISTORY_SIZE];
    int gaze_history_index_;
    
    // Helper methods
    void detect_face(const uint8_t* frame_data, int width, int height);
    void track_eyes(const uint8_t* frame_data, int width, int height);
    void extract_and_resize_roi(const uint8_t* frame_data, int frame_width, int frame_height);
    void add_to_history(float gaze_x, float gaze_y);
    void smooth_gaze();
};

#endif // EYE_TRACKING_OPTIMIZED_H