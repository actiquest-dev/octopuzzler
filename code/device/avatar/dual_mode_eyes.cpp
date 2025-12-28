/**
 * Dual-Mode Eyes Controller
 * 
 * Manages eye gaze with two modes:
 * 1. Local mode: Uses on-device eye tracking (MediaPipe)
 * 2. Backend mode: Uses gaze commands from backend
 * 
 * Hybrid behavior:
 * - Backend commands override local tracking for 2 seconds
 * - Falls back to local tracking after timeout
 * - Smooth interpolation between modes
 * 
 * Author: Octopus AI Team
 * Date: December 28, 2025
 * Version: 1.0
 */

#include "dual_mode_eyes.h"
#include "eye_tracking_optimized.h"
#include <cmath>
#include <cstdio>

// Mode timeout (2 seconds)
constexpr uint32_t BACKEND_MODE_TIMEOUT_MS = 2000;

// Interpolation speeds
constexpr float BLEND_SPEED_FAST = 0.2f;   // Fast blend to backend commands
constexpr float BLEND_SPEED_SLOW = 0.1f;   // Slow blend back to local

DualModeEyes::DualModeEyes()
    : eye_tracker_(nullptr)
    , current_mode_(MODE_LOCAL)
    , local_gaze_x_(0.0f)
    , local_gaze_y_(0.0f)
    , backend_gaze_x_(0.0f)
    , backend_gaze_y_(0.0f)
    , output_gaze_x_(0.0f)
    , output_gaze_y_(0.0f)
    , blend_factor_(0.0f)
    , last_backend_command_time_(0)
{
}

DualModeEyes::~DualModeEyes()
{
    if (eye_tracker_) {
        delete eye_tracker_;
    }
}

bool DualModeEyes::initialize()
{
    printf("[DualModeEyes] Initializing dual-mode eye controller...\n");
    
    // Initialize local eye tracking
    eye_tracker_ = new EyeTrackingSystem();
    if (!eye_tracker_->initialize()) {
        printf("[DualModeEyes] ERROR: Failed to initialize eye tracker\n");
        return false;
    }
    
    printf("[DualModeEyes] ✓ Initialized (mode: LOCAL)\n");
    return true;
}

void DualModeEyes::update(float delta_time)
{
    // Update local eye tracking
    update_local_tracking();
    
    // Check backend mode timeout
    check_mode_timeout();
    
    // Blend between modes
    blend_gaze(delta_time);
}

void DualModeEyes::process_camera_frame(const uint8_t* frame, int width, int height)
{
    // Process frame with local eye tracker
    if (eye_tracker_) {
        eye_tracker_->process_frame(frame, width, height);
    }
}

void DualModeEyes::set_backend_gaze(float gaze_x, float gaze_y)
{
    // Update backend gaze command
    backend_gaze_x_ = gaze_x;
    backend_gaze_y_ = gaze_y;
    
    // Switch to backend mode
    if (current_mode_ != MODE_BACKEND) {
        printf("[DualModeEyes] Switching to BACKEND mode\n");
        current_mode_ = MODE_BACKEND;
        blend_factor_ = 0.0f;  // Start blending to backend
    }
    
    // Reset timeout
    last_backend_command_time_ = get_current_time_ms();
}

void DualModeEyes::get_gaze(float& gaze_x, float& gaze_y)
{
    // Return blended output
    gaze_x = output_gaze_x_;
    gaze_y = output_gaze_y_;
}

EyeMode DualModeEyes::get_current_mode() const
{
    return current_mode_;
}

void DualModeEyes::force_local_mode()
{
    if (current_mode_ != MODE_LOCAL) {
        printf("[DualModeEyes] Forcing LOCAL mode\n");
        current_mode_ = MODE_LOCAL;
        blend_factor_ = 1.0f;  // Immediate switch
    }
}

void DualModeEyes::update_local_tracking()
{
    // Get gaze from local eye tracker
    if (eye_tracker_) {
        eye_tracker_->get_gaze(local_gaze_x_, local_gaze_y_);
    }
}

void DualModeEyes::check_mode_timeout()
{
    if (current_mode_ == MODE_BACKEND) {
        uint32_t current_time = get_current_time_ms();
        uint32_t time_since_command = current_time - last_backend_command_time_;
        
        if (time_since_command > BACKEND_MODE_TIMEOUT_MS) {
            // Timeout - switch back to local mode
            printf("[DualModeEyes] Backend timeout, switching to LOCAL mode\n");
            current_mode_ = MODE_LOCAL;
            blend_factor_ = 0.0f;  // Start blending to local
        }
    }
}

void DualModeEyes::blend_gaze(float delta_time)
{
    // Determine target gaze based on mode
    float target_x, target_y;
    float blend_speed;
    
    if (current_mode_ == MODE_BACKEND) {
        // Target: backend commands
        target_x = backend_gaze_x_;
        target_y = backend_gaze_y_;
        blend_speed = BLEND_SPEED_FAST;  // Fast blend to backend
    } else {
        // Target: local tracking
        target_x = local_gaze_x_;
        target_y = local_gaze_y_;
        blend_speed = BLEND_SPEED_SLOW;  // Slow blend to local
    }
    
    // Update blend factor
    blend_factor_ += blend_speed;
    if (blend_factor_ > 1.0f) {
        blend_factor_ = 1.0f;
    }
    
    // Interpolate output gaze
    output_gaze_x_ = output_gaze_x_ * (1.0f - blend_factor_) + target_x * blend_factor_;
    output_gaze_y_ = output_gaze_y_ * (1.0f - blend_factor_) + target_y * blend_factor_;
    
    // Clamp to valid range
    output_gaze_x_ = clamp(output_gaze_x_, -1.0f, 1.0f);
    output_gaze_y_ = clamp(output_gaze_y_, -1.0f, 1.0f);
}

float DualModeEyes::clamp(float value, float min, float max) const
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

uint32_t DualModeEyes::get_current_time_ms() const
{
    // TODO: Implement real-time clock access
    // For now, use FreeRTOS tick count
    // return xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    // Mock implementation
    static uint32_t mock_time = 0;
    mock_time += 16;  // ~60 FPS
    return mock_time;
}

DualModeEyesStats DualModeEyes::get_stats() const
{
    DualModeEyesStats stats;
    
    stats.current_mode = current_mode_;
    stats.local_gaze_x = local_gaze_x_;
    stats.local_gaze_y = local_gaze_y_;
    stats.backend_gaze_x = backend_gaze_x_;
    stats.backend_gaze_y = backend_gaze_y_;
    stats.output_gaze_x = output_gaze_x_;
    stats.output_gaze_y = output_gaze_y_;
    stats.blend_factor = blend_factor_;
    
    if (current_mode_ == MODE_BACKEND) {
        uint32_t current_time = get_current_time_ms();
        stats.time_since_backend_command = current_time - last_backend_command_time_;
    } else {
        stats.time_since_backend_command = 0;
    }
    
    return stats;
}