/**
 * Example: Dual-Mode Eyes Usage
 */

#include "dual_mode_eyes.h"
#include "octopus_avatar.h"
#include <cstdio>
#include <unistd.h>

// Mock camera
struct MockCamera {
    uint8_t* capture() {
        static uint8_t frame[640 * 480 * 3];
        return frame;
    }
};

// Mock backend commands
class MockBackend {
public:
    void simulate_gaze_commands(DualModeEyes& eyes) {
        // Simulate backend sending gaze commands
        
        // Command 1: Look left
        printf("\n[Backend] Command: Look left\n");
        eyes.set_backend_gaze(-0.5f, 0.0f);
        
        sleep(1);  // Backend mode for 1 second
        
        // Command 2: Look right
        printf("[Backend] Command: Look right\n");
        eyes.set_backend_gaze(0.5f, 0.0f);
        
        sleep(1);
        
        // Stop sending commands
        // After 2 seconds, should fall back to local
        printf("[Backend] Stopped sending commands\n");
    }
};

void example_basic_usage()
{
    printf("=== Dual-Mode Eyes Basic Usage ===\n");
    
    // Initialize systems
    DualModeEyes eyes;
    if (!eyes.initialize()) {
        printf("Failed to initialize dual-mode eyes\n");
        return;
    }
    
    OctopusAvatar avatar(320, 160);
    avatar.set_emotion(EMOTION_CURIOUS, 0.8f);
    
    MockCamera camera;
    
    // Simulation loop (60 FPS)
    const float DELTA_TIME = 1.0f / 60.0f;
    
    for (int frame = 0; frame < 180; frame++) {  // 3 seconds
        // Process camera frame (15 FPS)
        if (frame % 4 == 0) {
            uint8_t* cam_frame = camera.capture();
            eyes.process_camera_frame(cam_frame, 640, 480);
        }
        
        // Update eyes
        eyes.update(DELTA_TIME);
        
        // Get gaze
        float gaze_x, gaze_y;
        eyes.get_gaze(gaze_x, gaze_y);
        
        // Update avatar
        avatar.set_gaze(gaze_x, gaze_y);
        avatar.update(DELTA_TIME);
        avatar.render();
        
        // Print status every 30 frames (0.5s)
        if (frame % 30 == 0) {
            DualModeEyesStats stats = eyes.get_stats();
            printf("Frame %3d: mode=%s, gaze=(%.2f, %.2f)\n",
                   frame,
                   stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND",
                   stats.output_gaze_x,
                   stats.output_gaze_y);
        }
        
        usleep(16666);  // ~60 FPS
    }
}

void example_mode_switching()
{
    printf("\n=== Mode Switching Example ===\n");
    
    DualModeEyes eyes;
    eyes.initialize();
    
    MockCamera camera;
    MockBackend backend;
    
    const float DELTA_TIME = 1.0f / 60.0f;
    
    // Phase 1: Local mode (2 seconds)
    printf("\nPhase 1: Local mode\n");
    for (int frame = 0; frame < 120; frame++) {
        if (frame % 4 == 0) {
            eyes.process_camera_frame(camera.capture(), 640, 480);
        }
        eyes.update(DELTA_TIME);
        usleep(16666);
    }
    
    DualModeEyesStats stats = eyes.get_stats();
    printf("After phase 1: mode=%s\n",
           stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND");
    
    // Phase 2: Backend commands (2 seconds)
    printf("\nPhase 2: Backend commands\n");
    
    for (int i = 0; i < 4; i++) {
        float gaze_x = (i % 2 == 0) ? -0.5f : 0.5f;
        eyes.set_backend_gaze(gaze_x, 0.0f);
        
        printf("Backend command %d: gaze_x=%.1f\n", i + 1, gaze_x);
        
        // Hold for 0.5 seconds
        for (int frame = 0; frame < 30; frame++) {
            eyes.update(DELTA_TIME);
            usleep(16666);
        }
    }
    
    stats = eyes.get_stats();
    printf("After phase 2: mode=%s\n",
           stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND");
    
    // Phase 3: Timeout back to local (3 seconds)
    printf("\nPhase 3: Waiting for timeout...\n");
    
    for (int frame = 0; frame < 180; frame++) {
        if (frame % 4 == 0) {
            eyes.process_camera_frame(camera.capture(), 640, 480);
        }
        eyes.update(DELTA_TIME);
        
        // Check mode every second
        if (frame % 60 == 0) {
            stats = eyes.get_stats();
            printf("  %.1fs: mode=%s, time_since_backend=%ums\n",
                   frame / 60.0f,
                   stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND",
                   stats.time_since_backend_command);
        }
        
        usleep(16666);
    }
    
    stats = eyes.get_stats();
    printf("After phase 3: mode=%s\n",
           stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND");
}

void example_smooth_blending()
{
    printf("\n=== Smooth Blending Example ===\n");
    
    DualModeEyes eyes;
    eyes.initialize();
    
    const float DELTA_TIME = 1.0f / 60.0f;
    
    // Start in local mode (looking center)
    printf("Starting in LOCAL mode (center gaze)\n");
    
    for (int i = 0; i < 30; i++) {
        eyes.update(DELTA_TIME);
        usleep(16666);
    }
    
    // Backend command: Look right
    printf("\nBackend command: Look right (0.8, 0.0)\n");
    eyes.set_backend_gaze(0.8f, 0.0f);
    
    // Watch blending
    printf("\nBlending to backend gaze:\n");
    for (int frame = 0; frame < 60; frame++) {
        eyes.update(DELTA_TIME);
        
        if (frame % 10 == 0) {
            DualModeEyesStats stats = eyes.get_stats();
            printf("  Frame %2d: output=(%.3f, %.3f), blend=%.2f\n",
                   frame,
                   stats.output_gaze_x,
                   stats.output_gaze_y,
                   stats.blend_factor);
        }
        
        usleep(16666);
    }
    
    // Wait for timeout
    printf("\nWaiting for timeout (2 seconds)...\n");
    for (int i = 0; i < 120; i++) {
        eyes.update(DELTA_TIME);
        usleep(16666);
    }
    
    // Watch blending back to local
    printf("\nBlending back to local:\n");
    for (int frame = 0; frame < 60; frame++) {
        eyes.update(DELTA_TIME);
        
        if (frame % 10 == 0) {
            DualModeEyesStats stats = eyes.get_stats();
            printf("  Frame %2d: output=(%.3f, %.3f), blend=%.2f, mode=%s\n",
                   frame,
                   stats.output_gaze_x,
                   stats.output_gaze_y,
                   stats.blend_factor,
                   stats.current_mode == MODE_LOCAL ? "LOCAL" : "BACKEND");
        }
        
        usleep(16666);
    }
}

int main()
{
    example_basic_usage();
    example_mode_switching();
    example_smooth_blending();
    
    printf("\n✓ All examples completed\n");
    return 0;
}