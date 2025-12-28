/**
 * Example: Optimized Eye Tracking Usage
 */

#include "eye_tracking_optimized.h"
#include "octopus_avatar.h"
#include <cstdio>
#include <unistd.h>

// Mock camera
struct MockCamera {
    uint8_t* capture(int& width, int& height) {
        width = 640;
        height = 480;
        
        // Allocate dummy frame
        static uint8_t frame[640 * 480 * 3];
        return frame;
    }
};

void example_basic_tracking()
{
    printf("=== Basic Eye Tracking ===\n");
    
    // Initialize eye tracking
    EyeTrackingSystem tracker;
    if (!tracker.initialize()) {
        printf("Failed to initialize eye tracking\n");
        return;
    }
    
    MockCamera camera;
    
    // Process 30 frames (2 seconds @ 15 FPS)
    for (int i = 0; i < 30; i++) {
        int width, height;
        uint8_t* frame = camera.capture(width, height);
        
        bool detected = tracker.process_frame(frame, width, height);
        
        if (detected) {
            float gaze_x, gaze_y;
            tracker.get_gaze(gaze_x, gaze_y);
            
            printf("Frame %d: gaze=(%.2f, %.2f), blink=%d\n",
                   i, gaze_x, gaze_y, tracker.is_blinking());
        }
        
        usleep(66666);  // 66ms = ~15 FPS
    }
    
    // Print stats
    EyeTrackingStats stats = tracker.get_stats();
    printf("\nStats:\n");
    printf("  Total frames: %u\n", stats.total_frames);
    printf("  Face detections: %u\n", stats.face_detections);
    printf("  Eye tracks: %u\n", stats.eye_tracks);
}

void example_with_avatar()
{
    printf("\n=== Eye Tracking + Avatar ===\n");
    
    // Initialize systems
    EyeTrackingSystem tracker;
    tracker.initialize();
    
    OctopusAvatar avatar(320, 160);
    avatar.set_emotion(EMOTION_CURIOUS, 0.8f);
    
    MockCamera camera;
    
    // Dual-thread simulation
    // Thread 1: Camera @ 15 FPS
    // Thread 2: Display @ 30 FPS
    
    for (int frame = 0; frame < 60; frame++) {
        // Camera thread (every other frame)
        if (frame % 2 == 0) {
            int width, height;
            uint8_t* cam_frame = camera.capture(width, height);
            tracker.process_frame(cam_frame, width, height);
        }
        
        // Display thread (every frame)
        float alpha = 0.5f;  // Interpolation
        tracker.interpolate(alpha);
        
        float gaze_x, gaze_y;
        tracker.get_gaze(gaze_x, gaze_y);
        
        avatar.set_gaze(gaze_x, gaze_y);
        avatar.set_blink(tracker.is_blinking());
        avatar.update(1.0f / 30.0f);
        avatar.render();
        
        if (frame % 10 == 0) {
            printf("Display frame %d: gaze=(%.2f, %.2f)\n",
                   frame, gaze_x, gaze_y);
        }
        
        usleep(33333);  // 33ms = 30 FPS
    }
    
    printf("✓ 60 frames rendered\n");
}

int main()
{
    example_basic_tracking();
    example_with_avatar();
    
    printf("\n✓ All examples completed\n");
    return 0;
}