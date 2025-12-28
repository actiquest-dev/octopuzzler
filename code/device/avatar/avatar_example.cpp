/**
 * Example: How to use OctopusAvatar
 */

#include "octopus_avatar.h"
#include <cstdio>

void example_basic_usage()
{
    printf("=== Basic Avatar Usage ===\n");
    
    // Create avatar (320×160 for dual LCD)
    OctopusAvatar avatar(320, 160);
    
    // Set emotion
    avatar.set_emotion(EMOTION_HAPPY, 0.8f);
    
    // Animation loop (60 FPS)
    const float FPS = 60.0f;
    const float DELTA_TIME = 1.0f / FPS;
    
    for (int frame = 0; frame < 60; frame++) {
        // Update animation
        avatar.update(DELTA_TIME);
        
        // Render
        avatar.render();
        
        // Get framebuffer
        const uint8_t* fb = avatar.get_framebuffer();
        
        // TODO: Send to LCD
        // display.write(fb);
    }
    
    printf("Rendered 60 frames\n");
}

void example_emotion_transition()
{
    printf("\n=== Emotion Transition ===\n");
    
    OctopusAvatar avatar(320, 160);
    
    // Start with neutral
    avatar.set_emotion(EMOTION_NEUTRAL, 1.0f);
    
    const float DELTA_TIME = 1.0f / 30.0f;
    
    // Render for 1 second
    for (int i = 0; i < 30; i++) {
        avatar.update(DELTA_TIME);
        avatar.render();
    }
    
    // Transition to happy
    printf("Transitioning to happy...\n");
    avatar.set_emotion(EMOTION_HAPPY, 0.9f);
    
    // Render transition (0.5s = 15 frames)
    for (int i = 0; i < 15; i++) {
        avatar.update(DELTA_TIME);
        avatar.render();
    }
    
    printf("Transition complete\n");
}

void example_gaze_control()
{
    printf("\n=== Gaze Control ===\n");
    
    OctopusAvatar avatar(320, 160);
    avatar.set_emotion(EMOTION_CURIOUS, 0.7f);
    
    const float DELTA_TIME = 1.0f / 30.0f;
    
    // Look left
    printf("Looking left...\n");
    avatar.set_gaze(-0.5f, 0.0f);
    for (int i = 0; i < 30; i++) {
        avatar.update(DELTA_TIME);
        avatar.render();
    }
    
    // Look right
    printf("Looking right...\n");
    avatar.set_gaze(0.5f, 0.0f);
    for (int i = 0; i < 30; i++) {
        avatar.update(DELTA_TIME);
        avatar.render();
    }
    
    // Look center
    printf("Looking center...\n");
    avatar.set_gaze(0.0f, 0.0f);
    for (int i = 0; i < 30; i++) {
        avatar.update(DELTA_TIME);
        avatar.render();
    }
}

void example_rgb565_conversion()
{
    printf("\n=== RGB565 Conversion ===\n");
    
    OctopusAvatar avatar(320, 160);
    avatar.set_emotion(EMOTION_HAPPY, 1.0f);
    
    avatar.update(0.0f);
    avatar.render();
    
    // Allocate RGB565 buffer
    uint16_t* rgb565_buffer = new uint16_t[320 * 160];
    
    // Convert
    avatar.convert_to_rgb565(rgb565_buffer);
    
    printf("Converted to RGB565\n");
    printf("Buffer size: %zu bytes\n", 320 * 160 * 2);
    
    // TODO: Send to LCD
    // lcd.write_buffer(rgb565_buffer, 320 * 160);
    
    delete[] rgb565_buffer;
}

int main()
{
    example_basic_usage();
    example_emotion_transition();
    example_gaze_control();
    example_rgb565_conversion();
    
    printf("\n✓ All examples completed\n");
    return 0;
}