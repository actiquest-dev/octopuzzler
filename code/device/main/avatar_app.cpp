class AvatarApplication {
public:
    AvatarApplication() {
        // Initialize components
        octopus = std::make_unique<OctopusAvatar>();
        
        eyes = std::make_unique<DualModeEyes>(
            140, 50,  // Left eye
            180, 50   // Right eye
        );
        
        rtc_client = std::make_unique<RTCClient>("wss://backend/ws");
        
        // Framebuffer (320x160 RGBA)
        framebuffer = new uint32_t[320 * 160];
        
        // State
        current_emotion = OctopusAvatar::Emotion::NEUTRAL;
        current_mouth_shape = MouthShapes::Shape::SIL;
        emotion_intensity = 1.0f;
        
        // Register callbacks
        rtc_client->on_animation_command([this](const AnimationCommand& cmd) {
            handle_animation_command(cmd);
        });
        
        rtc_client->on_audio_data([this](const uint8_t* data, size_t len) {
            audio_play(data, len);
        });
    }
    
    ~AvatarApplication() {
        delete[] framebuffer;
    }
    
    void run() {
        uint32_t last_frame_time = millis();
        
        while (true) {
            uint32_t frame_start = millis();
            float delta_time = (frame_start - last_frame_time) / 1000.0f;
            
            // 1. Process network messages
            rtc_client->process_messages();
            
            // 2. Process scheduled sync markers
            process_scheduled_markers();
            
            // 3. Update animation
            octopus->animate(delta_time);
            eyes->update(delta_time);
            update_blink(delta_time);
            
            // 4. Render frame
            render_frame();
            
            // 5. Display on dual LCD
            display_update(framebuffer);
            
            // 6. Frame timing (30 FPS target)
            uint32_t frame_time = millis() - frame_start;
            if (frame_time < 33) {
                vTaskDelay(pdMS_TO_TICKS(33 - frame_time));
            }
            
            last_frame_time = frame_start;
        }
    }
    
private:
    std::unique_ptr<OctopusAvatar> octopus;
    std::unique_ptr<DualModeEyes> eyes;
    std::unique_ptr<RTCClient> rtc_client;
    
    uint32_t* framebuffer;
    
    OctopusAvatar::Emotion current_emotion;
    MouthShapes::Shape current_mouth_shape;
    float emotion_intensity;
    
    // Sync markers queue
    struct ScheduledMarker {
        uint32_t execute_at;
        std::string action;
        // Action-specific data
        int mouth_shape;
        float gaze_x, gaze_y;
        uint32_t blink_duration_ms;
        
        bool operator>(const ScheduledMarker& other) const {
            return execute_at > other.execute_at;
        }
    };
    
    std::priority_queue
        ScheduledMarker,
        std::vector<ScheduledMarker>,
        std::greater<ScheduledMarker>
    > marker_queue;
    
    // Blink state
    float blink_timer;
    float blink_duration;
    
    void handle_animation_command(const AnimationCommand& cmd) {
        // Parse emotion
        if (cmd.body_animation == "happy") {
            current_emotion = OctopusAvatar::Emotion::HAPPY;
        } else if (cmd.body_animation == "sad") {
            current_emotion = OctopusAvatar::Emotion::SAD;
        } else if (cmd.body_animation == "angry") {
            current_emotion = OctopusAvatar::Emotion::ANGRY;
        } else if (cmd.body_animation == "surprised") {
            current_emotion = OctopusAvatar::Emotion::SURPRISED;
        } else if (cmd.body_animation == "curious") {
            current_emotion = OctopusAvatar::Emotion::CURIOUS;
        } else {
            current_emotion = OctopusAvatar::Emotion::NEUTRAL;
        }
        
        emotion_intensity = cmd.emotion_intensity;
        
        // Regenerate octopus
        octopus->generate(current_emotion, emotion_intensity);
        
        // Schedule sync markers
        uint32_t now = millis();
        for (const auto& marker : cmd.sync_markers) {
            ScheduledMarker scheduled;
            scheduled.execute_at = now + marker.time_ms;
            scheduled.action = marker.action;
            
            if (marker.action == "mouth_shape") {
                scheduled.mouth_shape = marker.mouth_shape;
            } else if (marker.action == "gaze") {
                scheduled.gaze_x = marker.gaze_x;
                scheduled.gaze_y = marker.gaze_y;
            } else if (marker.action == "blink") {
                scheduled.blink_duration_ms = marker.blink_duration_ms;
            }
            
            marker_queue.push(scheduled);
        }
    }
    
    void process_scheduled_markers() {
        uint32_t now = millis();
        
        while (!marker_queue.empty() && marker_queue.top().execute_at <= now) {
            ScheduledMarker marker = marker_queue.top();
            marker_queue.pop();
            
            if (marker.action == "mouth_shape") {
                current_mouth_shape = static_cast<MouthShapes::Shape>(
                    marker.mouth_shape
                );
            } else if (marker.action == "gaze") {
                eyes->update_backend_gaze(marker.gaze_x, marker.gaze_y);
            } else if (marker.action == "blink") {
                trigger_blink(marker.blink_duration_ms);
            }
        }
    }
    
    void update_blink(float delta_time) {
        if (blink_timer > 0) {
            blink_timer -= delta_time;
            
            float progress = 1.0f - (blink_timer / blink_duration);
            
            if (progress < 0.5f) {
                // Closing
                float state = 1.0f - (progress * 2.0f);
                eyes->set_blink(state);
            } else {
                // Opening
                float state = (progress - 0.5f) * 2.0f;
                eyes->set_blink(state);
            }
        } else {
            eyes->set_blink(1.0f);  // Fully open
        }
    }
    
    void trigger_blink(float duration_ms) {
        blink_timer = duration_ms / 1000.0f;
        blink_duration = duration_ms / 1000.0f;
    }
    
    void render_frame() {
        // Clear framebuffer
        memset(framebuffer, 0, 320 * 160 * 4);
        
        // 1. Background (solid color)
        render_background();
        
        // 2. Octopus body
        octopus->render(framebuffer);
        
        // 3. Mouth overlay
        render_mouth();
        
        // 4. Eyes
        render_eyes();
    }
    
    void render_background() {
        // Emotion-based background color
        uint32_t bg_color;
        
        switch (current_emotion) {
            case OctopusAvatar::Emotion::HAPPY:
                bg_color = 0xFFFFE5B4;  // Light peach
                break;
            case OctopusAvatar::Emotion::SAD:
                bg_color = 0xFFD0E0F0;  // Light blue
                break;
            case OctopusAvatar::Emotion::ANGRY:
                bg_color = 0xFFFFDDDD;  // Light red
                break;
            case OctopusAvatar::Emotion::CURIOUS:
                bg_color = 0xFFD5F5E3;  // Light teal
                break;
            default:
                bg_color = 0xFFF0F0F0;  // Light grey
        }
        
        for (int i = 0; i < 320 * 160; i++) {
            framebuffer[i] = bg_color;
        }
    }
    
    void render_mouth() {
        auto mouth_canvas = tvg::SwCanvas::gen();
        mouth_canvas->target(
            framebuffer, 320, 160, 320*4, 
            tvg::ColorSpace::ARGB8888
        );
        
        auto mouth = MouthShapes::generate(
            current_mouth_shape,
            160,  // Center X
            80,   // Mouth Y position
            50, 50, 50  // Dark color
        );
        
        mouth_canvas->push(std::move(mouth));
        mouth_canvas->draw();
        mouth_canvas->sync();
    }
    
    void render_eyes() {
        auto eye_canvas = tvg::SwCanvas::gen();
        eye_canvas->target(
            framebuffer, 320, 160, 320*4,
            tvg::ColorSpace::ARGB8888
        );
        
        eyes->render(eye_canvas.get());
        eye_canvas->draw();
        eye_canvas->sync();
    }
    
    void display_update(uint32_t* buffer) {
        // Convert RGBA8888  RGB565
        uint16_t* rgb565 = convert_to_rgb565(buffer, 320 * 160);
        
        // Send to dual LCD
        lcd_update_dual(rgb565, 320, 160);
        
        free(rgb565);
    }
    
    uint16_t* convert_to_rgb565(uint32_t* rgba, int count) {
        uint16_t* rgb565 = (uint16_t*)malloc(count * 2);
        
        for (int i = 0; i < count; i++) {
            uint32_t pixel = rgba[i];
            
            uint8_t r = (pixel >> 16) & 0xFF;
            uint8_t g = (pixel >> 8) & 0xFF;
            uint8_t b = pixel & 0xFF;
            
            uint16_t r5 = (r >> 3) & 0x1F;
            uint16_t g6 = (g >> 2) & 0x3F;
            uint16_t b5 = (b >> 3) & 0x1F;
            
            rgb565[i] = (r5 << 11) | (g6 << 5) | b5;
        }
        
        return rgb565;
    }
};
