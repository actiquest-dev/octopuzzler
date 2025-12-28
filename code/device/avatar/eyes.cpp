class Eyes {
public:
    Eyes(float left_x, float left_y, float right_x, float right_y)
        : left_x(left_x), left_y(left_y),
          right_x(right_x), right_y(right_y),
          gaze_x(0.0f), gaze_y(0.0f),
          blink_state(1.0f) {}
    
    void set_gaze(float x, float y) {
        // Clamp to safe range
        gaze_x = std::max(-0.3f, std::min(0.3f, x));
        gaze_y = std::max(-0.3f, std::min(0.3f, y));
    }
    
    void set_blink(float state) {
        // 0.0 = fully closed, 1.0 = fully open
        blink_state = std::max(0.0f, std::min(1.0f, state));
    }
    
    void render(tvg::Canvas* canvas) {
        render_eye(canvas, left_x, left_y);
        render_eye(canvas, right_x, right_y);
    }
    
private:
    float left_x, left_y;
    float right_x, right_y;
    float gaze_x, gaze_y;
    float blink_state;
    
    void render_eye(tvg::Canvas* canvas, float x, float y) {
        // Eye white (sclera)
        auto sclera = tvg::Shape::gen();
        
        float eye_height = 12.0f * blink_state;  // Blink affects height
        sclera->appendCircle(x, y, 12, eye_height);
        
        sclera->fill(255, 255, 255);  // White
        sclera->stroke(50, 50, 50);   // Dark outline
        sclera->stroke(2.0f);
        
        canvas->push(std::move(sclera));
        
        // Only draw pupil if eye is open enough
        if (blink_state > 0.3f) {
            // Pupil position (follows gaze)
            float pupil_x = x + (gaze_x * 6.0f);
            float pupil_y = y + (gaze_y * 5.0f);
            
            // Pupil
            auto pupil = tvg::Shape::gen();
            pupil->appendCircle(pupil_x, pupil_y, 5, 5);
            pupil->fill(50, 50, 50);  // Dark
            canvas->push(std::move(pupil));
            
            // Shine (reflection)
            auto shine = tvg::Shape::gen();
            shine->appendCircle(pupil_x - 2, pupil_y - 2, 2, 2);
            shine->fill(255, 255, 255);
            canvas->push(std::move(shine));
        }
    }
};
