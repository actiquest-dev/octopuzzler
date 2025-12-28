/**
 * Octopus Avatar Renderer
 * 
 * SpongeBob-style flat vector graphics using ThorVG.
 * Renders procedurally-generated octopus with 8 tentacles.
 * 
 * Features:
 * - 7 emotion states (happy, sad, angry, neutral, surprised, curious, fearful)
 * - Color-based emotion expression
 * - Wiggling tentacle animation
 * - Smooth emotion transitions
 * 
 * Output: 320×160 RGBA8888 → RGB565 for LCD
 * 
 * Author: Octopus AI Team
 * Date: December 28, 2025
 * Version: 1.0
 */

#include "octopus_avatar.h"
#include <thorvg.h>
#include <cmath>
#include <cstring>

// ThorVG namespace
using namespace tvg;

// Emotion color schemes (RGB)
static const EmotionColors EMOTION_COLORS[7] = {
    // Happy - Bright Yellow/Orange
    {
        .body_primary = {255, 220, 0},      // Yellow
        .body_secondary = {255, 180, 0},    // Orange
        .eyes = {255, 255, 255},            // White
        .pupils = {50, 50, 50}              // Dark gray
    },
    // Sad - Blue/Purple
    {
        .body_primary = {100, 150, 220},    // Light blue
        .body_secondary = {70, 100, 180},   // Darker blue
        .eyes = {200, 220, 255},            // Pale blue
        .pupils = {40, 60, 100}             // Dark blue
    },
    // Angry - Red/Dark Red
    {
        .body_primary = {220, 50, 50},      // Red
        .body_secondary = {180, 30, 30},    // Dark red
        .eyes = {255, 200, 200},            // Light red
        .pupils = {80, 10, 10}              // Very dark red
    },
    // Neutral - Purple/Lavender
    {
        .body_primary = {150, 100, 200},    // Purple
        .body_secondary = {120, 70, 170},   // Darker purple
        .eyes = {255, 255, 255},            // White
        .pupils = {60, 40, 80}              // Dark purple
    },
    // Surprised - Bright Pink/Magenta
    {
        .body_primary = {255, 100, 200},    // Pink
        .body_secondary = {220, 70, 170},   // Magenta
        .eyes = {255, 255, 255},            // White
        .pupils = {50, 50, 50}              // Dark gray
    },
    // Curious - Green/Teal
    {
        .body_primary = {100, 220, 150},    // Green
        .body_secondary = {70, 180, 120},   // Darker green
        .eyes = {255, 255, 255},            // White
        .pupils = {40, 80, 60}              // Dark green
    },
    // Fearful - Gray/Dark Gray
    {
        .body_primary = {150, 150, 170},    // Light gray
        .body_secondary = {100, 100, 120},  // Dark gray
        .eyes = {230, 230, 240},            // Pale gray
        .pupils = {50, 50, 60}              // Very dark gray
    }
};

OctopusAvatar::OctopusAvatar(uint32_t width, uint32_t height)
    : width_(width)
    , height_(height)
    , current_emotion_(EMOTION_NEUTRAL)
    , target_emotion_(EMOTION_NEUTRAL)
    , emotion_blend_(1.0f)
    , animation_time_(0.0f)
    , canvas_(nullptr)
    , framebuffer_(nullptr)
{
    // Initialize ThorVG
    if (Initializer::init(CanvasEngine::Sw, 0) != Result::Success) {
        // Error: ThorVG initialization failed
        return;
    }
    
    // Create software canvas
    canvas_ = SwCanvas::gen();
    if (!canvas_) {
        return;
    }
    
    // Allocate framebuffer (RGBA8888)
    framebuffer_ = new uint8_t[width_ * height_ * 4];
    if (!framebuffer_) {
        return;
    }
    
    // Set canvas target
    canvas_->target(framebuffer_, width_, width_, height_, SwCanvas::ARGB8888);
    
    // Center point (octopus body center)
    center_x_ = width_ / 2;
    center_y_ = height_ / 2;
}

OctopusAvatar::~OctopusAvatar()
{
    if (framebuffer_) {
        delete[] framebuffer_;
    }
    
    // Cleanup ThorVG
    Initializer::term(CanvasEngine::Sw);
}

void OctopusAvatar::set_emotion(Emotion emotion, float intensity)
{
    target_emotion_ = emotion;
    emotion_intensity_ = intensity;
    emotion_blend_ = 0.0f;  // Start blending
}

void OctopusAvatar::update(float delta_time)
{
    // Update animation time
    animation_time_ += delta_time;
    
    // Blend emotions (smooth transition)
    if (emotion_blend_ < 1.0f) {
        emotion_blend_ += delta_time * 2.0f;  // 0.5s transition
        if (emotion_blend_ >= 1.0f) {
            emotion_blend_ = 1.0f;
            current_emotion_ = target_emotion_;
        }
    }
}

void OctopusAvatar::render()
{
    // Clear canvas
    canvas_->clear();
    
    // Get emotion colors (blend if transitioning)
    EmotionColors colors = get_blended_colors();
    
    // Render octopus body
    render_body(colors);
    
    // Render tentacles (8 tentacles)
    for (int i = 0; i < 8; i++) {
        render_tentacle(i, colors);
    }
    
    // Render eyes
    render_eyes(colors);
    
    // Draw to canvas
    canvas_->draw();
    canvas_->sync();
}

const uint8_t* OctopusAvatar::get_framebuffer() const
{
    return framebuffer_;
}

void OctopusAvatar::convert_to_rgb565(uint16_t* output) const
{
    // Convert RGBA8888 → RGB565
    for (uint32_t i = 0; i < width_ * height_; i++) {
        uint8_t r = framebuffer_[i * 4 + 0];
        uint8_t g = framebuffer_[i * 4 + 1];
        uint8_t b = framebuffer_[i * 4 + 2];
        
        // Pack to RGB565
        uint16_t r5 = (r >> 3) & 0x1F;
        uint16_t g6 = (g >> 2) & 0x3F;
        uint16_t b5 = (b >> 3) & 0x1F;
        
        output[i] = (r5 << 11) | (g6 << 5) | b5;
    }
}

EmotionColors OctopusAvatar::get_blended_colors() const
{
    if (emotion_blend_ >= 1.0f) {
        // No blending needed
        return EMOTION_COLORS[current_emotion_];
    }
    
    // Blend between current and target emotion colors
    const EmotionColors& current = EMOTION_COLORS[current_emotion_];
    const EmotionColors& target = EMOTION_COLORS[target_emotion_];
    
    EmotionColors result;
    
    // Linear interpolation
    float t = emotion_blend_;
    
    result.body_primary.r = current.body_primary.r * (1.0f - t) + target.body_primary.r * t;
    result.body_primary.g = current.body_primary.g * (1.0f - t) + target.body_primary.g * t;
    result.body_primary.b = current.body_primary.b * (1.0f - t) + target.body_primary.b * t;
    
    result.body_secondary.r = current.body_secondary.r * (1.0f - t) + target.body_secondary.r * t;
    result.body_secondary.g = current.body_secondary.g * (1.0f - t) + target.body_secondary.g * t;
    result.body_secondary.b = current.body_secondary.b * (1.0f - t) + target.body_secondary.b * t;
    
    result.eyes.r = current.eyes.r * (1.0f - t) + target.eyes.r * t;
    result.eyes.g = current.eyes.g * (1.0f - t) + target.eyes.g * t;
    result.eyes.b = current.eyes.b * (1.0f - t) + target.eyes.b * t;
    
    result.pupils.r = current.pupils.r * (1.0f - t) + target.pupils.r * t;
    result.pupils.g = current.pupils.g * (1.0f - t) + target.pupils.g * t;
    result.pupils.b = current.pupils.b * (1.0f - t) + target.pupils.b * t;
    
    return result;
}

void OctopusAvatar::render_body(const EmotionColors& colors)
{
    // Body is a circle with radial gradient
    constexpr float BODY_RADIUS = 35.0f;
    
    // Create circle shape
    auto circle = Shape::gen();
    circle->appendCircle(center_x_, center_y_, BODY_RADIUS, BODY_RADIUS);
    
    // Create radial gradient (center → edge)
    auto gradient = RadialGradient::gen();
    gradient->radial(center_x_, center_y_, BODY_RADIUS);
    
    // Add color stops
    Fill::ColorStop stops[2];
    
    // Center color (primary)
    stops[0].offset = 0.0f;
    stops[0].r = colors.body_primary.r;
    stops[0].g = colors.body_primary.g;
    stops[0].b = colors.body_primary.b;
    stops[0].a = 255;
    
    // Edge color (secondary)
    stops[1].offset = 1.0f;
    stops[1].r = colors.body_secondary.r;
    stops[1].g = colors.body_secondary.g;
    stops[1].b = colors.body_secondary.b;
    stops[1].a = 255;
    
    gradient->colorStops(stops, 2);
    
    // Apply gradient to shape
    circle->fill(std::move(gradient));
    
    // Add to canvas
    canvas_->push(std::move(circle));
}

void OctopusAvatar::render_tentacle(int tentacle_index, const EmotionColors& colors)
{
    // Tentacle parameters
    constexpr float TENTACLE_LENGTH = 30.0f;
    constexpr float TENTACLE_WIDTH = 6.0f;
    constexpr float BODY_RADIUS = 35.0f;
    
    // Angle for this tentacle (8 tentacles = 360° / 8 = 45° apart)
    float base_angle = (tentacle_index * 45.0f) * (M_PI / 180.0f);
    
    // Wiggle animation (sin wave based on time)
    float wiggle = sinf(animation_time_ * 2.0f + tentacle_index * 0.5f) * 0.2f;
    float angle = base_angle + wiggle;
    
    // Start point (on body edge)
    float start_x = center_x_ + cosf(angle) * BODY_RADIUS;
    float start_y = center_y_ + sinf(angle) * BODY_RADIUS;
    
    // End point
    float end_x = center_x_ + cosf(angle) * (BODY_RADIUS + TENTACLE_LENGTH);
    float end_y = center_y_ + sinf(angle) * (BODY_RADIUS + TENTACLE_LENGTH);
    
    // Create line shape (thick stroke)
    auto line = Shape::gen();
    line->moveTo(start_x, start_y);
    line->lineTo(end_x, end_y);
    
    // Set stroke
    line->stroke(colors.body_secondary.r, 
                 colors.body_secondary.g, 
                 colors.body_secondary.b);
    line->stroke(TENTACLE_WIDTH);
    line->stroke(StrokeCap::Round);
    
    // Add to canvas
    canvas_->push(std::move(line));
}

void OctopusAvatar::render_eyes(const EmotionColors& colors)
{
    // Eye parameters
    constexpr float EYE_RADIUS = 8.0f;
    constexpr float PUPIL_RADIUS = 4.0f;
    constexpr float EYE_SPACING = 20.0f;
    constexpr float EYE_Y_OFFSET = -5.0f;  // Slightly above center
    
    // Left eye position
    float left_eye_x = center_x_ - EYE_SPACING / 2.0f;
    float left_eye_y = center_y_ + EYE_Y_OFFSET;
    
    // Right eye position
    float right_eye_x = center_x_ + EYE_SPACING / 2.0f;
    float right_eye_y = center_y_ + EYE_Y_OFFSET;
    
    // Render left eye
    render_single_eye(left_eye_x, left_eye_y, EYE_RADIUS, PUPIL_RADIUS, colors);
    
    // Render right eye
    render_single_eye(right_eye_x, right_eye_y, EYE_RADIUS, PUPIL_RADIUS, colors);
}

void OctopusAvatar::render_single_eye(
    float x, float y, 
    float eye_radius, float pupil_radius,
    const EmotionColors& colors
)
{
    // Eye white (sclera)
    auto sclera = Shape::gen();
    sclera->appendCircle(x, y, eye_radius, eye_radius);
    sclera->fill(colors.eyes.r, colors.eyes.g, colors.eyes.b);
    canvas_->push(std::move(sclera));
    
    // Pupil position (can be offset for gaze direction)
    // TODO: Integrate with eye tracking system
    float pupil_x = x + gaze_x_ * (eye_radius - pupil_radius);
    float pupil_y = y + gaze_y_ * (eye_radius - pupil_radius);
    
    // Pupil
    auto pupil = Shape::gen();
    pupil->appendCircle(pupil_x, pupil_y, pupil_radius, pupil_radius);
    pupil->fill(colors.pupils.r, colors.pupils.g, colors.pupils.b);
    canvas_->push(std::move(pupil));
}

void OctopusAvatar::set_gaze(float x, float y)
{
    // Clamp to [-1, 1] range
    gaze_x_ = std::max(-1.0f, std::min(1.0f, x));
    gaze_y_ = std::max(-1.0f, std::min(1.0f, y));
}

void OctopusAvatar::set_blink(bool blinking)
{
    is_blinking_ = blinking;
}

Emotion OctopusAvatar::get_current_emotion() const
{
    return current_emotion_;
}

float OctopusAvatar::get_emotion_intensity() const
{
    return emotion_intensity_;
}