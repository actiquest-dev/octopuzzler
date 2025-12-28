/**
 * Mouth Shapes Controller
 * 
 * Renders 9 different mouth shapes for phoneme-based lip sync.
 * Uses ThorVG for vector graphics rendering.
 * 
 * Mouth Shapes:
 * 0 - SIL (silence, closed)
 * 1 - AA (wide open)
 * 2 - EH (mid-open)
 * 3 - IH (narrow)
 * 4 - OW (round)
 * 5 - UH (mid-narrow)
 * 6 - M (lips together)
 * 7 - F (friction, teeth visible)
 * 8 - L (tongue visible)
 * 
 * Author: Octopus AI Team
 * Date: December 28, 2025
 * Version: 1.0
 */

#include "mouth_shapes.h"
#include <thorvg.h>
#include <cmath>
#include <cstdio>

using namespace tvg;

// Mouth position (relative to avatar center)
constexpr float MOUTH_Y_OFFSET = 15.0f;  // Below center

// Mouth dimensions
constexpr float MOUTH_WIDTH = 20.0f;
constexpr float MOUTH_MAX_HEIGHT = 12.0f;

MouthShapesController::MouthShapesController()
    : current_shape_(0)
    , target_shape_(0)
    , blend_progress_(1.0f)
    , mouth_x_(0.0f)
    , mouth_y_(0.0f)
{
}

MouthShapesController::~MouthShapesController()
{
}

void MouthShapesController::initialize(float center_x, float center_y)
{
    mouth_x_ = center_x;
    mouth_y_ = center_y + MOUTH_Y_OFFSET;
    
    printf("[MouthShapes] Initialized at (%.1f, %.1f)\n", mouth_x_, mouth_y_);
}

void MouthShapesController::set_shape(int shape_id)
{
    if (shape_id < 0 || shape_id > 8) {
        printf("[MouthShapes] WARNING: Invalid shape_id %d\n", shape_id);
        return;
    }
    
    if (shape_id != target_shape_) {
        target_shape_ = shape_id;
        blend_progress_ = 0.0f;  // Start blending
    }
}

void MouthShapesController::update(float delta_time)
{
    // Blend to target shape
    if (blend_progress_ < 1.0f) {
        blend_progress_ += delta_time * BLEND_SPEED;
        
        if (blend_progress_ >= 1.0f) {
            blend_progress_ = 1.0f;
            current_shape_ = target_shape_;
        }
    }
}

void MouthShapesController::render(SwCanvas* canvas, const Color& color)
{
    // Get current and target shape parameters
    MouthParams current_params = get_shape_params(current_shape_);
    MouthParams target_params = get_shape_params(target_shape_);
    
    // Blend parameters
    MouthParams blended;
    blended.width = lerp(current_params.width, target_params.width, blend_progress_);
    blended.height = lerp(current_params.height, target_params.height, blend_progress_);
    blended.curve = lerp(current_params.curve, target_params.curve, blend_progress_);
    blended.openness = lerp(current_params.openness, target_params.openness, blend_progress_);
    blended.roundness = lerp(current_params.roundness, target_params.roundness, blend_progress_);
    
    // Render based on blended parameters
    render_mouth_shape(canvas, blended, color);
}

int MouthShapesController::get_current_shape() const
{
    return current_shape_;
}

MouthParams MouthShapesController::get_shape_params(int shape_id) const
{
    MouthParams params;
    
    switch (shape_id) {
        case 0:  // SIL - Silence (closed)
            params.width = MOUTH_WIDTH * 0.7f;
            params.height = 2.0f;
            params.curve = 0.0f;
            params.openness = 0.0f;
            params.roundness = 0.0f;
            break;
        
        case 1:  // AA - Wide open
            params.width = MOUTH_WIDTH * 1.0f;
            params.height = MOUTH_MAX_HEIGHT * 0.9f;
            params.curve = -0.2f;  // Slight frown
            params.openness = 1.0f;
            params.roundness = 0.2f;
            break;
        
        case 2:  // EH - Mid-open
            params.width = MOUTH_WIDTH * 0.9f;
            params.height = MOUTH_MAX_HEIGHT * 0.6f;
            params.curve = 0.0f;
            params.openness = 0.6f;
            params.roundness = 0.3f;
            break;
        
        case 3:  // IH - Narrow
            params.width = MOUTH_WIDTH * 0.6f;
            params.height = MOUTH_MAX_HEIGHT * 0.3f;
            params.curve = 0.1f;  // Slight smile
            params.openness = 0.3f;
            params.roundness = 0.1f;
            break;
        
        case 4:  // OW - Round
            params.width = MOUTH_WIDTH * 0.5f;
            params.height = MOUTH_MAX_HEIGHT * 0.7f;
            params.curve = 0.0f;
            params.openness = 0.7f;
            params.roundness = 1.0f;  // Maximum roundness
            break;
        
        case 5:  // UH - Mid-narrow
            params.width = MOUTH_WIDTH * 0.7f;
            params.height = MOUTH_MAX_HEIGHT * 0.4f;
            params.curve = 0.0f;
            params.openness = 0.4f;
            params.roundness = 0.5f;
            break;
        
        case 6:  // M - Lips together
            params.width = MOUTH_WIDTH * 0.8f;
            params.height = 3.0f;
            params.curve = 0.0f;
            params.openness = 0.0f;
            params.roundness = 0.0f;
            break;
        
        case 7:  // F - Friction (teeth visible)
            params.width = MOUTH_WIDTH * 0.8f;
            params.height = MOUTH_MAX_HEIGHT * 0.25f;
            params.curve = 0.0f;
            params.openness = 0.25f;
            params.roundness = 0.0f;
            break;
        
        case 8:  // L - Tongue visible
            params.width = MOUTH_WIDTH * 0.7f;
            params.height = MOUTH_MAX_HEIGHT * 0.35f;
            params.curve = 0.0f;
            params.openness = 0.35f;
            params.roundness = 0.2f;
            break;
        
        default:
            // Default to silence
            params = get_shape_params(0);
            break;
    }
    
    return params;
}

void MouthShapesController::render_mouth_shape(
    SwCanvas* canvas,
    const MouthParams& params,
    const Color& color
)
{
    if (params.roundness > 0.5f) {
        // Round mouth (for OW, UH)
        render_round_mouth(canvas, params, color);
    } else {
        // Regular mouth (for most phonemes)
        render_regular_mouth(canvas, params, color);
    }
}

void MouthShapesController::render_regular_mouth(
    SwCanvas* canvas,
    const MouthParams& params,
    const Color& color
)
{
    // Create path for mouth
    auto mouth = Shape::gen();
    
    // Calculate control points for cubic bezier
    float half_width = params.width / 2.0f;
    float half_height = params.height / 2.0f;
    
    // Curve amount (0 = straight, positive = smile, negative = frown)
    float curve_offset = params.curve * half_height;
    
    // Upper lip
    float left_x = mouth_x_ - half_width;
    float right_x = mouth_x_ + half_width;
    float top_y = mouth_y_ - half_height;
    float bottom_y = mouth_y_ + half_height;
    
    // Start at left corner
    mouth->moveTo(left_x, mouth_y_);
    
    // Upper curve
    mouth->cubicTo(
        left_x + half_width * 0.5f, top_y + curve_offset,      // Control point 1
        right_x - half_width * 0.5f, top_y + curve_offset,     // Control point 2
        right_x, mouth_y_                                      // End point
    );
    
    if (params.openness > 0.1f) {
        // Lower curve (if mouth is open)
        mouth->cubicTo(
            right_x - half_width * 0.5f, bottom_y - curve_offset,  // Control point 1
            left_x + half_width * 0.5f, bottom_y - curve_offset,   // Control point 2
            left_x, mouth_y_                                       // End point
        );
        
        mouth->close();
        
        // Fill mouth interior
        mouth->fill(color.r, color.g, color.b);
    } else {
        // Just a line (closed mouth)
        mouth->stroke(color.r, color.g, color.b);
        mouth->stroke(2.0f);
        mouth->stroke(StrokeCap::Round);
    }
    
    canvas->push(std::move(mouth));
}

void MouthShapesController::render_round_mouth(
    SwCanvas* canvas,
    const MouthParams& params,
    const Color& color
)
{
    // Round mouth (ellipse)
    auto mouth = Shape::gen();
    
    float radius_x = params.width / 2.0f;
    float radius_y = params.height / 2.0f;
    
    mouth->appendCircle(mouth_x_, mouth_y_, radius_x, radius_y);
    mouth->fill(color.r, color.g, color.b);
    
    canvas->push(std::move(mouth));
}

float MouthShapesController::lerp(float a, float b, float t) const
{
    return a * (1.0f - t) + b * t;
}