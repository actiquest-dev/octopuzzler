/**
 * Mouth Shapes Controller Header
 */

#ifndef MOUTH_SHAPES_H
#define MOUTH_SHAPES_H

#include <cstdint>

// Forward declarations
namespace tvg {
    class SwCanvas;
}

using namespace tvg;

/**
 * RGB color
 */
struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

/**
 * Mouth shape parameters
 */
struct MouthParams {
    float width;       // Mouth width
    float height;      // Mouth height (openness)
    float curve;       // Smile/frown (-1.0 to 1.0)
    float openness;    // How open (0.0 to 1.0)
    float roundness;   // How round vs wide (0.0 to 1.0)
};

/**
 * Mouth Shapes Controller
 * 
 * Renders 9 phoneme-based mouth shapes using ThorVG.
 * 
 * Shape IDs:
 * ----------
 * 0 - SIL: Silence (closed)
 * 1 - AA:  Wide open (ah)
 * 2 - EH:  Mid-open (eh)
 * 3 - IH:  Narrow (ih)
 * 4 - OW:  Round (oh)
 * 5 - UH:  Mid-narrow (uh)
 * 6 - M:   Lips together (m, p, b)
 * 7 - F:   Friction (f, v)
 * 8 - L:   Tongue visible (l, r)
 * 
 * Usage:
 * ------
 * MouthShapesController mouth;
 * mouth.initialize(center_x, center_y);
 * 
 * // Animation loop
 * while (true) {
 *     // Set shape from phoneme
 *     mouth.set_shape(shape_id);
 *     
 *     // Update (smooth blending)
 *     mouth.update(delta_time);
 *     
 *     // Render to canvas
 *     Color color = {255, 100, 100};  // Pink
 *     mouth.render(canvas, color);
 * }
 */
class MouthShapesController {
public:
    /**
     * Constructor
     */
    MouthShapesController();
    
    /**
     * Destructor
     */
    ~MouthShapesController();
    
    /**
     * Initialize mouth controller
     * 
     * @param center_x Avatar center X
     * @param center_y Avatar center Y
     */
    void initialize(float center_x, float center_y);
    
    /**
     * Set target mouth shape
     * 
     * Smoothly transitions to new shape.
     * 
     * @param shape_id Shape ID (0-8)
     */
    void set_shape(int shape_id);
    
    /**
     * Update mouth state
     * 
     * Call every frame to update blending.
     * 
     * @param delta_time Time since last update (seconds)
     */
    void update(float delta_time);
    
    /**
     * Render mouth to canvas
     * 
     * @param canvas ThorVG canvas
     * @param color Mouth color
     */
    void render(SwCanvas* canvas, const Color& color);
    
    /**
     * Get current shape ID
     * 
     * @return Current shape (0-8)
     */
    int get_current_shape() const;

private:
    // Blend speed (shapes per second)
    static constexpr float BLEND_SPEED = 10.0f;
    
    // State
    int current_shape_;
    int target_shape_;
    float blend_progress_;  // 0.0 to 1.0
    
    // Position
    float mouth_x_;
    float mouth_y_;
    
    // Helper methods
    MouthParams get_shape_params(int shape_id) const;
    
    void render_mouth_shape(
        SwCanvas* canvas,
        const MouthParams& params,
        const Color& color
    );
    
    void render_regular_mouth(
        SwCanvas* canvas,
        const MouthParams& params,
        const Color& color
    );
    
    void render_round_mouth(
        SwCanvas* canvas,
        const MouthParams& params,
        const Color& color
    );
    
    float lerp(float a, float b, float t) const;
};

#endif // MOUTH_SHAPES_H