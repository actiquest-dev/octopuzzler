/**
 * Gaze Calculator Header
 */

#ifndef GAZE_CALCULATOR_H
#define GAZE_CALCULATOR_H

/**
 * Face landmarks structure
 */
struct FaceLandmarks {
    float points[478 * 3];  // 478 landmarks × (x, y, z)
    int num_landmarks;
};

/**
 * Gaze calculation result
 */
struct GazeResult {
    float gaze_x;              // Horizontal gaze (-1.0 to 1.0)
    float gaze_y;              // Vertical gaze (-1.0 to 1.0)
    bool is_blinking;          // True if eyes closed
    float eye_aspect_ratio;    // EAR value
};

/**
 * Gaze Calculator
 * 
 * Calculates gaze direction from MediaPipe face landmarks.
 * 
 * Algorithm:
 * - Locate eye corners (landmarks 33, 133 for left eye)
 * - Locate iris center (landmarks 468-472 for left iris)
 * - Calculate: gaze = (iris_pos - eye_center) / (eye_width / 2)
 * - Detect blinks using Eye Aspect Ratio (EAR < 0.2)
 */
class GazeCalculator {
public:
    GazeCalculator();
    ~GazeCalculator();
    
    /**
     * Calculate gaze from face landmarks
     * 
     * @param landmarks MediaPipe face landmarks (478 points)
     * @return Gaze result
     */
    GazeResult calculate_gaze(const FaceLandmarks& landmarks);

private:
    void calculate_eye_gaze(
        const FaceLandmarks& landmarks,
        const int* eye_indices,
        int eye_count,
        const int* iris_indices,
        int iris_count,
        float& gaze_x,
        float& gaze_y
    );
    
    float calculate_ear(
        const FaceLandmarks& landmarks,
        const int* eye_indices,
        int eye_count
    );
};

#endif // GAZE_CALCULATOR_H