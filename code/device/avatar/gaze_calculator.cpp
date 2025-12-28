/**
 * Gaze Calculator
 * 
 * Calculates gaze direction from MediaPipe face landmarks.
 * Uses iris positions relative to eye corners.
 */

#include "gaze_calculator.h"
#include <cmath>

// MediaPipe landmark indices (468-point model)
constexpr int LEFT_EYE_INDICES[] = {33, 133, 160, 159, 158, 157, 173, 144};
constexpr int RIGHT_EYE_INDICES[] = {362, 263, 387, 386, 385, 384, 398, 373};

constexpr int LEFT_IRIS_INDICES[] = {468, 469, 470, 471, 472};
constexpr int RIGHT_IRIS_INDICES[] = {473, 474, 475, 476, 477};

// EAR threshold for blink detection
constexpr float BLINK_EAR_THRESHOLD = 0.2f;

GazeCalculator::GazeCalculator()
{
}

GazeCalculator::~GazeCalculator()
{
}

GazeResult GazeCalculator::calculate_gaze(const FaceLandmarks& landmarks)
{
    GazeResult result;
    
    // Calculate left eye gaze
    float left_gaze_x = 0.0f;
    float left_gaze_y = 0.0f;
    calculate_eye_gaze(
        landmarks,
        LEFT_EYE_INDICES, 8,
        LEFT_IRIS_INDICES, 5,
        left_gaze_x, left_gaze_y
    );
    
    // Calculate right eye gaze
    float right_gaze_x = 0.0f;
    float right_gaze_y = 0.0f;
    calculate_eye_gaze(
        landmarks,
        RIGHT_EYE_INDICES, 8,
        RIGHT_IRIS_INDICES, 5,
        right_gaze_x, right_gaze_y
    );
    
    // Average both eyes
    result.gaze_x = (left_gaze_x + right_gaze_x) / 2.0f;
    result.gaze_y = (left_gaze_y + right_gaze_y) / 2.0f;
    
    // Clamp to [-1, 1]
    result.gaze_x = std::max(-1.0f, std::min(1.0f, result.gaze_x));
    result.gaze_y = std::max(-1.0f, std::min(1.0f, result.gaze_y));
    
    // Calculate EAR (Eye Aspect Ratio) for blink detection
    float left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES, 8);
    float right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES, 8);
    float avg_ear = (left_ear + right_ear) / 2.0f;
    
    result.is_blinking = (avg_ear < BLINK_EAR_THRESHOLD);
    result.eye_aspect_ratio = avg_ear;
    
    return result;
}

void GazeCalculator::calculate_eye_gaze(
    const FaceLandmarks& landmarks,
    const int* eye_indices,
    int eye_count,
    const int* iris_indices,
    int iris_count,
    float& gaze_x,
    float& gaze_y
)
{
    // Get eye corners (leftmost and rightmost points)
    int left_corner_idx = eye_indices[0];
    int right_corner_idx = eye_indices[4];  // Approximate
    
    float eye_left_x = landmarks.points[left_corner_idx * 3 + 0];
    float eye_left_y = landmarks.points[left_corner_idx * 3 + 1];
    float eye_right_x = landmarks.points[right_corner_idx * 3 + 0];
    float eye_right_y = landmarks.points[right_corner_idx * 3 + 1];
    
    // Eye center
    float eye_center_x = (eye_left_x + eye_right_x) / 2.0f;
    float eye_center_y = (eye_left_y + eye_right_y) / 2.0f;
    
    // Eye width
    float eye_width = std::abs(eye_right_x - eye_left_x);
    
    // Iris center (average of iris landmarks)
    float iris_x = 0.0f;
    float iris_y = 0.0f;
    
    for (int i = 0; i < iris_count; i++) {
        int idx = iris_indices[i];
        iris_x += landmarks.points[idx * 3 + 0];
        iris_y += landmarks.points[idx * 3 + 1];
    }
    
    iris_x /= iris_count;
    iris_y /= iris_count;
    
    // Gaze = (iris_pos - eye_center) / (eye_width / 2)
    gaze_x = (iris_x - eye_center_x) / (eye_width / 2.0f);
    gaze_y = (iris_y - eye_center_y) / (eye_width / 2.0f);
}

float GazeCalculator::calculate_ear(
    const FaceLandmarks& landmarks,
    const int* eye_indices,
    int eye_count
)
{
    // Eye Aspect Ratio (EAR)
    // EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    
    // Get key points
    int p1_idx = eye_indices[1];  // Top
    int p2_idx = eye_indices[5];  // Bottom
    int p3_idx = eye_indices[0];  // Left corner
    int p4_idx = eye_indices[4];  // Right corner
    
    float p1_x = landmarks.points[p1_idx * 3 + 0];
    float p1_y = landmarks.points[p1_idx * 3 + 1];
    
    float p2_x = landmarks.points[p2_idx * 3 + 0];
    float p2_y = landmarks.points[p2_idx * 3 + 1];
    
    float p3_x = landmarks.points[p3_idx * 3 + 0];
    float p3_y = landmarks.points[p3_idx * 3 + 1];
    
    float p4_x = landmarks.points[p4_idx * 3 + 0];
    float p4_y = landmarks.points[p4_idx * 3 + 1];
    
    // Calculate distances
    float vertical = std::sqrt(
        (p1_x - p2_x) * (p1_x - p2_x) +
        (p1_y - p2_y) * (p1_y - p2_y)
    );
    
    float horizontal = std::sqrt(
        (p3_x - p4_x) * (p3_x - p4_x) +
        (p3_y - p4_y) * (p3_y - p4_y)
    );
    
    // EAR
    float ear = vertical / horizontal;
    
    return ear;
}