/**
 * BlazeFace Detector Header (STUB)
 */

#ifndef BLAZEFACE_DETECTOR_H
#define BLAZEFACE_DETECTOR_H

#include <cstdint>
#include <memory>
#include <vector>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

/**
 * Face detection result
 */
struct FaceDetection {
    int bbox_x;
    int bbox_y;
    int bbox_width;
    int bbox_height;
    float confidence;
};

/**
 * BlazeFace Detector
 * 
 * Lightweight face detection using TensorFlow Lite Micro.
 * 
 * TODO: Implement
 * - Model loading
 * - Image preprocessing
 * - Inference
 * - NMS post-processing
 * 
 * Usage (when implemented):
 * -------------------------
 * BlazeFaceDetector detector;
 * detector.initialize("/models/blazeface.tflite");
 * 
 * uint8_t* frame = camera.capture();
 * FaceDetection face;
 * 
 * if (detector.detect(frame, 640, 480, face)) {
 *     printf("Face detected at (%d,%d)\n", face.bbox_x, face.bbox_y);
 * }
 */
class BlazeFaceDetector {
public:
    BlazeFaceDetector();
    ~BlazeFaceDetector();
    
    /**
     * Initialize detector
     * 
     * @param model_path Path to blazeface.tflite
     * @return true if successful
     */
    bool initialize(const char* model_path);
    
    /**
     * Detect face in image
     * 
     * @param image_data RGB888 image data
     * @param width Image width
     * @param height Image height
     * @param detection Output detection
     * @return true if face detected
     */
    bool detect(
        const uint8_t* image_data,
        int width,
        int height,
        FaceDetection& detection
    );

private:
    struct Anchor {
        float x_center;
        float y_center;
        float w;
        float h;
    };

    // Helper methods
    void preprocess_image(const uint8_t* image_data, int width, int height);
    std::vector<FaceDetection> nms(const std::vector<FaceDetection>& detections, float iou_threshold);
    float calculate_iou(const FaceDetection& a, const FaceDetection& b);
    void build_anchors(int num_anchors);
    bool decode_detections(int width, int height, FaceDetection& detection);

    // TFLite state
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_boxes_tensor_;
    TfLiteTensor* output_scores_tensor_;

    std::vector<float> input_buffer_;
    std::vector<Anchor> anchors_;
    int input_w_;
    int input_h_;
    bool is_quantized_;
};

#endif // BLAZEFACE_DETECTOR_H
