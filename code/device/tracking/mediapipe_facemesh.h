/**
 * MediaPipe Face Mesh Wrapper (TFLite)
 */

#ifndef MEDIAPIPE_FACEMESH_H
#define MEDIAPIPE_FACEMESH_H

#include <cstdint>
#include <memory>

#include "gaze_calculator.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

class MediaPipeFaceMesh {
public:
    MediaPipeFaceMesh();
    ~MediaPipeFaceMesh();

    bool initialize(const char* model_path);
    bool process(const uint8_t* face_roi, int width, int height, FaceLandmarks& out);

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_landmarks_;

    uint8_t* resize_image(const uint8_t* src, int src_w, int src_h, int dst_w, int dst_h);
};

#endif // MEDIAPIPE_FACEMESH_H
