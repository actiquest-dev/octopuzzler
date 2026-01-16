#include "mediapipe_facemesh.h"

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include "mediapipe/util/tflite/operations/landmarks_to_transform_matrix.h"
#include "mediapipe/util/tflite/operations/transform_landmarks.h"
#include "mediapipe/util/tflite/operations/transform_tensor_bilinear.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>

MediaPipeFaceMesh::MediaPipeFaceMesh()
    : input_tensor_(nullptr)
    , output_landmarks_(nullptr)
{
}

MediaPipeFaceMesh::~MediaPipeFaceMesh() = default;

bool MediaPipeFaceMesh::initialize(const char* model_path)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model_) {
        printf("[FaceMesh] Failed to load model: %s\n", model_path);
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    auto* l2tm_v1 = mediapipe::tflite_operations::RegisterLandmarksToTransformMatrixV1();
    auto* l2tm_v2 = mediapipe::tflite_operations::RegisterLandmarksToTransformMatrixV2();
    auto* tland_v1 = mediapipe::tflite_operations::RegisterTransformLandmarksV1();
    auto* tland_v2 = mediapipe::tflite_operations::RegisterTransformLandmarksV2();
    auto* tbilin_v1 = mediapipe::tflite_operations::RegisterTransformTensorBilinearV1();
    auto* tbilin_v2 = mediapipe::tflite_operations::RegisterTransformTensorBilinearV2();
    fprintf(stderr,
            "[FaceMesh] Register custom ops: L2TM(v1=%p v2=%p) TL(v1=%p v2=%p) TB(v1=%p v2=%p)\n",
            (void*)l2tm_v1, (void*)l2tm_v2, (void*)tland_v1, (void*)tland_v2,
            (void*)tbilin_v1, (void*)tbilin_v2);
    fflush(stderr);
    if (l2tm_v2) resolver.AddCustom("Landmarks2TransformMatrix", l2tm_v2, 1, 3);
    else if (l2tm_v1) resolver.AddCustom("Landmarks2TransformMatrix", l2tm_v1, 1, 3);
    if (tland_v2) resolver.AddCustom("TransformLandmarks", tland_v2, 1, 3);
    else if (tland_v1) resolver.AddCustom("TransformLandmarks", tland_v1, 1, 3);
    if (tbilin_v2) resolver.AddCustom("TransformTensorBilinear", tbilin_v2, 1, 3);
    else if (tbilin_v1) resolver.AddCustom("TransformTensorBilinear", tbilin_v1, 1, 3);
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    if (!interpreter_) {
        printf("[FaceMesh] Failed to build interpreter\n");
        return false;
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        printf("[FaceMesh] AllocateTensors failed\n");
        return false;
    }

    input_tensor_ = interpreter_->input_tensor(0);
    output_landmarks_ = interpreter_->output_tensor(0);
    printf("[FaceMesh] Initialized\n");
    return true;
}

uint8_t* MediaPipeFaceMesh::resize_image(const uint8_t* src, int src_w, int src_h, int dst_w, int dst_h)
{
    uint8_t* out = (uint8_t*)malloc(dst_w * dst_h * 3);
    if (!out) return nullptr;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int sx = x * src_w / dst_w;
            int sy = y * src_h / dst_h;
            const uint8_t* p = src + (sy * src_w + sx) * 3;
            uint8_t* d = out + (y * dst_w + x) * 3;
            d[0] = p[0];
            d[1] = p[1];
            d[2] = p[2];
        }
    }

    return out;
}

bool MediaPipeFaceMesh::process(const uint8_t* face_roi, int width, int height, FaceLandmarks& out)
{
    if (!interpreter_ || !input_tensor_ || !output_landmarks_) return false;

    const int dst_w = 192;
    const int dst_h = 192;
    uint8_t* resized = resize_image(face_roi, width, height, dst_w, dst_h);
    if (!resized) return false;

    if (input_tensor_->type != kTfLiteFloat32) {
        free(resized);
        printf("[FaceMesh] Unexpected input tensor type\n");
        return false;
    }

    float* input_data = input_tensor_->data.f;
    const int total = dst_w * dst_h * 3;
    for (int i = 0; i < total; i++) {
        input_data[i] = resized[i] / 255.0f;
    }

    TfLiteStatus status = interpreter_->Invoke();
    free(resized);
    if (status != kTfLiteOk) {
        printf("[FaceMesh] Invoke failed\n");
        return false;
    }

    if (output_landmarks_->type != kTfLiteFloat32) {
        printf("[FaceMesh] Unexpected output tensor type\n");
        return false;
    }

    const float* data = output_landmarks_->data.f;
    const int count = std::min(478, (int)(output_landmarks_->bytes / (sizeof(float) * 3)));
    out.num_landmarks = count;

    for (int i = 0; i < count; ++i) {
        out.points[i * 3 + 0] = data[i * 3 + 0] * width;
        out.points[i * 3 + 1] = data[i * 3 + 1] * height;
        out.points[i * 3 + 2] = data[i * 3 + 2];
    }

    for (int i = count; i < 478; ++i) {
        out.points[i * 3 + 0] = 0.0f;
        out.points[i * 3 + 1] = 0.0f;
        out.points[i * 3 + 2] = 0.0f;
    }

    return true;
}
