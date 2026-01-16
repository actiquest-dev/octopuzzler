/**
 * BlazeFace Detector (STUB)
 * 
 * Lightweight face detection using Google BlazeFace model.
 * Runs on-device with TensorFlow Lite Micro.
 * 
 * Model: BlazeFace (SSD-MobileNet-based)
 * Input: 128×128 RGB
 * Output: Face bounding boxes + keypoints
 * 
 * Performance: ~20ms on BK7258 CPU1
 * 
 * TODO: Implement full detector
 * - Load TFLite model
 * - Image preprocessing
 * - Non-max suppression
 * - Keypoint extraction
 * 
 * Author: Octopus AI Team
 * Date: December 28, 2025
 * Version: 1.0 (STUB)
 */

#include "blazeface_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

// Model input size
constexpr int MODEL_INPUT_WIDTH = 128;
constexpr int MODEL_INPUT_HEIGHT = 128;
constexpr int MODEL_INPUT_CHANNELS = 3;

// Detection thresholds
constexpr float DETECTION_THRESHOLD = 0.4f;
constexpr float IOU_THRESHOLD = 0.3f;

BlazeFaceDetector::BlazeFaceDetector()
    : input_tensor_(nullptr)
    , output_boxes_tensor_(nullptr)
    , output_scores_tensor_(nullptr)
    , input_w_(MODEL_INPUT_WIDTH)
    , input_h_(MODEL_INPUT_HEIGHT)
    , is_quantized_(false)
{
}

BlazeFaceDetector::~BlazeFaceDetector()
{
}

bool BlazeFaceDetector::initialize(const char* model_path)
{
    fprintf(stderr, "[BlazeFace] Initializing detector...\n");
    fprintf(stderr, "[BlazeFace] Model path: %s\n", model_path);

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model_) {
        fprintf(stderr, "[BlazeFace] ERROR: Failed to load model\n");
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    if (!interpreter_) {
        fprintf(stderr, "[BlazeFace] ERROR: Failed to build interpreter\n");
        return false;
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "[BlazeFace] ERROR: AllocateTensors failed\n");
        return false;
    }

    input_tensor_ = interpreter_->input_tensor(0);
    if (!input_tensor_) {
        fprintf(stderr, "[BlazeFace] ERROR: input tensor missing\n");
        return false;
    }
    output_boxes_tensor_ = interpreter_->output_tensor(0);
    output_scores_tensor_ = interpreter_->output_tensor(1);
    if (!output_boxes_tensor_ || !output_scores_tensor_) {
        fprintf(stderr, "[BlazeFace] ERROR: output tensors missing\n");
        return false;
    }

    if (input_tensor_->type == kTfLiteUInt8) {
        is_quantized_ = true;
    } else if (input_tensor_->type == kTfLiteFloat32) {
        is_quantized_ = false;
    } else {
        fprintf(stderr, "[BlazeFace] ERROR: unsupported input tensor type\n");
        return false;
    }

    if (input_tensor_->dims && input_tensor_->dims->size >= 4) {
        input_h_ = input_tensor_->dims->data[1];
        input_w_ = input_tensor_->dims->data[2];
    }

    int num_anchors = 0;
    if (output_scores_tensor_->dims && output_scores_tensor_->dims->size >= 2) {
        num_anchors = output_scores_tensor_->dims->data[output_scores_tensor_->dims->size - 2];
    }
    if (num_anchors <= 0 && output_boxes_tensor_->dims && output_boxes_tensor_->dims->size >= 2) {
        num_anchors = output_boxes_tensor_->dims->data[output_boxes_tensor_->dims->size - 2];
    }
    if (num_anchors <= 0) {
        fprintf(stderr, "[BlazeFace] ERROR: unable to determine anchor count\n");
        return false;
    }
    build_anchors(num_anchors);

    fprintf(stderr, "[BlazeFace] Initialized (anchors=%d, input=%dx%d, quantized=%s)\n",
            num_anchors, input_w_, input_h_, is_quantized_ ? "yes" : "no");
    fflush(stderr);
    return true;
}

bool BlazeFaceDetector::detect(
    const uint8_t* image_data,
    int width,
    int height,
    FaceDetection& detection
)
{
    static uint32_t frame_counter = 0;
    frame_counter++;
    fprintf(stderr, "[BlazeFace] frame=%u\n", frame_counter);
    if (!interpreter_ || !input_tensor_ || !output_boxes_tensor_ || !output_scores_tensor_) {
        return false;
    }

    preprocess_image(image_data, width, height);
    if (interpreter_->Invoke() != kTfLiteOk) {
        fprintf(stderr, "[BlazeFace] ERROR: Inference failed\n");
        return false;
    }

    const bool ok = decode_detections(width, height, detection);
    if (ok) {
        fprintf(stderr, "[BlazeFace] det x=%d y=%d w=%d h=%d conf=%.2f\n",
                detection.bbox_x, detection.bbox_y, detection.bbox_width,
                detection.bbox_height, detection.confidence);
    } else {
        fprintf(stderr, "[BlazeFace] det none\n");
    }
    fflush(stderr);
    return ok;
}

void BlazeFaceDetector::preprocess_image(
    const uint8_t* image_data,
    int width,
    int height
)
{
    const int dst_w = input_w_;
    const int dst_h = input_h_;
    const int dst_size = dst_w * dst_h * 3;

    if (!is_quantized_) {
        if ((int)input_buffer_.size() != dst_size) {
            input_buffer_.assign(dst_size, 0.0f);
        }
    }

    const float scale = 1.0f / 255.0f;
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int sx = x * width / dst_w;
            const int sy = y * height / dst_h;
            const uint8_t* p = image_data + (sy * width + sx) * 3;
            const int di = (y * dst_w + x) * 3;
            if (is_quantized_) {
                uint8_t* dst = input_tensor_->data.uint8 + di;
                dst[0] = p[0];
                dst[1] = p[1];
                dst[2] = p[2];
            } else {
                input_buffer_[di + 0] = p[0] * scale;
                input_buffer_[di + 1] = p[1] * scale;
                input_buffer_[di + 2] = p[2] * scale;
            }
        }
    }

    if (!is_quantized_) {
        float* dst = input_tensor_->data.f;
        std::memcpy(dst, input_buffer_.data(), sizeof(float) * dst_size);
    }
}

std::vector<FaceDetection> BlazeFaceDetector::nms(
    const std::vector<FaceDetection>& detections,
    float iou_threshold
)
{
    if (detections.empty()) {
        return detections;
    }

    std::vector<FaceDetection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const FaceDetection& a, const FaceDetection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<FaceDetection> kept;
    std::vector<bool> suppressed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            if (calculate_iou(sorted[i], sorted[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

float BlazeFaceDetector::calculate_iou(
    const FaceDetection& a,
    const FaceDetection& b
)
{
    // TODO: Calculate Intersection over Union
    // IoU = (area of intersection) / (area of union)
    
    int x1 = std::max(a.bbox_x, b.bbox_x);
    int y1 = std::max(a.bbox_y, b.bbox_y);
    int x2 = std::min(a.bbox_x + a.bbox_width, b.bbox_x + b.bbox_width);
    int y2 = std::min(a.bbox_y + a.bbox_height, b.bbox_y + b.bbox_height);
    
    if (x2 < x1 || y2 < y1) {
        return 0.0f;  // No intersection
    }
    
    int intersection = (x2 - x1) * (y2 - y1);
    int area_a = a.bbox_width * a.bbox_height;
    int area_b = b.bbox_width * b.bbox_height;
    int union_area = area_a + area_b - intersection;
    
    return static_cast<float>(intersection) / union_area;
}

void BlazeFaceDetector::build_anchors(int num_anchors)
{
    anchors_.clear();
    if (num_anchors <= 0) return;

    const int fm1 = input_w_ / 8;
    const int fm2 = input_w_ / 16;
    const int cells1 = fm1 * fm1;
    const int cells2 = fm2 * fm2;

    int k1 = 2;
    int k2 = 0;
    if (cells1 > 0 && cells2 > 0) {
        int remaining = num_anchors - cells1 * k1;
        if (remaining > 0 && remaining % cells2 == 0) {
            k2 = remaining / cells2;
        } else {
            k1 = 4;
            remaining = num_anchors - cells1 * k1;
            if (remaining > 0 && remaining % cells2 == 0) {
                k2 = remaining / cells2;
            }
        }
    }
    if (k2 <= 0) {
        k1 = 1;
        k2 = 1;
    }

    const float min_scale = 0.1484375f;
    const float max_scale = 0.75f;
    const int num_layers = 2;

    for (int layer = 0; layer < num_layers; ++layer) {
        const int fm = (layer == 0) ? fm1 : fm2;
        const int stride = (layer == 0) ? 8 : 16;
        const int k = (layer == 0) ? k1 : k2;
        if (fm <= 0 || k <= 0) continue;

        const float scale = min_scale + (max_scale - min_scale) * layer / (num_layers - 1);
        for (int y = 0; y < fm; ++y) {
            for (int x = 0; x < fm; ++x) {
                const float cx = (x + 0.5f) / fm;
                const float cy = (y + 0.5f) / fm;
                for (int a = 0; a < k; ++a) {
                    float s = scale * (1.0f + 0.15f * (a - (k - 1) * 0.5f));
                    s = std::max(0.05f, std::min(0.95f, s));
                    Anchor anchor;
                    anchor.x_center = cx;
                    anchor.y_center = cy;
                    anchor.w = s;
                    anchor.h = s;
                    anchors_.push_back(anchor);
                    if ((int)anchors_.size() >= num_anchors) {
                        return;
                    }
                }
            }
        }
    }
}

bool BlazeFaceDetector::decode_detections(int width, int height, FaceDetection& detection)
{
    static uint32_t decode_counter = 0;
    decode_counter++;
    const int num_anchors = (int)anchors_.size();
    if (num_anchors <= 0) return false;

    const float* boxes = output_boxes_tensor_->data.f;
    std::vector<float> box_buf;
    if (output_boxes_tensor_->type == kTfLiteUInt8) {
        const float scale = output_boxes_tensor_->params.scale;
        const int zero = output_boxes_tensor_->params.zero_point;
        const uint8_t* src = output_boxes_tensor_->data.uint8;
        const int total = output_boxes_tensor_->bytes;
        box_buf.resize(total);
        for (int i = 0; i < total; ++i) {
            box_buf[i] = (static_cast<int>(src[i]) - zero) * scale;
        }
        boxes = box_buf.data();
    }

    const float* scores = nullptr;

    if (output_scores_tensor_->type == kTfLiteFloat32) {
        scores = output_scores_tensor_->data.f;
    } else if (output_scores_tensor_->type == kTfLiteUInt8) {
        // Dequantize scores on the fly.
        static std::vector<float> score_buf;
        score_buf.resize(num_anchors);
        const float scale = output_scores_tensor_->params.scale;
        const int zero = output_scores_tensor_->params.zero_point;
        const uint8_t* src = output_scores_tensor_->data.uint8;
        for (int i = 0; i < num_anchors; ++i) {
            score_buf[i] = (static_cast<int>(src[i]) - zero) * scale;
        }
        scores = score_buf.data();
    }

    if (!boxes || !scores) return false;

    const int box_stride = (output_boxes_tensor_->dims && output_boxes_tensor_->dims->size >= 2)
        ? output_boxes_tensor_->dims->data[output_boxes_tensor_->dims->size - 1]
        : 16;
    const int score_stride = (output_scores_tensor_->dims && output_scores_tensor_->dims->size >= 2)
        ? output_scores_tensor_->dims->data[output_scores_tensor_->dims->size - 1]
        : 1;

    constexpr float X_SCALE = 128.0f;
    constexpr float Y_SCALE = 128.0f;
    constexpr float W_SCALE = 128.0f;
    constexpr float H_SCALE = 128.0f;

    int best_idx = -1;
    float best_score = -1.0f;
    float best_raw0 = 0.0f;
    float best_raw1 = 0.0f;
    for (int i = 0; i < num_anchors; ++i) {
        const float raw0 = scores[i * score_stride];
        const float raw1 = (score_stride >= 2) ? scores[i * score_stride + 1] : raw0;
        const float raw = (score_stride >= 2) ? raw1 : raw0;
        const float score = (raw > 1.0f || raw < 0.0f) ? (1.0f / (1.0f + std::exp(-raw))) : raw;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
            best_raw0 = raw0;
            best_raw1 = raw1;
        }
    }

    if (best_idx < 0 || best_score < DETECTION_THRESHOLD) {
        return false;
    }

    const Anchor& a = anchors_[best_idx];
    const float* b = boxes + best_idx * box_stride;
    const float y_center = b[0] / Y_SCALE * a.h + a.y_center;
    const float x_center = b[1] / X_SCALE * a.w + a.x_center;
    const float h = b[2] / H_SCALE * a.h;
    const float w = b[3] / W_SCALE * a.w;

    float xmin = x_center - w * 0.5f;
    float ymin = y_center - h * 0.5f;
    float xmax = x_center + w * 0.5f;
    float ymax = y_center + h * 0.5f;

    if (!(w > 0.0f && h > 0.0f)) {
        const float size = 0.4f;
        xmin = a.x_center - size * 0.5f;
        ymin = a.y_center - size * 0.5f;
        xmax = a.x_center + size * 0.5f;
        ymax = a.y_center + size * 0.5f;
    }

    detection.bbox_x = std::max(0, (int)(xmin * width));
    detection.bbox_y = std::max(0, (int)(ymin * height));
    detection.bbox_width = std::min(width, (int)(xmax * width)) - detection.bbox_x;
    detection.bbox_height = std::min(height, (int)(ymax * height)) - detection.bbox_y;
    detection.confidence = best_score;
    fprintf(stderr,
            "[BlazeFace] score_stride=%d box_stride=%d raw0=%.4f raw1=%.4f best=%.4f\n",
            score_stride, box_stride, best_raw0, best_raw1, best_score);
    fprintf(stderr,
            "[BlazeFace] box0=%.4f box1=%.4f box2=%.4f box3=%.4f\n",
            b[0], b[1], b[2], b[3]);
    fflush(stderr);
    return detection.bbox_width > 0 && detection.bbox_height > 0;
}
