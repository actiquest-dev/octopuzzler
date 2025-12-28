class MediaPipeFaceMesh {
public:
    MediaPipeFaceMesh() {
        // Load TFLite model (lite variant)
        model = tflite::FlatBufferModel::BuildFromFile(
            "/flash/models/facemesh_lite.tflite"
        );
        
        if (!model) {
            printf("Failed to load MediaPipe model\n");
            return;
        }
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        
        if (!interpreter) {
            printf("Failed to build MediaPipe interpreter\n");
            return;
        }
        
        interpreter->AllocateTensors();
        
        input_tensor = interpreter->input_tensor(0);
        output_landmarks = interpreter->output_tensor(0);
        
        printf("MediaPipe Face Mesh initialized\n");
    }
    
    bool detect(const uint8_t* face_roi, int width, int height, 
                FaceLandmarks* out) {
        // 1. Resize to 192x192 (MediaPipe input)
        uint8_t* resized = resize_image(face_roi, width, height, 192, 192);
        
        // 2. Normalize to [0, 1]
        float* input_data = input_tensor->data.f;
        for (int i = 0; i < 192 * 192 * 3; i++) {
            input_data[i] = resized[i] / 255.0f;
        }
        
        // 3. Run inference
        TfLiteStatus status = interpreter->Invoke();
        if (status != kTfLiteOk) {
            free(resized);
            return false;
        }
        
        // 4. Parse landmarks
        // Output: 478 landmarks x 3 coordinates (x, y, z)
        float* landmarks = output_landmarks->data.f;
        
        for (int i = 0; i < 478; i++) {
            // Convert from normalized [0, 1] to pixel coordinates
            out->points[i].x = landmarks[i * 3 + 0] * width;
            out->points[i].y = landmarks[i * 3 + 1] * height;
            out->points[i].z = landmarks[i * 3 + 2];  // Depth (relative)
        }
        
        out->confidence = 0.9f;  // Simplified (model doesn't output confidence)
        
        free(resized);
        return true;
    }
    
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_landmarks;
    
    uint8_t* resize_image(const uint8_t* src, int src_w, int src_h,
                          int dst_w, int dst_h);  // Same as BlazeFace
};

struct Point {
    float x, y, z;
};

struct FaceLandmarks {
    Point points[478];  // 468 face + 10 iris landmarks
    float confidence;
    
    // Key landmark indices (MediaPipe standard)
    static const int LEFT_EYE_INNER = 133;
    static const int LEFT_EYE_OUTER = 33;
    static const int LEFT_EYE_TOP = 159;
    static const int LEFT_EYE_BOTTOM = 145;
    static const int LEFT_IRIS_CENTER = 468;
    
    static const int RIGHT_EYE_INNER = 362;
    static const int RIGHT_EYE_OUTER = 263;
    static const int RIGHT_EYE_TOP = 386;
    static const int RIGHT_EYE_BOTTOM = 374;
    static const int RIGHT_IRIS_CENTER = 473;
};
