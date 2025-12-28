void camera_task(void* param) {
    OptimizedEyeTracker* tracker = (OptimizedEyeTracker*)param;
    
    // Initialize camera
    camera_config_t cam_config = {
        .width = 640,
        .height = 480,
        .fps = 15,
        .format = CAMERA_FORMAT_RGB888
    };
    
    camera_handle_t camera = bk_camera_init(&cam_config);
    if (!camera) {
        printf("Failed to initialize camera\n");
        vTaskDelete(NULL);
        return;
    }
    
    printf("Camera task started\n");
    
    while (true) {
        uint8_t* frame = nullptr;
        size_t frame_len = 0;
        
        // Capture frame
        if (bk_camera_capture(camera, &frame, &frame_len) == BK_OK) {
            // Process frame
            // - BlazeFace every 6 frames (2.5 FPS)
            // - MediaPipe every 3 frames (5 FPS)
            tracker->process_frame(frame, 640, 480);
            
            // Release frame
            bk_camera_release_frame(camera, frame);
        }
        
        // 15 FPS = 66ms per frame
        vTaskDelay(pdMS_TO_TICKS(66));
    }
}
