/**
 * Octopus AI Device Configuration
 * 
 * Hardware and software configuration parameters.
 * Modify these values to tune system behavior.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================
// Hardware Configuration
// ============================================================

// Display (Dual ST7789 320×160 each)
#define DISPLAY_WIDTH       320
#define DISPLAY_HEIGHT      160
#define DISPLAY_ROTATION    0
#define DISPLAY_SPI_FREQ    40000000  // 40 MHz
#define DISPLAY_FPS         30

// Camera (GC0308 VGA)
#define CAMERA_WIDTH        640
#define CAMERA_HEIGHT       480
#define CAMERA_FPS          15
#define CAMERA_FORMAT       CAMERA_FORMAT_RGB888

// Audio
#define AUDIO_SAMPLE_RATE   16000
#define AUDIO_CHANNELS      1
#define AUDIO_BIT_DEPTH     16
#define AUDIO_BUFFER_SIZE   4096

// ============================================================
// Avatar Configuration
// ============================================================

// Octopus body
#define AVATAR_BODY_RADIUS      35.0f
#define AVATAR_TENTACLE_COUNT   8
#define AVATAR_TENTACLE_LENGTH  30.0f
#define AVATAR_TENTACLE_WIDTH   6.0f

// Eyes
#define AVATAR_EYE_RADIUS       8.0f
#define AVATAR_PUPIL_RADIUS     4.0f
#define AVATAR_EYE_SPACING      20.0f

// Animation
#define AVATAR_EMOTION_BLEND_TIME   0.5f  // seconds
#define AVATAR_WIGGLE_SPEED         2.0f  // Hz

// ============================================================
// Eye Tracking Configuration
// ============================================================

// Frame intervals
#define EYE_TRACKING_CAMERA_FPS         15
#define EYE_TRACKING_PROCESS_FPS        5
#define EYE_TRACKING_DISPLAY_FPS        30

#define FACE_DETECTION_INTERVAL         6   // Every 6 frames (2.5 FPS)
#define EYE_TRACKING_INTERVAL           3   // Every 3 frames (5 FPS)

// Models
#define BLAZEFACE_MODEL_PATH        "/models/blazeface.tflite"
#define FACEMESH_MODEL_PATH         "/models/facemesh_lite.tflite"

// Face ROI
#define FACE_ROI_SIZE               200

// Gaze smoothing
#define GAZE_HISTORY_SIZE           5

// Blink detection
#define BLINK_EAR_THRESHOLD         0.2f

// ============================================================
// Session Configuration
// ============================================================

// Wake word
#define WAKE_WORD_MODEL_PATH        "/models/wake_word.tflite"
#define WAKE_WORD_THRESHOLD         0.7f
#define WAKE_WORDS                  {"привет", "okay", "hi octopus"}

// Session timeouts
#define SESSION_IDLE_TIMEOUT        30000   // 30 seconds (ms)
#define SESSION_MAX_DURATION        300000  // 5 minutes (ms)

// Face capture
#define FACE_CAPTURE_DELAY          500     // 500ms after wake word
#define FACE_IMAGE_SIZE             200     // 200×200

// ============================================================
// Network Configuration
// ============================================================

// WiFi
#define WIFI_SSID_MAX_LEN           32
#define WIFI_PASSWORD_MAX_LEN       64
#define WIFI_CONNECT_TIMEOUT        10000   // 10 seconds

// WebRTC
#define RTC_SERVER_URL              "wss://rtc.octopus-ai.com"
#define RTC_RECONNECT_DELAY         5000    // 5 seconds
#define RTC_PING_INTERVAL           30000   // 30 seconds

// Audio streaming
#define RTC_AUDIO_CODEC             "G711A"
#define RTC_AUDIO_BITRATE           64000   // 64 kbps
#define RTC_AUDIO_PACKET_SIZE       160     // 20ms @ 16kHz

// ============================================================
// Memory Configuration
// ============================================================

// RAM allocation
#define HEAP_SIZE                   50 * 1024 * 1024   // 50MB
#define RTC_SDK_MEMORY              80 * 1024 * 1024   // 80MB
#define CAMERA_BUFFERS              3
#define CAMERA_BUFFER_SIZE          640 * 480 * 3      // RGB888
#define AUDIO_BUFFER_COUNT          16
#define AUDIO_SINGLE_BUFFER_SIZE    4096

// TFLite arena sizes
#define BLAZEFACE_ARENA_SIZE        20 * 1024 * 1024   // 20MB
#define FACEMESH_ARENA_SIZE         10 * 1024 * 1024   // 10MB
#define WAKE_WORD_ARENA_SIZE        5 * 1024 * 1024    // 5MB

// ============================================================
// Performance Tuning
// ============================================================

// CPU affinity
#define CPU_CORE_NETWORK            0   // WiFi, RTC, Audio on CPU0
#define CPU_CORE_VISION             1   // Camera, TFLite, Avatar on CPU1

// Task priorities (FreeRTOS)
#define PRIORITY_NETWORK            5
#define PRIORITY_AUDIO_CAPTURE      4
#define PRIORITY_AUDIO_PLAYBACK     4
#define PRIORITY_CAMERA             3
#define PRIORITY_EYE_TRACKING       2
#define PRIORITY_DISPLAY            2
#define PRIORITY_SESSION            1

// Stack sizes
#define STACK_SIZE_NETWORK          8192
#define STACK_SIZE_AUDIO            4096
#define STACK_SIZE_CAMERA           4096
#define STACK_SIZE_TRACKING         8192
#define STACK_SIZE_DISPLAY          4096
#define STACK_SIZE_SESSION          4096

// ============================================================
// Debug Configuration
// ============================================================

// Enable debug features
#define DEBUG_ENABLE                1

#if DEBUG_ENABLE
    #define DEBUG_UART              1
    #define DEBUG_PRINT_FPS         1
    #define DEBUG_PRINT_MEMORY      1
    #define DEBUG_PRINT_LATENCY     1
    
    #define LOG_LEVEL               LOG_LEVEL_DEBUG
#else
    #define LOG_LEVEL               LOG_LEVEL_INFO
#endif

// Log levels
#define LOG_LEVEL_ERROR             0
#define LOG_LEVEL_WARNING           1
#define LOG_LEVEL_INFO              2
#define LOG_LEVEL_DEBUG             3

// UART debug
#define DEBUG_UART_PORT             0
#define DEBUG_UART_BAUD             115200

// ============================================================
// Feature Flags
// ============================================================

#define FEATURE_EYE_TRACKING        1
#define FEATURE_FACE_RECOGNITION    1
#define FEATURE_WAKE_WORD           1
#define FEATURE_EMOTION_ANIMATION   1

// ============================================================
// Assertions
// ============================================================

#if DEBUG_ENABLE
    #define ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                printf("ASSERT FAILED: %s at %s:%d\n", message, __FILE__, __LINE__); \
                while(1); \
            } \
        } while(0)
#else
    #define ASSERT(condition, message) ((void)0)
#endif

#endif // CONFIG_H