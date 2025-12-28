/**
 * Wake Word Detector Header
 */

#ifndef WAKE_WORD_DETECTOR_H
#define WAKE_WORD_DETECTOR_H

#include <cstdint>

// Forward declarations
namespace tflite {
    class Model;
    class MicroInterpreter;
}

struct TfLiteTensor;

/**
 * Wake word detector statistics
 */
struct WakeWordStats {
    bool is_enabled;
    int buffer_fill;
    uint32_t last_detection_time;
    float detection_threshold;
};

/**
 * Wake Word Detector
 * 
 * Detects activation phrases using on-device keyword spotting.
 * 
 * Supported wake words:
 * - "привет" (Russian: hello)
 * - "okay" (English: okay)
 * - "hi octopus" (English: hi octopus)
 * 
 * Algorithm:
 * ----------
 * 1. Continuous audio buffering (1-second sliding window)
 * 2. MFCC feature extraction (40 coefficients × 98 frames)
 * 3. CNN-RNN inference (~15ms)
 * 4. Confidence thresholding (0.7)
 * 5. Debouncing (500ms cooldown)
 * 
 * Usage:
 * ------
 * WakeWordDetector detector;
 * detector.initialize();
 * 
 * // Audio loop (microphone callback)
 * void on_audio_data(int16_t* samples, int count) {
 *     detector.process_audio(samples, count);
 *     
 *     if (detector.is_wake_word_detected()) {
 *         printf("Wake word detected!\n");
 *         // Start session...
 *     }
 * }
 * 
 * Performance:
 * -----------
 * - Inference: ~15ms per window
 * - CPU: ~5% (continuous)
 * - RAM: ~5MB (tensor arena + buffers)
 */
class WakeWordDetector {
public:
    /**
     * Constructor
     */
    WakeWordDetector();
    
    /**
     * Destructor
     */
    ~WakeWordDetector();
    
    /**
     * Initialize detector
     * 
     * Loads TFLite model and allocates buffers.
     * 
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Process audio chunk
     * 
     * Call from microphone callback with new samples.
     * Automatically runs detection when enough data accumulated.
     * 
     * @param audio_chunk PCM16 audio samples
     * @param num_samples Number of samples in chunk
     */
    void process_audio(const int16_t* audio_chunk, int num_samples);
    
    /**
     * Check if wake word was detected
     * 
     * Returns true for 100ms after detection.
     * 
     * @return true if wake word detected recently
     */
    bool is_wake_word_detected();
    
    /**
     * Reset detector state
     * 
     * Clears audio buffer and detection flags.
     */
    void reset();
    
    /**
     * Enable/disable detection
     * 
     * When disabled, audio is still buffered but not processed.
     * 
     * @param enabled Enable state
     */
    void set_enabled(bool enabled);
    
    /**
     * Get detector statistics
     * 
     * @return Current statistics
     */
    WakeWordStats get_stats() const;

private:
    // TFLite components
    const tflite::Model* model_;
    tflite::MicroInterpreter* interpreter_;
    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_tensor_;
    uint8_t* tensor_arena_;
    
    // Audio buffers
    int16_t* audio_buffer_;  // Circular buffer (1 second)
    float* mfcc_features_;   // MFCC features (40 × 98)
    int buffer_write_pos_;
    
    // Detection state
    uint32_t last_detection_time_;
    bool is_enabled_;
    
    // Helper methods
    bool load_model(const char* model_path);
    void run_detection();
    void extract_mfcc_features();
    void compute_power_spectrum(const float* signal, int signal_length, float* power_spectrum, int fft_size);
    void apply_mel_filterbank(const float* power_spectrum, float* mel_energies);
    void normalize_features();
    uint32_t get_current_time_ms() const;
};

#endif // WAKE_WORD_DETECTOR_H