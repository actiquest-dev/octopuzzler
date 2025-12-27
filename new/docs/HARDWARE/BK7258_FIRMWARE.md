#include "audio_capture.h"
#include <stdint.h>

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256  // Ring buffer chunks
#define CHUNK_SIZE 800   // 50ms @ 16kHz

typedef struct {
    uint16_t samples[CHUNK_SIZE];
    int write_pos;
    int read_pos;
} ring_buffer_t;

ring_buffer_t rb;

void init_audio() {
    // Configure ADC
    configure_adc(SAMPLE_RATE, 16);
    
    // Start continuous capture
    start_adc_dma(&rb.samples[0], sizeof(rb.samples));
    
    rb.write_pos = 0;
    rb.read_pos = 0;
}

uint16_t* audio_capture_chunk() {
    // Wait until chunk ready
    while(rb.write_pos == rb.read_pos) {
        sleep_ms(1);
    }
    
    uint16_t* chunk = &rb.samples[rb.read_pos * CHUNK_SIZE];
    rb.read_pos = (rb.read_pos + 1) % BUFFER_SIZE;
    
    return chunk;
}

// ADC DMA interrupt
void adc_dma_isr() {
    rb.write_pos = (rb.write_pos + 1) % BUFFER_SIZE;
}
