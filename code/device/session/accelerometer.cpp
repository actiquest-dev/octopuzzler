class AccelerometerTrigger {
public:
    AccelerometerTrigger() {
        // Initialize I2C for MPU6050
        i2c_config_t conf = {
            .mode = I2C_MODE_MASTER,
            .sda_io_num = GPIO_NUM_21,
            .scl_io_num = GPIO_NUM_22,
            .sda_pullup_en = GPIO_PULLUP_ENABLE,
            .scl_pullup_en = GPIO_PULLUP_ENABLE,
            .master.clk_speed = 100000
        };
        
        i2c_param_config(I2C_NUM_0, &conf);
        i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0);
        
        // Configure MPU6050
        init_mpu6050();
    }
    
    bool detect_pickup_gesture() {
        // Read accelerometer
        AccelData accel = read_accel();
        
        // Calculate magnitude
        float magnitude = sqrt(
            accel.x * accel.x + 
            accel.y * accel.y + 
            accel.z * accel.z
        );
        
        // Detect sudden movement (>1.5g)
        if (magnitude > 1.5f * GRAVITY) {
            // Verify it's a pickup (not just vibration)
            if (verify_pickup_pattern()) {
                return true;
            }
        }
        
        return false;
    }
    
private:
    bool verify_pickup_pattern() {
        // Read samples over 500ms
        const int num_samples = 50;
        float samples[num_samples];
        
        for (int i = 0; i < num_samples; i++) {
            AccelData accel = read_accel();
            samples[i] = sqrt(
                accel.x * accel.x + 
                accel.y * accel.y + 
                accel.z * accel.z
            );
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        
        // Check for sustained elevation (pickup vs tap)
        int elevated_count = 0;
        for (int i = 0; i < num_samples; i++) {
            if (samples[i] > 1.2f * GRAVITY) {
                elevated_count++;
            }
        }
        
        // Pickup = sustained (>30 samples elevated)
        return elevated_count > 30;
    }
    
    AccelData read_accel() {
        uint8_t data[6];
        i2c_master_read_from_device(
            I2C_NUM_0, MPU6050_ADDR,
            MPU6050_ACCEL_XOUT_H, data, 6, 100
        );
        
        int16_t raw_x = (data[0] << 8) | data[1];
        int16_t raw_y = (data[2] << 8) | data[3];
        int16_t raw_z = (data[4] << 8) | data[5];
        
        return {
            .x = raw_x * ACCEL_SCALE,
            .y = raw_y * ACCEL_SCALE,
            .z = raw_z * ACCEL_SCALE
        };
    }
    
    const float ACCEL_SCALE = GRAVITY / 16384.0f;  // 2g range
    const float GRAVITY = 9.81f;
};
