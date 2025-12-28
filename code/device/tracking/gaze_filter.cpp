class GazeFilter {
public:
    GazeFilter(int window_size = 5) : window_size(window_size) {
        history_x.reserve(window_size);
        history_y.reserve(window_size);
    }
    
    struct Gaze {
        float x, y;
    };
    
    Gaze filter(float raw_x, float raw_y) {
        // Add to history
        history_x.push_back(raw_x);
        history_y.push_back(raw_y);
        
        // Maintain window size
        if (history_x.size() > window_size) {
            history_x.erase(history_x.begin());
            history_y.erase(history_y.begin());
        }
        
        // Moving average
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        
        for (size_t i = 0; i < history_x.size(); i++) {
            sum_x += history_x[i];
            sum_y += history_y[i];
        }
        
        return {
            sum_x / history_x.size(),
            sum_y / history_y.size()
        };
    }
    
    void reset() {
        history_x.clear();
        history_y.clear();
    }
    
private:
    int window_size;
    std::vector<float> history_x;
    std::vector<float> history_y;
};
