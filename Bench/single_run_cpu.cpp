#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Configuration
const std::string MODEL_PATH = "Models/yolo12n_op12.onnx";
const int BATCH_SIZE = 1;
const int IMG_CHANNELS = 3;
const int IMG_HEIGHT = 128;
const int IMG_WIDTH = 128;

int main() {
    // 1. Setup Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SingleRunCPU");
    Ort::SessionOptions session_options;

    // CPU Optimizations matching your Python script
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetOptimizedModelFilePath("optimized_cpu.onnx");
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();

    std::cout << "Loading model: " << MODEL_PATH << "..." << std::endl;
    
    try {
        // 2. Create Session
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);

        // 3. Prepare Input Data
        size_t input_tensor_size = BATCH_SIZE * IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Random initialization
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& val : input_tensor_values) val = dis(gen);

        // 4. Input Info
        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name = session.GetInputName(0, allocator);
        const char* input_names[] = { input_name.c_str() };
        
        std::vector<int64_t> input_node_dims = {BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH};

        // 5. Create Tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_size, 
            input_node_dims.data(), 
            input_node_dims.size()
        );

        // 6. Run Inference
        std::cout << "Running single CPU inference with input shape: (" 
                  << BATCH_SIZE << ", " << IMG_CHANNELS << ", " << IMG_HEIGHT << ", " << IMG_WIDTH << ")" << std::endl;

        // Get Output Name
        std::string output_name = session.GetOutputName(0, allocator);
        const char* output_names[] = { output_name.c_str() };

        auto outputs = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1
        );

        std::cout << "Single CPU inference completed." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}