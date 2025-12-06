#include <iostream>
#include <vector>
#include <random>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Check for provider availability (macros usually defined by build system)
// For 1.6.0 we use the C-API struct or specific helper
#include <onnxruntime/core/session/onnxruntime_c_api.h>

const std::string MODEL_PATH = "Models/yolo12n_op12.onnx";
const int BATCH_SIZE = 1;
const int IMG_SIZE = 128;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SingleRunCUDA");
    Ort::SessionOptions session_options;

    // Optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetOptimizedModelFilePath("optimized_cuda.onnx");
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(4);

    // Append CUDA Provider
    // Note: ORT 1.6.0 C++ API uses OrtCUDAProviderOptions
    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;

        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "CUDA Provider Appended." << std::endl;
    } catch (...) {
        std::cerr << "Warning: Could not append CUDA provider. Is ORT built with CUDA?" << std::endl;
    }

    try {
        std::cout << "Loading model: " << MODEL_PATH << std::endl;
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);

        // Data Prep
        std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
        size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
        std::vector<float> input_data(input_size);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for(auto& v : input_data) v = dis(gen);

        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        const char* input_names[] = { input_name_str.c_str() };
        
        std::string output_name_str = session.GetOutputName(0, allocator);
        const char* output_names[] = { output_name_str.c_str() };

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );

        std::cout << "Running single GPU inference..." << std::endl;
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        std::cout << "Single GPU inference completed." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}