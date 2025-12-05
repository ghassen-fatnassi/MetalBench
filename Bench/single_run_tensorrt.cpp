#include <iostream>
#include <vector>
#include <random>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

const std::string MODEL_PATH = "Models/yolo12n_op12.onnx";

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SingleRunTRT");
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetIntraOpNumThreads(2);
    session_options.SetInterOpNumThreads(2);

    // 1. Append TensorRT
    try {
        // Basic TRT options for 1.6.0
        OrtTensorRTProviderOptions trt_options;
        trt_options.device_id = 0;
        // trt_options.trt_fp16_enable = 1; // Uncomment if you want FP16 mode
        session_options.AppendExecutionProvider_TensorRT(trt_options);
        std::cout << "TensorRT Provider Appended." << std::endl;
    } catch (...) {
        std::cerr << "Failed to append TensorRT." << std::endl;
    }

    // 2. Append CUDA (Fall back)
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (...) {}

    try {
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);

        // Dummy Input (1, 3, 128, 128)
        std::vector<int64_t> dims = {1, 3, 128, 128};
        std::vector<float> vals(1*3*128*128, 0.5f);

        Ort::AllocatorWithDefaultOptions allocator;
        std::string in_name = session.GetInputName(0, allocator);
        std::string out_name = session.GetOutputName(0, allocator);
        const char* in_names[] = { in_name.c_str() };
        const char* out_names[] = { out_name.c_str() };

        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem_info, vals.data(), vals.size(), dims.data(), dims.size());

        std::cout << "Running TensorRT Inference..." << std::endl;
        session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
        std::cout << "Inference Completed." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}