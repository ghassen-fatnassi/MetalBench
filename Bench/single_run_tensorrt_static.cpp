#include <iostream>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

static const char* MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";

int main() {
    try {
        // ---- ENV ----
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TRT_Test");
        Ort::SessionOptions session_options;

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ---- APPEND TENSORRT EP (ORT 1.6 STYLE) ----
        try {
            int device_id = 0;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device_id));
            std::cout << "[OK] TensorRT EP appended.\n";
        } catch (...) {
            std::cout << "[FAIL] Could not append TensorRT EP.\n";
        }

        // ---- APPEND CUDA EP (required fallback) ----
        try {
            int device_id = 0;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id));
            std::cout << "[OK] CUDA EP appended.\n";
        } catch (...) {
            std::cout << "[FAIL] Could not append CUDA EP.\n";
        }

        // ---- LOAD MODEL ----
        std::cout << "Loading: " << MODEL_PATH << std::endl;
        Ort::Session session(env, MODEL_PATH, session_options);

        // ---- INPUT ----
        Ort::AllocatorWithDefaultOptions alloc;

        char* input_name = session.GetInputName(0, alloc);
        std::vector<int64_t> dims = {1, 3, 640, 640};
        const size_t input_size = 1 * 3 * 640 * 640;

        std::vector<float> input_data(input_size, 1.0f);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            dims.data(), dims.size());

        const char* input_names[] = { input_name };
        const char* output_names[] = { session.GetOutputName(0, alloc) };

        // ---- RUN ----
        std::cout << "Running inference...\n";
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        std::cout << "Done.\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << "\n";
    }

    return 0;
}
