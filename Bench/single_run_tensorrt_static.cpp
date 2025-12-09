// Bench/test_tensorrt_fixed.cpp
#include <iostream>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

static const char* MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";

int main() {
    try {
        // ---- ENV ----
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TRT_Test");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Get raw C API pointer for status/error handling
        const OrtApi* g_api = Ort::GetApi();

        // ---- APPEND TENSORRT EP (C API) ----
        {
            // Many ORT C++ wrappers provide implicit conversion to OrtSessionOptions*.
            // We obtain the raw pointer here.
            OrtSessionOptions* raw_opts = session_options;
            OrtStatus* st = OrtSessionOptionsAppendExecutionProvider_Tensorrt(raw_opts, 0);
            if (st) {
                const char* msg = g_api->GetErrorMessage(st);
                std::cerr << "[FAIL] Append TensorRT EP: " << (msg ? msg : "unknown") << "\n";
                g_api->ReleaseStatus(st);
            } else {
                std::cout << "[OK] TensorRT EP appended.\n";
            }
        }

        // ---- APPEND CUDA EP (C API) ----
        {
            OrtSessionOptions* raw_opts = session_options;
            OrtStatus* st = OrtSessionOptionsAppendExecutionProvider_CUDA(raw_opts, 0);
            if (st) {
                const char* msg = g_api->GetErrorMessage(st);
                std::cerr << "[WARN] Append CUDA EP: " << (msg ? msg : "unknown") << "\n";
                g_api->ReleaseStatus(st);
            } else {
                std::cout << "[OK] CUDA EP appended.\n";
            }
        }

        // ---- LOAD MODEL ----
        std::cout << "Loading: " << MODEL_PATH << std::endl;
        Ort::Session session(env, MODEL_PATH, session_options);

        // ---- INPUT ----
        Ort::AllocatorWithDefaultOptions alloc;

        // Get and copy input name, then free allocator memory
        char* input_name_cstr = session.GetInputName(0, alloc);
        if (!input_name_cstr) {
            std::cerr << "Failed to get input name\n";
            return -1;
        }
        std::string input_name_str(input_name_cstr);
        alloc.Free(input_name_cstr);

        // Get and copy output name, then free allocator memory
        char* output_name_cstr = session.GetOutputName(0, alloc);
        if (!output_name_cstr) {
            std::cerr << "Failed to get output name\n";
            return -1;
        }
        std::string output_name_str(output_name_cstr);
        alloc.Free(output_name_cstr);

        std::vector<int64_t> dims = {1, 3, 640, 640};
        const size_t input_size = 1 * 3 * 640 * 640;
        std::vector<float> input_data(input_size, 1.0f);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            dims.data(), dims.size());

        const char* input_names[] = { input_name_str.c_str() };
        const char* output_names[] = { output_name_str.c_str() };

        // ---- RUN ----
        std::cout << "Running inference...\n";
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        std::cout << "Done.\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << "\n";
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "STD Exception: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception\n";
        return -1;
    }

    return 0;
}
