#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/common/logging/logging.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/session/ort_env.h>      // Environment::Create
#include <core/session/inference_session.h>

using namespace onnxruntime;

static const char* MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";

int main() {
    try {

        // ---------------------------
        // LOGGING
        // ---------------------------
        std::string log_id = "Foo";

        auto logging_manager = std::make_unique<LoggingManager>(
            std::unique_ptr<ISink>{new logging::CLogSink{}},
            Severity::kWARNING,
            /*filter_user_data=*/false,
            LoggingManager::InstanceType::Default,
            &log_id
        );

        std::unique_ptr<Environment> env;
        ORT_THROW_ON_ERROR(Environment::Create(std::move(logging_manager), env));

        // ---------------------------
        // SESSION OPTIONS
        // ---------------------------
        SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Append TensorRT EP (C API)
        int device_id = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(so, device_id));

        // ---------------------------
        // CREATE SESSION OBJECT
        // ---------------------------
        InferenceSession session_object{so, *env};

        // Load model AFTER EP registration
        ORT_THROW_ON_ERROR(session_object.Load(MODEL_PATH));
        ORT_THROW_ON_ERROR(session_object.Initialize());

        // ---------------------------
        // PREPARE INPUT
        // ---------------------------
        std::vector<int64_t> dims = {1, 3, 640, 640};
        size_t input_size = 1 * 3 * 640 * 640;
        std::vector<float> input_data(input_size, 1.0f);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_data.data(),
            input_data.size(),
            dims.data(),
            dims.size()
        );

        // manually get IO names
        Ort::AllocatorWithDefaultOptions alloc;
        const char* input_name = session_object.GetInputName(0, alloc);
        const char* output_name = session_object.GetOutputName(0, alloc);

        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};

        // ---------------------------
        // RUN INFERENCE
        // ---------------------------
        Ort::RunOptions run_options;
        std::vector<Ort::Value> outputs;

        ORT_THROW_ON_ERROR(session_object.Run(
            run_options,
            input_names, &input_tensor, 1,
            output_names, 1,
            outputs
        ));

        std::cout << "Inference complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
