#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <onnxruntime_cxx_api.h>

const std::string MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";
const int BATCH_SIZE = 1;
const int IMG_SIZE = 640;
const int NUM_WARMUP = 5;
const int NUM_RUNS   = 50;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Bench");
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(4);

    // CUDA provider
    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_TensorRT(cuda_options);
        std::cout << "CUDA Provider Appended.\n";
    } catch (...) {
        std::cerr << "WARNING: Could not append CUDA provider.\n";
    }

    try {
        std::cout << "Loading model: " << MODEL_PATH << std::endl;
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);

        // Prepare data
        std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
        size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
        std::vector<float> input_data(input_size);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& v : input_data) v = dis(gen);

        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        const char* input_names[] = {input_name_str.c_str()};

        std::string output_name_str = session.GetOutputName(0, allocator);
        const char* output_names[] = {output_name_str.c_str()};

        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_size,
            input_dims.data(), input_dims.size()
        );

        // Warmup
        std::cout << "Warmup...\n";
        for (int i = 0; i < NUM_WARMUP; i++) {
            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);
        }

        // Benchmark
        std::vector<double> times;
        times.reserve(NUM_RUNS);

        std::cout << "Benchmarking...\n";
        for (int i = 0; i < NUM_RUNS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            times.push_back(ms);
        }

        // Stats
        double sum = 0, mn = 1e9, mx = 0;
        for (auto v : times) {
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double avg = sum / times.size();

        std::cout << "\n---- RESULTS (" << NUM_RUNS << " runs) ----\n";
        std::cout << "Avg: " << avg << " ms\n";
        std::cout << "Min: " << mn << " ms\n";
        std::cout << "Max: " << mx << " ms\n";
        std::cout << "FPS: " << 1000.0 / avg << "\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
