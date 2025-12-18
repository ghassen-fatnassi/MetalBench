#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <thread>
#include <random>
#include <cstdlib> // Required for setenv
// REQUIRED for TensorRT in ORT 1.6
#include <onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

// ================= CONFIG =================
const std::string MODEL_PATH = "Models/optimized_cuda.onnx";
const int NUM_WARMUP = 5;       // Increased warmup for TRT stability
const int NUM_RUNS = 20;
const int COOLING_DELAY_MS = 5000; 

struct BenchmarkConfig {
    std::string description;
    GraphOptimizationLevel opt_level;
    int intra_op_threads;
    int inter_op_threads;
    int batch_size;
    int resolution;
};

struct BenchmarkResult {
    BenchmarkConfig config;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double throughput_fps;
    bool success;
    std::string error_msg;
};

// ================= HELPERS =================

std::vector<float> generate_random_input(size_t size) {
    std::vector<float> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : data) v = dis(gen);
    return data;
}

void print_result(const BenchmarkResult& res) {
    std::cout << "\n[RESULT] " << res.config.description << "\n";
    if (res.success) {
        std::cout << "  Mean Latency (per img): " << res.mean_ms << " ms (+/- " << res.std_ms << ")\n";
        std::cout << "  Throughput:             " << res.throughput_fps << " FPS\n";
        std::cout << "  Min/Max (per img):      " << res.min_ms << " / " << res.max_ms << " ms\n";
    } else {
        std::cout << "  FAILED: " << res.error_msg << "\n";
    }
    std::cout << "------------------------------------------------------------\n";
}

BenchmarkResult run_config(const BenchmarkConfig& config, Ort::Env& env) {
    BenchmarkResult res;
    res.config = config;
    
    std::cout << "\nCooling down for " << COOLING_DELAY_MS / 1000 << "s...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(COOLING_DELAY_MS));

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(config.intra_op_threads);
    so.SetInterOpNumThreads(config.inter_op_threads); 
    so.SetGraphOptimizationLevel(config.opt_level);
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    // --- TENSORRT PROVIDER SETUP (ORT 1.6 Legacy API) ---
    try {
        int device_id = 0;
        // In ORT 1.6, we use the C-API wrapper for TRT
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(so, device_id));
        // Note: For FP16 in ORT 1.6, you usually need to set the environment variable:
        // export ORT_TENSORRT_FP16_ENABLE=1
        std::cout << "TensorRT Provider appended (Device ID: " << device_id << ").\n";
    } catch (const std::exception& e) {
        res.success = false;
        res.error_msg = std::string("TRT Init Failed: ") + e.what();
        print_result(res);
        return res;
    }

    try {
        // Session creation
        // TRT builds the engine here. This can take several minutes on the first run!
        std::cout << "Creating Session (Building TensorRT Engine, please wait...)\n";
        auto t_start_build = std::chrono::high_resolution_clock::now();
        
        Ort::Session session(env, MODEL_PATH.c_str(), so);
        
        auto t_end_build = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t_end_build - t_start_build).count();
        std::cout << "Session created. Build time: " << build_time << " seconds.\n";

        // Input Setup
        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        const char* input_names[] = { input_name_str.c_str() };
        
        std::string output_name_str = session.GetOutputName(0, allocator);
        const char* output_names[] = { output_name_str.c_str() };

        std::vector<int64_t> input_dims = {
            config.batch_size, 3, config.resolution, config.resolution
        };
        size_t input_size = (size_t)config.batch_size * 3 * config.resolution * config.resolution;
        auto input_data = generate_random_input(input_size);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );

        // Warmup
        std::cout << "Warming up...\n";
        for(int i=0; i<NUM_WARMUP; i++) {
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        }

        // Measure
        std::cout << "Measuring " << NUM_RUNS << " runs...\n";
        std::vector<double> latencies_per_image;
        latencies_per_image.reserve(NUM_RUNS);

        for(int i=0; i<NUM_RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            latencies_per_image.push_back(elapsed.count() / config.batch_size); 
        }

        // Stats
        double sum = std::accumulate(latencies_per_image.begin(), latencies_per_image.end(), 0.0);
        res.mean_ms = sum / latencies_per_image.size();
        res.throughput_fps = 1000.0 / res.mean_ms; 
        
        double sq_sum = std::inner_product(latencies_per_image.begin(), latencies_per_image.end(), latencies_per_image.begin(), 0.0);
        res.std_ms = std::sqrt(sq_sum / latencies_per_image.size() - res.mean_ms * res.mean_ms);
        
        auto minmax = std::minmax_element(latencies_per_image.begin(), latencies_per_image.end());
        res.min_ms = *minmax.first;
        res.max_ms = *minmax.second;
        res.success = true;

    } catch (const Ort::Exception& e) {
        res.success = false;
        res.error_msg = e.what();
    }

    print_result(res);
    return res;
}

// ================= MAIN =================
int main() {
    // --- LINUX INTEGRATION: Set Env Vars Automatically ---
    // 1. Enable TensorRT Engine Caching (Critical for multiple runs)
    setenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1", 1);
    
    // 2. Set the cache path to a folder named 'trt_cache' in the current dir
    //    Make sure to create this folder or let TRT create it
    setenv("ORT_TENSORRT_CACHE_PATH", "./trt_engine_cache", 1);

    // 3. Enable FP16 (Optional, but recommended for Jetson)
    setenv("ORT_TENSORRT_FP16_ENABLE", "1", 1);
    
    // 4. Create the directory if it doesn't exist (Linux command)
    system("mkdir -p ./trt_engine_cache");
    // -----------------------------------------------------

    std::cout << "==========================================\n";
    std::cout << "ONNX Runtime 1.6 - TensorRT Benchmark\n";
    std::cout << "==========================================\n";
    std::cout << "ONNX Runtime 1.6 - TensorRT Benchmark\n";
    std::cout << "Model: " << MODEL_PATH << "\n";
    std::cout << "==========================================\n";

    // ORT 1.6 TRT logging is usually controlled by env vars or ORT logging level
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TensorRTBench");

    std::vector<BenchmarkConfig> configs;

    // TensorRT Benchmarks
    // Note: Changing input size in TRT often triggers a full engine rebuild 
    // for each config unless dynamic shapes are pre-configured.
    std::vector<int> batches = {1, 4, 8};
    std::vector<int> resolutions = {320, 640}; 

    for (int b : batches) {
        for (int r : resolutions) {
            BenchmarkConfig c;
            c.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL; // Let TRT optimize
            c.intra_op_threads = 2;
            c.inter_op_threads = 2; 
            c.batch_size = b;
            c.resolution = r;
            c.description = "TensorRT | Batch:" + std::to_string(b) + " | Res:" + std::to_string(r);
            configs.push_back(c);
        }
    }
    
    std::cout << "Generated " << configs.size() << " configurations.\n";
    std::cout << "NOTE: TensorRT engine building happens at the start of EACH configuration.\n";
    std::cout << "This may take 1-3 minutes per config. Please be patient.\n";
    std::cout << "------------------------------------------------------------\n";

    int success_count = 0;
    for (const auto& conf : configs) {
        BenchmarkResult r = run_config(conf, env);
        if(r.success) success_count++;
    }

    std::cout << "\n==========================================\n";
    std::cout << "Benchmark Finished. Successful: " << success_count << "/" << configs.size() << std::endl;
    std::cout << "==========================================\n";
    return 0;
}