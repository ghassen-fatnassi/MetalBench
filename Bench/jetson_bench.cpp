#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <thread>
#include <random>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// ================= CONFIG =================
const std::string MODEL_PATH = "Models/yolo12n_op12.onnx";
const int NUM_WARMUP = 3;
const int NUM_RUNS = 30;
const int COOLING_DELAY_MS = 2000; // 2 seconds between configs

struct BenchmarkConfig {
    std::string description;
    std::string execution_provider; // "CPU", "CUDA"
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
        std::cout << "  Mean Latency: " << res.mean_ms << " ms (+/- " << res.std_ms << ")\n";
        std::cout << "  Throughput:   " << res.throughput_fps << " FPS\n";
        std::cout << "  Min/Max:      " << res.min_ms << " / " << res.max_ms << " ms\n";
    } else {
        std::cout << "  FAILED: " << res.error_msg << "\n";
    }
    std::cout << "------------------------------------------------------------\n";
}

BenchmarkResult run_config(const BenchmarkConfig& config, Ort::Env& env) {
    BenchmarkResult res;
    res.config = config;
    
    // Cool down
    std::this_thread::sleep_for(std::chrono::milliseconds(COOLING_DELAY_MS));

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(config.intra_op_threads);
    so.SetInterOpNumThreads(config.inter_op_threads);
    so.SetGraphOptimizationLevel(config.opt_level);
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    // Provider Setup
    if (config.execution_provider == "CUDA") {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
            so.AppendExecutionProvider_CUDA(cuda_opts);
        } catch(std::exception& e) {
            res.success = false;
            res.error_msg = std::string("CUDA Init Failed: ") + e.what();
            return res;
        }
    }

    try {
        Ort::Session session(env, MODEL_PATH.c_str(), so);
        
        // Input Setup
        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        const char* input_names[] = { input_name_str.c_str() };
        
        std::string output_name_str = session.GetOutputName(0, allocator);
        const char* output_names[] = { output_name_str.c_str() };

        std::vector<int64_t> input_dims = {
            config.batch_size, 3, config.resolution, config.resolution
        };
        size_t input_size = config.batch_size * 3 * config.resolution * config.resolution;
        auto input_data = generate_random_input(input_size);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );

        // Warmup
        for(int i=0; i<NUM_WARMUP; i++) {
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        }

        // Measure
        std::vector<double> latencies;
        latencies.reserve(NUM_RUNS);

        for(int i=0; i<NUM_RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> elapsed = end - start;
            latencies.push_back(elapsed.count() / config.batch_size); // Latency per image if preferred, or per batch
        }

        // Stats
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        res.mean_ms = sum / latencies.size();
        res.throughput_fps = 1000.0 / res.mean_ms * config.batch_size;
        
        double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
        res.std_ms = std::sqrt(sq_sum / latencies.size() - res.mean_ms * res.mean_ms);
        
        auto minmax = std::minmax_element(latencies.begin(), latencies.end());
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
    std::cout << "==========================================\n";
    std::cout << "ONNX Runtime Benchmark for Jetson (C++)\n";
    std::cout << "Model: " << MODEL_PATH << "\n";
    std::cout << "==========================================\n";

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "JetsonBench");

    // Define Configurations (mimicking the Python generator)
    std::vector<BenchmarkConfig> configs;

    // CUDA Configs
    std::vector<int> batches = {1, 2, 4, 8};
    // Reduced resolution list for example, add more if needed
    std::vector<int> resolutions; 
    for(int r=128; r<=512; r+=128) resolutions.push_back(r);

    for (int b : batches) {
        for (int r : resolutions) {
            BenchmarkConfig c;
            c.execution_provider = "CUDA";
            c.opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
            c.intra_op_threads = 2;
            c.inter_op_threads = 2;
            c.batch_size = b;
            c.resolution = r;
            c.description = "EP:CUDA, Batch:" + std::to_string(b) + ", Res:" + std::to_string(r);
            configs.push_back(c);
        }
    }
    
    // You can add CPU configs here if desired
    
    std::cout << "Generated " << configs.size() << " configurations.\n";

    // Run Benchmarks
    int success_count = 0;
    for (const auto& conf : configs) {
        BenchmarkResult r = run_config(conf, env);
        if(r.success) success_count++;
    }

    std::cout << "Benchmark Finished. Successful: " << success_count << "/" << configs.size() << std::endl;
    return 0;
}