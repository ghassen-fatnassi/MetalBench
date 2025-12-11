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
// Increased cooling delay to 5 seconds (5000ms) to ensure CUDA cleanup
const int COOLING_DELAY_MS = 5000; 

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
        // We report latency per image since the latency vector was calculated per image
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
    
    // Cool down: Pause to allow resources from the previous run to fully release
    std::cout << "\nCooling down for " << COOLING_DELAY_MS / 1000 << "s before running: " << config.description << "...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(COOLING_DELAY_MS));

    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(config.intra_op_threads);
    // Inter-op threads set to 1 to reduce contention on the CPU side during setup
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
            print_result(res);
            return res;
        }
    }

    try {
        // Session creation (This is where the potential failure happens)
        std::cout << "Attempting to create session for " << config.description << "...\n";
        Ort::Session session(env, MODEL_PATH.c_str(), so);
        std::cout << "Session created successfully.\n";

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
            // Record latency per image (Total elapsed time / batch size)
            latencies_per_image.push_back(elapsed.count() / config.batch_size); 
        }

        // Stats
        double sum = std::accumulate(latencies_per_image.begin(), latencies_per_image.end(), 0.0);
        res.mean_ms = sum / latencies_per_image.size();
        // Throughput is calculated based on the mean latency per image
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
    std::cout << "==========================================\n";
    std::cout << "ONNX Runtime Benchmark (C++)\n";
    std::cout << "Model: " << MODEL_PATH << "\n";
    std::cout << "==========================================\n";

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "JetsonBench");

    // Define Configurations
    std::vector<BenchmarkConfig> configs;

    // CUDA Configs (Optimization Level is ORT_DISABLE_ALL)
    std::vector<int> batches = {1, 2, 4, 8};
    std::vector<int> resolutions; 
    for(int r=128; r<=512; r+=128) resolutions.push_back(r);

    for (int b : batches) {
        for (int r : resolutions) {
            BenchmarkConfig c;
            c.execution_provider = "CUDA";
            c.opt_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
            c.intra_op_threads = 2;
            c.inter_op_threads = 1; // Recommended change for dynamic CUDA
            c.batch_size = b;
            c.resolution = r;
            c.description = "EP:CUDA, Batch:" + std::to_string(b) + ", Res:" + std::to_string(r);
            configs.push_back(c);
        }
    }
    
    std::cout << "Generated " << configs.size() << " configurations.\n";
    std::cout << "------------------------------------------------------------\n";

    // Run Benchmarks
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