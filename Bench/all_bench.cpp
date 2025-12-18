#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <thread>
#include <random>
#include <cstdlib>
#include <cstdio>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h>
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

// ================= CONFIG =================
const int NUM_WARMUP = 3;
const int NUM_RUNS   = 10;
const int COOLING_DELAY_MS = 3000;

struct BenchmarkConfig {
    std::string ep_name;      // CPU | CUDA | TRT
    int batch_size;
    int resolution;
    std::string model_path;
    GraphOptimizationLevel opt_level;
};

struct BenchmarkResult {
    BenchmarkConfig config;
    double mean_ms;
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

void save_results_to_csv(const std::vector<BenchmarkResult>& results,
                         const std::string& filename) {
    std::ofstream file(filename);
    file << "Provider,Batch,Resolution,Latency_ms_per_img,Throughput_FPS,Status\n";
    for (const auto& r : results) {
        file << r.config.ep_name << ","
             << r.config.batch_size << ","
             << r.config.resolution << ",";
        if (r.success) {
            file << r.mean_ms << "," << r.throughput_fps << ",Success\n";
        } else {
            file << "0,0,Failed: " << r.error_msg << "\n";
        }
    }
    std::cout << "\n[INFO] Results saved to " << filename << "\n";
}

BenchmarkResult run_benchmark(const BenchmarkConfig& config, Ort::Env& env) {
    BenchmarkResult res{};
    res.config = config;
    res.success = false;

    std::cout << "\nCooling for " << COOLING_DELAY_MS/1000 << "s | "
              << config.ep_name << " B:" << config.batch_size
              << " R:" << config.resolution << "..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(COOLING_DELAY_MS));

    try {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(config.opt_level);
        so.SetIntraOpNumThreads(2);
        so.SetInterOpNumThreads(2);

        // -------- Execution Provider Selection (SAFE ORDER) --------
        if (config.ep_name == "TRT") {
            // Avoid FP16 for tiny shapes (prevents kernel build timeouts)
            if (config.resolution < 256) {
                unsetenv("ORT_TENSORRT_FP16_ENABLE");
            }

            int device_id = 0;
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_Tensorrt(so, device_id));
        }
        else if (config.ep_name == "CUDA") {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            so.AppendExecutionProvider_CUDA(cuda_opts);
        }
        // CPU → no EP appended

        // -------- Session Creation (engine build happens here for TRT) --------
        Ort::Session session(env, config.model_path.c_str(), so);
        Ort::AllocatorWithDefaultOptions allocator;

        char* in_name  = session.GetInputName(0, allocator);
        char* out_name = session.GetOutputName(0, allocator);
        const char* input_names[]  = { in_name };
        const char* output_names[] = { out_name };

        std::vector<int64_t> dims = {
            config.batch_size, 3, config.resolution, config.resolution
        };
        size_t count = (size_t)config.batch_size * 3
                     * config.resolution * config.resolution;

        auto input_data = generate_random_input(count);
        auto mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, input_data.data(), count, dims.data(), dims.size());

        // -------- Force TRT engine build (not timed) --------
        session.Run(Ort::RunOptions{nullptr},
                    input_names, &input_tensor, 1,
                    output_names, 1);

        // -------- Warmup --------
        for (int i = 0; i < NUM_WARMUP; ++i) {
            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);
        }

        // -------- Timed Runs --------
        std::vector<double> latencies;
        latencies.reserve(NUM_RUNS);

        for (int i = 0; i < NUM_RUNS; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        double avg_total = std::accumulate(
            latencies.begin(), latencies.end(), 0.0) / NUM_RUNS;

        res.mean_ms = avg_total / config.batch_size;
        res.throughput_fps = 1000.0 / res.mean_ms;
        res.success = true;

    } catch (const std::exception& e) {
        res.error_msg = e.what();
        std::cerr << "FAILED: " << e.what() << std::endl;
    }

    return res;
}

int main() {
    // -------- Global TRT Environment --------
    setenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1", 1);
    setenv("ORT_TENSORRT_CACHE_PATH", "./trt_cache", 1);
    setenv("ORT_TENSORRT_FP16_ENABLE", "0", 1);
    system("mkdir -p ./trt_cache");

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "MetalBenchUnifiedSafe");
    std::vector<BenchmarkResult> all_results;

    // IMPORTANT: TRT must run BEFORE CUDA in same process
    std::vector<std::string> providers = { "TRT", "CPU", "CUDA"};
    std::vector<int> batches = {1, 2, 4, 8};
    std::vector<int> resolutions = {128, 256, 384, 512, 640};

    std::cout << "Starting Unified Benchmark (TRT-safe order)..." << std::endl;

    for (const auto& ep : providers) {
        for (int b : batches) {
            for (int r : resolutions) {
                // Skip pathological TRT configs
                if (ep == "TRT" && r < 256) continue;

                BenchmarkConfig conf;
                conf.ep_name = ep;
                conf.batch_size = b;
                conf.resolution = r;
                conf.model_path = "Models/yolo12n_op12_static_"
                                + std::to_string(b) + "_"
                                + std::to_string(r) + ".onnx";
                conf.opt_level = ORT_DISABLE_ALL;

                all_results.push_back(run_benchmark(conf, env));
            }
        }
    }

    // -------- Summary --------
    std::cout << "\n==================================================\n";
    std::cout << "SUMMARY TABLE\n";
    std::cout << "EP\tBatch\tRes\tFPS\n";
    for (const auto& r : all_results) {
        if (r.success) {
            std::printf("%s\t%d\t%d\t%.2f\n",
                        r.config.ep_name.c_str(),
                        r.config.batch_size,
                        r.config.resolution,
                        r.throughput_fps);
        }
    }

    save_results_to_csv(all_results, "benchmark_results.csv");
    return 0;
}
