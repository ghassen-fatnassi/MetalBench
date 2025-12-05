// jetson_bench.cpp
// A Jetson-oriented ONNX Runtime benchmark (condensed translation of your Python script).
// Features:
// - Config generator (CUDA only by default in this translation)
// - Warmup iterations
// - NUM_RUNS measurement loop
// - Simple latency stats + JSON-ish output (no external JSON lib)

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <string>
#include <numeric>
#include <cmath>

using namespace std;
using clock_t = std::chrono::high_resolution_clock;

const char* MODEL_PATH = "Models/yolo12n_op12.onnx";
const int IMG_C = 3;
const int DEFAULT_RES = 128;
const int NUM_WARMUP = 3;
const int NUM_RUNS = 30;
const double COOLING_DELAY = 5.0; // seconds
const double TIMEOUT_SECONDS = 60.0;

bool file_exists(const string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

vector<float> generate_input(int batch, int resolution) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    vector<float> data(batch * IMG_C * resolution * resolution);
    for (auto &v : data) v = dist(rng);
    return data;
}

struct Config {
    int optimization; // use ORT enums
    int intra;
    int inter;
    int batch;
    int resolution;
    string execution_provider; // "CUDA" or "CPU"
    string description;
};

vector<Config> generate_test_configurations() {
    vector<Config> configs;
    // We'll only generate a small set similar to your Python script
    for (auto ep : {"CUDA"}) {
        for (auto opt : {GraphOptimizationLevel::ORT_DISABLE_ALL, GraphOptimizationLevel::ORT_ENABLE_EXTENDED}) {
            for (int batch : {1,2,4}) {
                for (int res = 128; res <= 384; res += 128) {
                    Config c;
                    c.optimization = opt;
                    c.intra = 2;
                    c.inter = 2;
                    c.batch = batch;
                    c.resolution = res;
                    c.execution_provider = ep;
                    c.description = string("EP:") + ep + ", opt:" + to_string(opt) + ", batch:" + to_string(batch) + ", res:" + to_string(res);
                    configs.push_back(c);
                }
            }
        }
    }
    return configs;
}

Ort::Session create_session_for_config(Ort::Env& env, const Config& cfg) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(cfg.intra);
    so.SetInterOpNumThreads(cfg.inter);
    so.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(cfg.optimization));
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    if (cfg.execution_provider == "CUDA") {
        OrtSessionOptions* raw = so;
        OrtSessionOptionsAppendExecutionProvider_CUDA(raw, 0);
    }
    // else CPU default

    return Ort::Session(env, MODEL_PATH, so);
}

struct Stats {
    int n;
    double mean_ms;
    double std_ms;
    double median_ms;
    double min_ms;
    double max_ms;
    double throughput_fps;
};

Stats calc_stats(const vector<double>& values_s) {
    Stats s{};
    if (values_s.empty()) {
        return s;
    }
    vector<double> v = values_s;
    int n = v.size();
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - mean)*(x - mean);
    double var = sq_sum / n;
    double stddev = sqrt(var);
    sort(v.begin(), v.end());
    double median = v[n/2];
    s.n = n;
    s.mean_ms = mean * 1000.0;
    s.std_ms = stddev * 1000.0;
    s.median_ms = median * 1000.0;
    s.min_ms = v.front() * 1000.0;
    s.max_ms = v.back() * 1000.0;
    s.throughput_fps = (mean > 0.0) ? (1.0 / mean) : 0.0;
    return s;
}

void do_cooling() {
    cout << "Cooling for " << COOLING_DELAY << "s\n";
    std::this_thread::sleep_for(std::chrono::duration<double>(COOLING_DELAY));
}

int benchmark_configuration(const Config& cfg, bool enable_profiling=false) {
    cout << "=== Running: " << cfg.description << " ===\n";
    do_cooling();
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "jetson_bench");
    Ort::Session session = create_session_for_config(env, cfg);

    vector<int64_t> input_shape = {cfg.batch, IMG_C, cfg.resolution, cfg.resolution};
    vector<float> input_data = generate_input(cfg.batch, cfg.resolution);

    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
    );

    char* input_name = session.GetInputNameAllocated(0, allocator).release();
    vector<const char*> input_names = {input_name};
    vector<const char*> output_names;
    size_t out_count = session.GetOutputCount();
    for (size_t i = 0; i < out_count; ++i) {
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).release());
    }

    // Warmup
    cout << "Warming up (" << NUM_WARMUP << " iters)\n";
    for (int i = 0; i < NUM_WARMUP; ++i) {
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
    }

    // Main runs
    vector<double> latencies_s;
    auto run_start = clock_t::now();
    for (int i = 0; i < NUM_RUNS; ++i) {
        auto t0 = clock_t::now();
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
        auto t1 = clock_t::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        latencies_s.push_back(s / cfg.batch);
    }
    auto run_end = clock_t::now();
    double run_seconds = std::chrono::duration<double>(run_end - run_start).count();

    auto stats = calc_stats(latencies_s);
    cout << "Completed " << NUM_RUNS << " runs in " << run_seconds << "s\n";
    cout << "Mean latency: " << stats.mean_ms << " ms, Throughput: " << stats.throughput_fps << " FPS\n";

    // Save summary to a simple text file
    string outname = "bench_result_" + to_string(cfg.batch) + "_" + to_string(cfg.resolution) + ".txt";
    ofstream f(outname);
    f << "description: " << cfg.description << "\n";
    f << "mean_ms: " << stats.mean_ms << "\n";
    f << "std_ms: " << stats.std_ms << "\n";
    f << "throughput_fps: " << stats.throughput_fps << "\n";
    f << "n_runs: " << stats.n << "\n";
    f.close();
    cout << "Saved to " << outname << "\n";

    return 0;
}

int main() {
    cout << "ONNX Runtime Jetson benchmark\n";
    if (!file_exists(MODEL_PATH)) {
        cerr << "Model missing: " << MODEL_PATH << "\n";
        return 1;
    }
    auto configs = generate_test_configurations();
    cout << "Generated " << configs.size() << " configurations\n";
    for (const auto &cfg : configs) {
        benchmark_configuration(cfg, false);
    }
    cout << "Benchmarking finished\n";
    return 0;
}
