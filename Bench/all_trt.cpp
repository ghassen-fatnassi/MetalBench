#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

// ============================================================================
// PRECISION MODE ENUM
// ============================================================================
enum class PrecisionMode { FP32, FP16 };

std::string precisionToString(PrecisionMode pm) {
    return (pm == PrecisionMode::FP32) ? "fp32" : "fp16";
}

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================
struct ModelConfig {
    std::string name;
    std:: string static_path;
    int channels;
    int height;
    int width;
};

const std::vector<ModelConfig> MODELS = {
    {
        "unet",
        "Models/unet/UNET_static.onnx",
        3, 512, 512
    },
    {
        "mobilenetv2",
        "Models/mobilenetv2/MobileNetV2_static.onnx",
        3, 224, 224
    }
};

const std::vector<PrecisionMode> PRECISION_MODES = {
    PrecisionMode::FP32,
    PrecisionMode::FP16
};

// ============================================================================
// PROFILING CONFIGURATION
// ============================================================================
const int WARMUP_RUNS = 10;
const int BENCHMARK_RUNS = 100;
const int COOLDOWN_SECONDS = 1;
const std::string OUTPUT_BASE_DIR = "profiling_results_trt";
const int BATCH_SIZE = 1;  // Static models with batch size 1 only

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct LatencyStats {
    double min_ms = 0, max_ms = 0, mean_ms = 0, std_dev_ms = 0;
    double p50_ms = 0, p90_ms = 0, p95_ms = 0, p99_ms = 0;
    double jitter_ms = 0;
    double variance_ratio = 0;
};

struct IterationMetrics {
    int iteration;
    double wall_time_ms;
};

struct RunConfig {
    std::string model_name;
    std:: string model_path;
    int batch_size;
    int channels, height, width;
    std::string output_dir;
    PrecisionMode precision;
};

struct RunSummary {
    LatencyStats latency;
    double throughput_fps;
    long max_rss_kb;
};

// ============================================================================
// LATENCY ANALYZER
// ============================================================================
class LatencyAnalyzer {
public:
    static LatencyStats calculate(const std::vector<double>& latencies_ms) {
        LatencyStats stats = {};
        if (latencies_ms.empty()) return stats;
        
        std::vector<double> sorted = latencies_ms;
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        stats.min_ms = sorted.front();
        stats.max_ms = sorted.back();
        
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        stats.mean_ms = sum / n;
        
        double sq_sum = 0.0;
        for (double val : sorted) {
            sq_sum += (val - stats.mean_ms) * (val - stats.mean_ms);
        }
        stats.std_dev_ms = std::sqrt(sq_sum / n);
        
        stats.p50_ms = getPercentile(sorted, 50.0);
        stats.p90_ms = getPercentile(sorted, 90.0);
        stats.p95_ms = getPercentile(sorted, 95.0);
        stats.p99_ms = getPercentile(sorted, 99.0);
        
        stats.jitter_ms = stats.max_ms - stats.min_ms;
        stats.variance_ratio = (stats.min_ms > 0) ? (stats.max_ms / stats.min_ms) : 0;
        
        return stats;
    }

private:
    static double getPercentile(const std::vector<double>& sorted, double p) {
        double rank = (p / 100.0) * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = static_cast<size_t>(std::ceil(rank));
        if (lo == hi) return sorted[lo];
        return sorted[lo] * (1.0 - (rank - lo)) + sorted[hi] * (rank - lo);
    }
};

// ============================================================================
// METRICS RECORDER
// ============================================================================
class MetricsRecorder {
public:
    void clear() {
        iterations_.clear();
        latencies_ms_.clear();
    }
    
    void record(int iteration, double wall_time_ms) {
        iterations_.push_back({iteration, wall_time_ms});
        latencies_ms_.push_back(wall_time_ms);
    }
    
    LatencyStats computeLatencyStats() {
        return LatencyAnalyzer::calculate(latencies_ms_);
    }
    
    void saveToJson(const std::string& filepath, const RunConfig& config, const RunSummary& summary) {
        std::ofstream file(filepath);
        
        file << "{\n";
        file << "  \"config\": {\n";
        file << "    \"model_name\":  \"" << config.model_name << "\",\n";
        file << "    \"precision\": \"" << precisionToString(config.precision) << "\",\n";
        file << "    \"batch_size\": " << config.batch_size << ",\n";
        file << "    \"execution_provider\": \"tensorrt\",\n";
        file << "    \"input_shape\": [" << config.batch_size << ", " << config.channels
             << ", " << config.height << ", " << config.width << "]\n";
        file << "  },\n";
        
        file << "  \"summary\": {\n";
        file << "    \"throughput_fps\":  " << std::fixed << std:: setprecision(2) << summary.throughput_fps << ",\n";
        file << "    \"latency_mean_ms\": " << summary.latency.mean_ms << ",\n";
        file << "    \"latency_p50_ms\":  " << summary.latency.p50_ms << ",\n";
        file << "    \"latency_p90_ms\": " << summary.latency.p90_ms << ",\n";
        file << "    \"latency_p95_ms\": " << summary.latency.p95_ms << ",\n";
        file << "    \"latency_p99_ms\": " << summary.latency.p99_ms << ",\n";
        file << "    \"latency_min_ms\": " << summary.latency.min_ms << ",\n";
        file << "    \"latency_max_ms\": " << summary.latency.max_ms << ",\n";
        file << "    \"latency_std_dev_ms\": " << std::setprecision(4) << summary.latency.std_dev_ms << ",\n";
        file << "    \"jitter_ms\":  " << std::setprecision(2) << summary.latency.jitter_ms << ",\n";
        file << "    \"variance_ratio\": " << summary.latency.variance_ratio << ",\n";
        file << "    \"max_rss_kb\": " << summary.max_rss_kb << "\n";
        file << "  },\n";
        
        file << "  \"num_iterations\": " << iterations_.size() << ",\n";
        file << "  \"iterations\":  [\n";
        
        for (size_t i = 0; i < iterations_.size(); ++i) {
            const auto& m = iterations_[i];
            file << "    {\"iter\": " << m.iteration;
            file << ", \"wall_ms\": " << std::setprecision(3) << m.wall_time_ms << "}";
            file << (i < iterations_.size() - 1 ? ",\n" : "\n");
        }
        
        file << "  ]\n}\n";
    }

private:
    std::vector<IterationMetrics> iterations_;
    std:: vector<double> latencies_ms_;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
void createDirectoryRecursive(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

long getMaxRSS() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return ru.ru_maxrss;
    }
    return 0;
}

// ============================================================================
// VERIFICATION RUN
// ============================================================================
bool verifyTensorRT(const std::string& model_path, PrecisionMode precision,
                    int batch_size, int channels, int height, int width) {
    std::cout << "\n[VERIFY] TensorRT " << precisionToString(precision) 
              << " with batch=" << batch_size << "...\n";
    
    try {
        Ort:: Env env(ORT_LOGGING_LEVEL_WARNING, "Verify");
        Ort::SessionOptions session_options;
        
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        
        // Append TensorRT provider
        int device_id = 0;
        Ort:: ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device_id));
        
        Ort::Session session(env, model_path.c_str(), session_options);
        
        size_t input_size = batch_size * channels * height * width;
        std::vector<float> input_data(input_size, 0.5f);
        std::vector<int64_t> input_dims = {batch_size, channels, height, width};
        
        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        std::string output_name_str = session.GetOutputName(0, allocator);
        
        const char* input_names[] = {input_name_str.c_str()};
        const char* output_names[] = {output_name_str.c_str()};
        
        auto mem_info = Ort:: MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort:: Value::CreateTensor<float>(
            mem_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );
        
        auto start = std::chrono:: high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono:: duration<double, std::milli>(end - start).count();
        std::cout << "[VERIFY] OK - " << std::fixed << std:: setprecision(2) << ms << " ms\n";
        return true;
        
    } catch (const Ort:: Exception& e) {
        std::cerr << "[VERIFY] FAILED - " << e.what() << "\n";
        return false;
    } catch (...) {
        std::cerr << "[VERIFY] FAILED - Unknown error\n";
        return false;
    }
}

// ============================================================================
// SINGLE RUN PROFILER
// ============================================================================
bool profileSingleRun(const RunConfig& config, Ort:: Env& env) {
    std:: cout << "\n" << std::string(50, '=') << "\n";
    std:: cout << "Model:       " << config.model_name << "\n";
    std:: cout << "Precision:  " << precisionToString(config.precision) << "\n";
    std::cout << "Batch:       " << config.batch_size << "\n";
    std::cout << "Provider:   TensorRT\n";
    std::cout << std::string(50, '=') << "\n";
    
    if (! fileExists(config.model_path)) {
        std::cerr << "[ERROR] Model not found: " << config.model_path << "\n";
        return false;
    }
    
    createDirectoryRecursive(config.output_dir);
    
    std::string prec_str = precisionToString(config.precision);
    std::string run_name = "static_batch" + std::to_string(config.batch_size) + "_trt_" + prec_str;
    std::string metrics_path = config.output_dir + "/" + run_name + "_metrics.json";
    std::string log_path = config.output_dir + "/" + run_name + ".log";
    
    // Redirect logs to file
    int saved_stdout = dup(STDOUT_FILENO);
    int saved_stderr = dup(STDERR_FILENO);
    FILE* log_file = fopen(log_path.c_str(), "w");
    if (log_file) {
        int log_fd = fileno(log_file);
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
    }
    
    auto now = std::chrono::system_clock:: now();
    auto time_now = std::chrono::system_clock:: to_time_t(now);
    std::cout << "=== Log started:  " << std::ctime(&time_now);
    std::cout << "=== Config: " << config.model_name << " " << prec_str
              << " batch=" << config.batch_size << " TensorRT\n\n";
    
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetIntraOpNumThreads(4);
    session_options.SetInterOpNumThreads(4);
    
    bool success = false;
    
    try {
        // Append TensorRT provider
        int device_id = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device_id));
        std::cout << "[INFO] TensorRT Provider appended\n";
        
        std::cout << "[INFO] Loading model...\n";
        auto load_start = std:: chrono::high_resolution_clock::now();
        Ort::Session session(env, config.model_path.c_str(), session_options);
        auto load_end = std:: chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std:: milli>(load_end - load_start).count();
        std::cout << "[INFO] Model loaded in " << std::fixed << std:: setprecision(0) << load_ms << " ms\n";
        
        // Prepare input
        size_t input_size = config.batch_size * config.channels * config.height * config.width;
        std::vector<float> input_data(input_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& val : input_data) val = dis(gen);
        
        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        std::string output_name_str = session.GetOutputName(0, allocator);
        
        const char* input_names[] = {input_name_str.c_str()};
        const char* output_names[] = {output_name_str.c_str()};
        
        std::vector<int64_t> input_dims = {config.batch_size, config.channels, config.height, config.width};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );
        
        // Warmup
        std::cout << "[INFO] Warmup (" << WARMUP_RUNS << " runs)...\n";
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        }
        std::cout << "[INFO] Warmup complete\n";
        
        // Benchmark
        MetricsRecorder metrics_recorder;
        
        std::cout << "[INFO] Benchmarking (" << BENCHMARK_RUNS << " runs)...\n";
        
        auto benchmark_start = std::chrono::high_resolution_clock:: now();
        
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            auto t0 = std:: chrono::high_resolution_clock::now();
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            metrics_recorder.record(i, wall_ms);
            
            if ((i + 1) % 25 == 0) {
                std:: cout << "[INFO] Progress: " << (i + 1) << "/" << BENCHMARK_RUNS << "\n";
            }
        }
        
        auto benchmark_end = std::chrono::high_resolution_clock::now();
        double total_time_sec = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
        
        // Compute summary
        RunSummary summary;
        summary.latency = metrics_recorder.computeLatencyStats();
        summary.throughput_fps = BENCHMARK_RUNS / total_time_sec;
        summary.max_rss_kb = getMaxRSS();
        
        // Save results
        std::cout << "\n[INFO] Saving results...\n";
        metrics_recorder.saveToJson(metrics_path, config, summary);
        
        std::cout << "\n[RESULTS]\n";
        std::cout << "  Throughput:       " << std::fixed << std::setprecision(2) << summary.throughput_fps << " FPS\n";
        std::cout << "  Latency (mean):  " << summary.latency.mean_ms << " ms\n";
        std::cout << "  Latency (P50):   " << summary.latency.p50_ms << " ms\n";
        std::cout << "  Latency (P90):   " << summary.latency.p90_ms << " ms\n";
        std::cout << "  Latency (P95):   " << summary.latency.p95_ms << " ms\n";
        std::cout << "  Latency (P99):   " << summary.latency.p99_ms << " ms\n";
        std::cout << "  Latency (min):   " << summary.latency.min_ms << " ms\n";
        std:: cout << "  Latency (max):   " << summary.latency.max_ms << " ms\n";
        std::cout << "  Jitter:           " << summary.latency.jitter_ms << " ms\n";
        std::cout << "  Variance ratio:   " << summary.latency.variance_ratio << "x\n";
        std::cout << "  Max RSS:         " << summary.max_rss_kb << " KB\n";
        
        std::cout << "\n[OUTPUT FILES]\n";
        std::cout << "  Metrics:  " << metrics_path << "\n";
        std::cout << "  Log:      " << log_path << "\n";
        
        success = true;
        
    } catch (const Ort:: Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime:  " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
    }
    
    // Restore stdout/stderr
    fflush(stdout);
    fflush(stderr);
    dup2(saved_stdout, STDOUT_FILENO);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stdout);
    close(saved_stderr);
    if (log_file) fclose(log_file);
    
    return success;
}

// ============================================================================
// GENERATE MASTER INDEX
// ============================================================================
void generateMasterIndex(const std:: vector<RunConfig>& completed_runs) {
    std::string index_path = OUTPUT_BASE_DIR + "/index.json";
    std:: ofstream file(index_path);
    
    file << "{\n";
    file << "  \"profiling_config\": {\n";
    file << "    \"warmup_runs\":  " << WARMUP_RUNS << ",\n";
    file << "    \"benchmark_runs\":  " << BENCHMARK_RUNS << ",\n";
    file << "    \"execution_provider\": \"tensorrt\"\n";
    file << "  },\n";
    file << "  \"runs\": [\n";
    
    for (size_t i = 0; i < completed_runs.size(); ++i) {
        const auto& c = completed_runs[i];
        std::string prec_str = precisionToString(c.precision);
        std::string run_name = "static_batch" + std::to_string(c.batch_size) + "_trt_" + prec_str;
        
        file << "    {\n";
        file << "      \"model\": \"" << c.model_name << "\",\n";
        file << "      \"precision\":  \"" << prec_str << "\",\n";
        file << "      \"batch_size\": " << c.batch_size << ",\n";
        file << "      \"provider\":  \"tensorrt\",\n";
        file << "      \"files\": {\n";
        file << "        \"metrics\": \"" << c.model_name << "/" << run_name << "_metrics.json\",\n";
        file << "        \"log\": \"" << c.model_name << "/" << run_name << ".log\"\n";
        file << "      }\n";
        file << "    }" << (i < completed_runs.size() - 1 ?  "," : "") << "\n";
    }
    
    file << "  ]\n}\n";
    
    std::cout << "Master index saved:  " << index_path << "\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     JETSON NANO TENSORRT MULTI-MODEL PROFILER          ║\n";
    std::cout << "║              (FP32 & FP16 Precision)                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    mkdir(OUTPUT_BASE_DIR.c_str(), 0755);
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "JetsonTRTProfiler");
    
    std::map<std::string, std::vector<PrecisionMode>> available_precisions;
    
    // ========================================
    // PHASE 1: Verify TensorRT for all models
    // ========================================
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "PHASE 1: Verifying TensorRT Provider\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    for (const auto& model : MODELS) {
        std::cout << "\nModel: " << model.name << "\n";
        available_precisions[model.name] = {};
        
        for (auto precision :  PRECISION_MODES) {
            if (1) {
                available_precisions[model.name].push_back(precision);
            }
        }
        
        std::cout << "Available precisions: ";
        for (auto pm : available_precisions[model.name]) {
            std:: cout << precisionToString(pm) << " ";
        }
        std::cout << "\n";
    }
    
    // ========================================
    // PHASE 2: Run benchmarks
    // ========================================
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PHASE 2: Running Benchmarks\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    std::vector<RunConfig> completed_runs;
    int total_runs = 0;
    int successful_runs = 0;
    
    for (const auto& model : MODELS) {
        total_runs += available_precisions[model.name].size();
    }
    
    std::cout << "\nTotal runs planned: " << total_runs << "\n";
    
    int current_run = 0;
    
    for (const auto& model : MODELS) {
        std::cout << "\n########################################\n";
        std::cout << "# MODEL: " << model.name << "\n";
        std::cout << "########################################\n";
        
        for (auto precision : available_precisions[model.name]) {
            current_run++;
            std::cout << "\n[" << current_run << "/" << total_runs << "] ";
            std::cout << model.name << " static batch=1 TensorRT " 
                      << precisionToString(precision) << "\n";
            
            RunConfig config;
            config.model_name = model.name;
            config.model_path = model.static_path;
            config.batch_size = BATCH_SIZE;
            config.channels = model.channels;
            config.height = model.height;
            config.width = model.width;
            config.output_dir = OUTPUT_BASE_DIR + "/" + model.name;
            config.precision = precision;
            
            if (profileSingleRun(config, env)) {
                completed_runs.push_back(config);
                successful_runs++;
            }
            
            std::cout << "Cooling down (" << COOLDOWN_SECONDS << "s)...\n";
            std::this_thread::sleep_for(std::chrono::seconds(COOLDOWN_SECONDS));
        }
    }
    
    // ========================================
    // PHASE 3: Generate index
    // ========================================
    std:: cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PHASE 3: Generating Master Index\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    generateMasterIndex(completed_runs);
    
    // ========================================
    // Summary
    // ========================================
    std:: cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PROFILING COMPLETE\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "Successful runs: " << successful_runs << "/" << total_runs << "\n";
    std::cout << "Output directory: " << OUTPUT_BASE_DIR << "/\n";
    std::cout << "\nGenerated files per run:\n";
    std::cout << "  - *_metrics.json (latency, throughput, etc.)\n";
    std::cout << "  - *.log          (execution logs for debugging)\n";
    std::cout << "\nMaster index:  " << OUTPUT_BASE_DIR << "/index.json\n";
    
    return (successful_runs == total_runs) ? 0 : 1;
}