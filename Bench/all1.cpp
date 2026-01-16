#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <regex>
#include <cstdio>
#include <memory>
#include <array>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <fcntl.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

// ============================================================================
// EXECUTION PROVIDER ENUM
// ============================================================================
enum class ExecutionProvider { CPU, CUDA };

std::string epToString(ExecutionProvider ep) {
    return (ep == ExecutionProvider::CPU) ? "cpu" : "cuda";
}

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================
struct ModelConfig {
    std::string name;
    std:: string static_path;
    std:: string dynamic_path;
    int channels;
    int height;
    int width;
    std::vector<int> batch_sizes;
};

const std::vector<ModelConfig> MODELS = {
    {
        "yolov3",
        "Models/yolo/YOLOV3_static.onnx",
        "Models/yolo/YOLOV3_dynamic.onnx",
        3, 416, 416,
        {1, 2, 4}
    },
    {
        "unet",
        "Models/unet/UNET_static.onnx",
        "Models/unet/UNET_dynamic.onnx",
        3, 512, 512,
        {1, 2, 4, 8}
    },
    {
        "mobilenetv2",
        "Models/mobilenetv2/MobileNetV2_static.onnx",
        "Models/mobilenetv2/MobileNetV2_dynamic.onnx",
        3, 224, 224,
        {1, 2, 4, 8}
    }
};

const std::vector<ExecutionProvider> EXECUTION_PROVIDERS = {
    ExecutionProvider:: CUDA,
    ExecutionProvider::CPU
};

// ============================================================================
// PROFILING CONFIGURATION
// ============================================================================
const int WARMUP_RUNS = 10;
const int BENCHMARK_RUNS = 100;
const int TEGRASTATS_INTERVAL_MS = 100;
const int COOLDOWN_SECONDS = 0.5;
const std::string OUTPUT_BASE_DIR = "profiling_results";

// Power sensor paths for Jetson Nano
const std:: string POWER_GPU_PATH = "/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio: device0/in_power0_input";
const std::string POWER_CPU_PATH = "/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0/in_power1_input";
const std::string POWER_TOTAL_PATH = "/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0/in_power2_input";

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct LatencyStats {
    double min_ms = 0, max_ms = 0, mean_ms = 0, std_dev_ms = 0;
    double p50_ms = 0, p90_ms = 0, p95_ms = 0, p99_ms = 0;
    double jitter_ms = 0;
    double variance_ratio = 0;
};

struct JetsonSample {
    double timestamp;
    int ram_used_mb, ram_total_mb;
    std::vector<int> cpu_utilization;
    int gpu_utilization;
    float temp_cpu, temp_gpu;
    int power_gpu_mw, power_cpu_mw, power_total_mw;
};

struct IterationMetrics {
    int iteration;
    double wall_time_ms;
    double cpu_time_ms;
};

struct RunConfig {
    std:: string model_name;
    std:: string model_path;
    std::string variant;
    int batch_size;
    int channels, height, width;
    std::string output_dir;
    ExecutionProvider exec_provider;
};

struct RunSummary {
    LatencyStats latency;
    double throughput_fps;
    double avg_cpu_util;
    double avg_gpu_util;
    double avg_power_mw;
    double avg_temp_cpu;
    double avg_temp_gpu;
    double energy_per_inference_mj;
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
        stats.std_dev_ms = std:: sqrt(sq_sum / n);
        
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
// JETSON HARDWARE PROFILER
// ============================================================================
class JetsonProfiler {
public:
    JetsonProfiler(int interval_ms = 100) : interval_ms_(interval_ms), running_(false) {}
    ~JetsonProfiler() { stop(); }
    
    void start() {
        samples_.clear();
        running_ = true;
        monitor_thread_ = std::thread(&JetsonProfiler::monitorLoop, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) monitor_thread_.join();
    }
    
    void saveToJson(const std::string& filepath) {
        std::lock_guard<std:: mutex> lock(samples_mutex_);
        std::ofstream file(filepath);
        
        file << "{\n  \"interval_ms\": " << interval_ms_;
        file << ",\n  \"num_samples\": " << samples_.size();
        file << ",\n  \"samples\": [\n";
        
        for (size_t i = 0; i < samples_.size(); ++i) {
            const auto& s = samples_[i];
            file << "    {";
            file << "\"timestamp\": " << std::fixed << std:: setprecision(3) << s.timestamp;
            file << ", \"ram_used_mb\": " << s.ram_used_mb;
            file << ", \"cpu_util\": [";
            for (size_t j = 0; j < s.cpu_utilization.size(); ++j) {
                file << s.cpu_utilization[j] << (j < s.cpu_utilization.size() - 1 ? "," : "");
            }
            file << "], \"gpu_util\": " << s.gpu_utilization;
            file << ", \"temp_cpu\": " << std::setprecision(1) << s.temp_cpu;
            file << ", \"temp_gpu\": " << s.temp_gpu;
            file << ", \"power_total_mw\":  " << s.power_total_mw;
            file << "}" << (i < samples_.size() - 1 ? ",\n" : "\n");
        }
        file << "  ]\n}\n";
    }
    
    void computeAverages(double& avg_cpu, double& avg_gpu, double& avg_power,
                         double& avg_temp_cpu, double& avg_temp_gpu) {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        if (samples_.empty()) {
            avg_cpu = avg_gpu = avg_power = avg_temp_cpu = avg_temp_gpu = 0;
            return;
        }
        
        double sum_cpu = 0, sum_gpu = 0, sum_power = 0, sum_tcpu = 0, sum_tgpu = 0;
        for (const auto& s : samples_) {
            if (! s.cpu_utilization.empty()) {
                double core_avg = std::accumulate(s.cpu_utilization.begin(),
                                                  s.cpu_utilization.end(), 0.0) / s.cpu_utilization.size();
                sum_cpu += core_avg;
            }
            sum_gpu += s.gpu_utilization;
            sum_power += s.power_total_mw;
            sum_tcpu += s.temp_cpu;
            sum_tgpu += s.temp_gpu;
        }
        
        size_t n = samples_.size();
        avg_cpu = sum_cpu / n;
        avg_gpu = sum_gpu / n;
        avg_power = sum_power / n;
        avg_temp_cpu = sum_tcpu / n;
        avg_temp_gpu = sum_tgpu / n;
    }

private:
    int interval_ms_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    std::vector<JetsonSample> samples_;
    std::mutex samples_mutex_;
    
    int readPowerSensor(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return 0;
        int value = 0;
        file >> value;
        return value;
    }
    
    double getTimestamp() {
        auto now = std:: chrono::steady_clock::now();
        return std::chrono:: duration<double>(now.time_since_epoch()).count();
    }
    
    JetsonSample parseTegrastats(const std::string& line) {
        JetsonSample s = {};
        s.timestamp = getTimestamp();
        
        std::smatch match;
        
        std::regex ram_regex(R"(RAM (\d+)/(\d+)MB)");
        if (std::regex_search(line, match, ram_regex)) {
            s.ram_used_mb = std::stoi(match[1]);
            s.ram_total_mb = std::stoi(match[2]);
        }
        
        std::regex cpu_regex(R"(CPU \[([\d%@,]+)\])");
        if (std::regex_search(line, match, cpu_regex)) {
            std::string cpu_str = match[1];
            std::regex core_regex(R"((\d+)%@\d+)");
            std::sregex_iterator iter(cpu_str.begin(), cpu_str.end(), core_regex);
            while (iter != std:: sregex_iterator()) {
                s.cpu_utilization.push_back(std::stoi((*iter)[1]));
                ++iter;
            }
        }
        
        std:: regex gpu_regex(R"(GR3D_FREQ (\d+)%)");
        if (std::regex_search(line, match, gpu_regex)) {
            s.gpu_utilization = std::stoi(match[1]);
        }
        
        std::regex temp_cpu_regex(R"(CPU@([\d.]+)C)");
        std::regex temp_gpu_regex(R"(GPU@([\d.]+)C)");
        if (std::regex_search(line, match, temp_cpu_regex)) s.temp_cpu = std::stof(match[1]);
        if (std::regex_search(line, match, temp_gpu_regex)) s.temp_gpu = std::stof(match[1]);
        
        s.power_gpu_mw = readPowerSensor(POWER_GPU_PATH);
        s.power_cpu_mw = readPowerSensor(POWER_CPU_PATH);
        s.power_total_mw = readPowerSensor(POWER_TOTAL_PATH);
        
        return s;
    }
    
    void monitorLoop() {
        std::string cmd = "tegrastats --interval " + std::to_string(interval_ms_);
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) return;
        
        std::array<char, 512> buffer;
        while (running_ && fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            JetsonSample sample = parseTegrastats(std::string(buffer.data()));
            std::lock_guard<std::mutex> lock(samples_mutex_);
            samples_.push_back(sample);
        }
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
    
    void record(int iteration, double wall_time_ms, double cpu_time_ms) {
        iterations_.push_back({iteration, wall_time_ms, cpu_time_ms});
        latencies_ms_.push_back(wall_time_ms);
    }
    
    LatencyStats computeLatencyStats() {
        return LatencyAnalyzer::calculate(latencies_ms_);
    }
    
    void saveToJson(const std::string& filepath, const RunConfig& config, const RunSummary& summary) {
        std::ofstream file(filepath);
        
        file << "{\n";
        file << "  \"config\": {\n";
        file << "    \"model_name\": \"" << config.model_name << "\",\n";
        file << "    \"variant\": \"" << config.variant << "\",\n";
        file << "    \"batch_size\": " << config.batch_size << ",\n";
        file << "    \"execution_provider\": \"" << epToString(config.exec_provider) << "\",\n";
        file << "    \"input_shape\": [" << config.batch_size << ", " << config.channels
             << ", " << config.height << ", " << config.width << "]\n";
        file << "  },\n";
        
        file << "  \"summary\":  {\n";
        file << "    \"throughput_fps\": " << std::fixed << std:: setprecision(2) << summary.throughput_fps << ",\n";
        file << "    \"latency_mean_ms\": " << summary.latency.mean_ms << ",\n";
        file << "    \"latency_p99_ms\": " << summary.latency.p99_ms << ",\n";
        file << "    \"latency_min_ms\": " << summary.latency.min_ms << ",\n";
        file << "    \"latency_max_ms\": " << summary.latency.max_ms << ",\n";
        file << "    \"latency_std_dev_ms\": " << std::setprecision(4) << summary.latency.std_dev_ms << ",\n";
        file << "    \"jitter_ms\":  " << std::setprecision(2) << summary.latency.jitter_ms << ",\n";
        file << "    \"variance_ratio\": " << summary.latency.variance_ratio << ",\n";
        file << "    \"avg_cpu_util\": " << std:: setprecision(1) << summary.avg_cpu_util << ",\n";
        file << "    \"avg_gpu_util\": " << summary.avg_gpu_util << ",\n";
        file << "    \"avg_power_mw\": " << std:: setprecision(0) << summary.avg_power_mw << ",\n";
        file << "    \"avg_temp_cpu\": " << std::setprecision(1) << summary.avg_temp_cpu << ",\n";
        file << "    \"avg_temp_gpu\": " << summary.avg_temp_gpu << ",\n";
        file << "    \"energy_per_inference_mj\": " << std::setprecision(2) << summary.energy_per_inference_mj << ",\n";
        file << "    \"max_rss_kb\": " << summary.max_rss_kb << "\n";
        file << "  },\n";
        
        file << "  \"num_iterations\": " << iterations_.size() << ",\n";
        file << "  \"iterations\": [\n";
        
        for (size_t i = 0; i < iterations_.size(); ++i) {
            const auto& m = iterations_[i];
            file << "    {\"iter\": " << m.iteration;
            file << ", \"wall_ms\": " << std::setprecision(3) << m.wall_time_ms;
            file << ", \"cpu_ms\": " << m.cpu_time_ms << "}";
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
    std:: ifstream f(path);
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
// EXECUTION PROVIDER SETUP (ONNX Runtime 1.6 compatible)
// ============================================================================
bool setupExecutionProvider(Ort::SessionOptions& options, ExecutionProvider ep) {
    if (ep == ExecutionProvider::CPU) {
        std::cout << "[INFO] Using CPU Execution Provider\n";
        options.SetIntraOpNumThreads(4);
        options.SetInterOpNumThreads(1);
        return true;
    }
    
    // CUDA - ORT 1.6 API
    try {
        OrtCUDAProviderOptions cuda_options;
        memset(&cuda_options, 0, sizeof(cuda_options));
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = 1;
        // Note: cudnn_conv_algo_search uses enum value directly in 1.6
        // 0 = EXHAUSTIVE, 1 = HEURISTIC, 2 = DEFAULT
        cuda_options.cudnn_conv_algo_search = static_cast<OrtCudnnConvAlgoSearch>(0);
        
        options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[INFO] CUDA Execution Provider appended\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[WARNING] CUDA provider failed: " << e.what() << "\n";
        return false;
    } catch (...) {
        std::cerr << "[WARNING] CUDA provider failed (unknown error)\n";
        return false;
    }
}

// ============================================================================
// VERIFICATION RUN (ONNX Runtime 1.6 compatible)
// ============================================================================
bool verifyProvider(const std::string& model_path, ExecutionProvider ep,
                    int batch_size, int channels, int height, int width) {
    std::cout << "\n[VERIFY] " << epToString(ep) << " with batch=" << batch_size << "...\n";
    
    try {
        Ort:: Env env(ORT_LOGGING_LEVEL_WARNING, "Verify");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (! setupExecutionProvider(options, ep)) {
            std::cout << "[VERIFY] Provider setup failed\n";
            return false;
        }
        
        Ort::Session session(env, model_path.c_str(), options);
        
        size_t input_size = batch_size * channels * height * width;
        std::vector<float> input_data(input_size, 0.5f);
        std::vector<int64_t> input_dims = {batch_size, channels, height, width};
        
        // ORT 1.6 API:  GetInputName returns char* that needs to be freed
        Ort::AllocatorWithDefaultOptions allocator;
        char* input_name_ptr = session.GetInputName(0, allocator);
        char* output_name_ptr = session.GetOutputName(0, allocator);
        
        std::string input_name_str(input_name_ptr);
        std::string output_name_str(output_name_ptr);
        
        // Free allocated names
        allocator.Free(input_name_ptr);
        allocator.Free(output_name_ptr);
        
        const char* input_names[] = {input_name_str.c_str()};
        const char* output_names[] = {output_name_str.c_str()};
        
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value:: CreateTensor<float>(
            mem_info, input_data.data(), input_size, input_dims.data(), input_dims.size()
        );
        
        auto start = std::chrono:: high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[VERIFY] OK - " << std::fixed << std:: setprecision(2) << ms << " ms\n";
        return true;
        
    } catch (const std::exception& e) {
        std:: cerr << "[VERIFY] FAILED - " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// SINGLE RUN PROFILER (ONNX Runtime 1.6 compatible)
// ============================================================================
bool profileSingleRun(const RunConfig& config, Ort:: Env& env) {
    std:: cout << "\n" << std::string(50, '=') << "\n";
    std:: cout << "Model:      " << config.model_name << "\n";
    std::cout << "Variant:   " << config.variant << "\n";
    std:: cout << "Batch:      " << config.batch_size << "\n";
    std::cout << "Provider:   " << epToString(config.exec_provider) << "\n";
    std::cout << std::string(50, '=') << "\n";
    
    if (! fileExists(config.model_path)) {
        std::cerr << "[ERROR] Model not found: " << config.model_path << "\n";
        return false;
    }
    
    createDirectoryRecursive(config.output_dir);
    
    std::string ep_str = epToString(config.exec_provider);
    std::string run_name = config.variant + "_batch" + std::to_string(config.batch_size) + "_" + ep_str;
    std::string onnx_profile_prefix = config.output_dir + "/" + run_name + "_onnx";
    std::string tegrastats_path = config.output_dir + "/" + run_name + "_tegrastats.json";
    std::string metrics_path = config.output_dir + "/" + run_name + "_metrics.json";
    std::string log_path = config.output_dir + "/" + run_name + ".log";
    
    // Redirect logs
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
    std::cout << "=== Config: " << config.model_name << " " << config.variant
              << " batch=" << config.batch_size << " " << ep_str << "\n\n";
    
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.EnableProfiling(onnx_profile_prefix.c_str());
    session_options.SetLogSeverityLevel(1);
    
    bool success = false;
    
    try {
        if (!setupExecutionProvider(session_options, config.exec_provider)) {
            throw std::runtime_error("Failed to setup execution provider");
        }
        
        std::cout << "[INFO] Loading model...\n";
        auto load_start = std:: chrono::high_resolution_clock::now();
        Ort::Session session(env, config.model_path.c_str(), session_options);
        auto load_end = std::chrono::high_resolution_clock:: now();
        double load_ms = std::chrono::duration<double, std:: milli>(load_end - load_start).count();
        std::cout << "[INFO] Model loaded in " << std::fixed << std:: setprecision(0) << load_ms << " ms\n";
        
        // Prepare input
        size_t input_size = config.batch_size * config.channels * config.height * config.width;
        std::vector<float> input_data(input_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& val : input_data) val = dis(gen);
        
        // ORT 1.6 API
        Ort::AllocatorWithDefaultOptions allocator;
        char* input_name_ptr = session.GetInputName(0, allocator);
        char* output_name_ptr = session.GetOutputName(0, allocator);
        
        std::string input_name_str(input_name_ptr);
        std::string output_name_str(output_name_ptr);
        
        allocator.Free(input_name_ptr);
        allocator.Free(output_name_ptr);
        
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
        
        // Start profilers
        JetsonProfiler jetson_profiler(TEGRASTATS_INTERVAL_MS);
        MetricsRecorder metrics_recorder;
        
        std::cout << "[INFO] Benchmarking (" << BENCHMARK_RUNS << " runs)...\n";
        jetson_profiler.start();
        
        auto benchmark_start = std:: chrono::high_resolution_clock::now();
        
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            struct timespec cpu_start, cpu_end;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            session.Run(Ort:: RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            auto t1 = std::chrono::high_resolution_clock:: now();
            
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
            
            double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                           (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;
            
            metrics_recorder.record(i, wall_ms, cpu_ms);
            
            if ((i + 1) % 25 == 0) {
                std:: cout << "[INFO] Progress: " << (i + 1) << "/" << BENCHMARK_RUNS << "\n";
            }
        }
        
        auto benchmark_end = std::chrono::high_resolution_clock:: now();
        double total_time_sec = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
        
        jetson_profiler.stop();
        
        // Compute summary
        RunSummary summary;
        summary.latency = metrics_recorder.computeLatencyStats();
        summary.throughput_fps = BENCHMARK_RUNS / total_time_sec;
        
        jetson_profiler.computeAverages(
            summary.avg_cpu_util, summary.avg_gpu_util, summary.avg_power_mw,
            summary.avg_temp_cpu, summary.avg_temp_gpu
        );
        
        summary.energy_per_inference_mj = (summary.avg_power_mw / 1000.0) *
                                          (summary.latency.mean_ms / 1000.0) * 1000.0;
        summary.max_rss_kb = getMaxRSS();
        
        // End ONNX profiling (ORT 1.6 API)
        std::string onnx_profile_path = session.EndProfiling(allocator);
        
        // Save results
        std::cout << "\n[INFO] Saving results...\n";
        jetson_profiler.saveToJson(tegrastats_path);
        metrics_recorder.saveToJson(metrics_path, config, summary);
        
        std::cout << "\n[RESULTS]\n";
        std::cout << "  Throughput:       " << std::fixed << std:: setprecision(2) << summary.throughput_fps << " FPS\n";
        std::cout << "  Latency (mean):  " << summary.latency.mean_ms << " ms\n";
        std::cout << "  Latency (P99):   " << summary.latency.p99_ms << " ms\n";
        std::cout << "  Jitter:           " << summary.latency.jitter_ms << " ms\n";
        std::cout << "  Variance ratio:  " << summary.latency.variance_ratio << "x\n";
        std::cout << "  Power (avg):     " << std::setprecision(0) << summary.avg_power_mw << " mW\n";
        std::cout << "  Energy/inf:      " << std::setprecision(2) << summary.energy_per_inference_mj << " mJ\n";
        std::cout << "  CPU util:        " << std::setprecision(1) << summary.avg_cpu_util << " %\n";
        std::cout << "  GPU util:         " << summary.avg_gpu_util << " %\n";
        
        std::cout << "\n[OUTPUT FILES]\n";
        std::cout << "  Metrics:     " << metrics_path << "\n";
        std:: cout << "  Tegrastats:  " << tegrastats_path << "\n";
        std::cout << "  ONNX prof:   " << onnx_profile_path << "\n";
        std::cout << "  Log:         " << log_path << "\n";
        
        success = true;
        
    } catch (const Ort::Exception& e) {
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
void generateMasterIndex(const std::vector<RunConfig>& completed_runs) {
    std::string index_path = OUTPUT_BASE_DIR + "/index.json";
    std::ofstream file(index_path);
    
    file << "{\n";
    file << "  \"profiling_config\": {\n";
    file << "    \"warmup_runs\":  " << WARMUP_RUNS << ",\n";
    file << "    \"benchmark_runs\":  " << BENCHMARK_RUNS << ",\n";
    file << "    \"tegrastats_interval_ms\": " << TEGRASTATS_INTERVAL_MS << "\n";
    file << "  },\n";
    file << "  \"runs\": [\n";
    
    for (size_t i = 0; i < completed_runs.size(); ++i) {
        const auto& c = completed_runs[i];
        std::string ep_str = epToString(c.exec_provider);
        std::string run_name = c.variant + "_batch" + std::to_string(c.batch_size) + "_" + ep_str;
        
        file << "    {\n";
        file << "      \"model\": \"" << c.model_name << "\",\n";
        file << "      \"variant\":  \"" << c.variant << "\",\n";
        file << "      \"batch_size\": " << c.batch_size << ",\n";
        file << "      \"provider\": \"" << ep_str << "\",\n";
        file << "      \"files\": {\n";
        file << "        \"metrics\": \"" << c.model_name << "/" << run_name << "_metrics.json\",\n";
        file << "        \"tegrastats\": \"" << c.model_name << "/" << run_name << "_tegrastats.json\",\n";
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
    std:: cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     JETSON NANO ONNX RUNTIME MULTI-MODEL PROFILER      ║\n";
    std::cout << "║                   (ORT 1.6 Compatible)                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    mkdir(OUTPUT_BASE_DIR.c_str(), 0755);
    
    Ort:: Env env(ORT_LOGGING_LEVEL_WARNING, "JetsonProfiler");
    
    std::map<std::string, std::vector<ExecutionProvider>> available_providers;
    
    // ========================================
    // PHASE 1: Verify all providers
    // ========================================
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "PHASE 1: Verifying Execution Providers\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    for (const auto& model : MODELS) {
        std::cout << "\nModel: " << model.name << "\n";
        available_providers[model.name] = {};
        
        for (auto ep :  EXECUTION_PROVIDERS) {
            if (verifyProvider(model.static_path, ep, 1, model.channels, model.height, model.width)) {
                available_providers[model.name].push_back(ep);
            }
        }
        
        std::cout << "Available providers:  ";
        for (auto ep : available_providers[model.name]) {
            std::cout << epToString(ep) << " ";
        }
        std:: cout << "\n";
    }
    
    // ========================================
    // PHASE 2: Run benchmarks
    // ========================================
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PHASE 2: Running Benchmarks\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    std:: vector<RunConfig> completed_runs;
    int total_runs = 0;
    int successful_runs = 0;
    
    for (const auto& model : MODELS) {
        size_t n_providers = available_providers[model.name].size();
        total_runs += n_providers * (1 + model.batch_sizes.size());
    }
    
    std::cout << "\nTotal runs planned: " << total_runs << "\n";
    
    int current_run = 0;
    
    for (const auto& model : MODELS) {
        std::cout << "\n########################################\n";
        std::cout << "# MODEL: " << model.name << "\n";
        std::cout << "########################################\n";
        
        for (auto ep : available_providers[model.name]) {
            // Static model (batch 1)
            {
                current_run++;
                std::cout << "\n[" << current_run << "/" << total_runs << "] ";
                std::cout << model.name << " static batch=1 " << epToString(ep) << "\n";
                
                RunConfig config;
                config.model_name = model.name;
                config.model_path = model.static_path;
                config.variant = "static";
                config.batch_size = 1;
                config.channels = model.channels;
                config.height = model.height;
                config.width = model.width;
                config.output_dir = OUTPUT_BASE_DIR + "/" + model.name;
                config.exec_provider = ep;
                
                if (profileSingleRun(config, env)) {
                    completed_runs.push_back(config);
                    successful_runs++;
                }
                
                std::cout << "Cooling down (" << COOLDOWN_SECONDS << "s)...\n";
                std::this_thread::sleep_for(std::chrono::seconds(COOLDOWN_SECONDS));
            }
            
            // Dynamic model (various batch sizes)
            for (int batch_size : model.batch_sizes) {
                current_run++;
                std::cout << "\n[" << current_run << "/" << total_runs << "] ";
                std::cout << model.name << " dynamic batch=" << batch_size << " " << epToString(ep) << "\n";
                
                RunConfig config;
                config.model_name = model.name;
                config.model_path = model.dynamic_path;
                config.variant = "dynamic";
                config.batch_size = batch_size;
                config.channels = model.channels;
                config.height = model.height;
                config.width = model.width;
                config.output_dir = OUTPUT_BASE_DIR + "/" + model.name;
                config.exec_provider = ep;
                
                if (profileSingleRun(config, env)) {
                    completed_runs.push_back(config);
                    successful_runs++;
                }
                
                std:: cout << "Cooling down (" << COOLDOWN_SECONDS << "s)...\n";
                std::this_thread::sleep_for(std::chrono:: seconds(COOLDOWN_SECONDS));
            }
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
    std::cout << "  - *_metrics.json    (latency, throughput, power, etc.)\n";
    std::cout << "  - *_tegrastats.json (hardware metrics time series)\n";
    std::cout << "  - *_onnx_*.json     (ONNX operator profiling)\n";
    std::cout << "  - *.log             (execution logs for debugging)\n";
    std::cout << "\nMaster index:  " << OUTPUT_BASE_DIR << "/index.json\n";
    
    return (successful_runs == total_runs) ? 0 : 1;
}