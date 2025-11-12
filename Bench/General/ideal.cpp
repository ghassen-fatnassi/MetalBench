#include "build/onnxruntime-linux-x64-1.23.2/include/onnxruntime_cxx_api.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <cmath>
#include <cstring>
#include <sys/sysinfo.h>

using namespace std;
using clk = std::chrono::high_resolution_clock;

// Utility for timing
template <typename F>
double time_it(F&& func, int iters = 1) {
    auto t1 = clk::now();
    for (int i = 0; i < iters; ++i) func();
    auto t2 = clk::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
}

// --------------------------------------------
// 🔹 1. System Info Benchmark
// --------------------------------------------
void system_info() {
    cout << "===== SYSTEM INFO =====" << endl;
    cout << "CPU threads: " << std::thread::hardware_concurrency() << endl;

    struct sysinfo info;
    if (sysinfo(&info) == 0)
        cout << "Total RAM: " << (info.totalram / (1024 * 1024)) << " MB" << endl;

    cout << "=======================" << endl;
}

// --------------------------------------------
// 🔹 2. Cache Pressure Benchmark
// --------------------------------------------
void cache_pressure() {
    cout << "\n[Cache Pressure Test]" << endl;
    vector<size_t> sizes = {64 * 1024, 512 * 1024, 4 * 1024 * 1024, 32 * 1024 * 1024, 256 * 1024 * 1024};
    for (auto sz : sizes) {
        vector<float> a(sz / sizeof(float), 1.0f);
        double t = time_it([&]() {
            volatile float sum = 0;
            for (size_t i = 0; i < a.size(); i += 16)
                sum += a[i];
        }, 10);
        cout << "Working set " << (sz / 1024) << " KB -> " << t << " ms" << endl;
    }
}

// --------------------------------------------
// 🔹 3. Memory Bandwidth Test
// --------------------------------------------
void memory_bandwidth() {
    cout << "\n[Memory Bandwidth Test]" << endl;
    const size_t N = 100'000'000; // ~400 MB
    vector<float> a(N, 1.1f), b(N, 2.2f), c(N);
    double ms = time_it([&]() {
        for (size_t i = 0; i < N; ++i)
            c[i] = a[i] + b[i];
    }, 1);
    double bytes = 3.0 * N * sizeof(float);
    double gbps = (bytes / (ms / 1000.0)) / 1e9;
    cout << "Bandwidth: " << gbps << " GB/s" << endl;
}

// --------------------------------------------
// 🔹 4. ILP (Instruction-Level Parallelism)
// --------------------------------------------
void ilp_test() {
    cout << "\n[ILP / Pipeline Test]" << endl;
    const int N = 100'000'000;
    double dep_time = time_it([&]() {
        double x = 1.0;
        for (int i = 0; i < N; ++i)
            x = std::sin(x); // dependent chain
        volatile double sink = x;
    });

    double indep_time = time_it([&]() {
        double a = 1.0, b = 2.0, c = 3.0, d = 4.0;
        for (int i = 0; i < N / 4; ++i) {
            a = std::sin(a);
            b = std::sin(b);
            c = std::sin(c);
            d = std::sin(d);
        }
        volatile double sink = a + b + c + d;
    });

    cout << "Dependent ops: " << dep_time << " ms" << endl;
    cout << "Independent ops: " << indep_time << " ms" << endl;
    cout << "ILP Gain: " << dep_time / indep_time << "x" << endl;
}

// --------------------------------------------
// 🔹 5. TLB Pressure Test
// --------------------------------------------
void tlb_pressure() {
    cout << "\n[TLB Pressure Test]" << endl;
    const size_t N = 64 * 1024 * 1024; // 64 MB
    const size_t stride = 4096; // 4K stride hits new page each time
    vector<int> arr(N / sizeof(int), 1);
    double t = time_it([&]() {
        volatile int sum = 0;
        for (size_t i = 0; i < arr.size(); i += (stride / sizeof(int)))
            sum += arr[i];
    }, 5);
    cout << "Access time (4K stride over 64MB): " << t << " ms" << endl;
}

// --------------------------------------------
// 🔹 6. Contention / Synchronization Test
// --------------------------------------------
void contention_test() {
    cout << "\n[Contention / Mutex Test]" << endl;
    const int threads = std::min(8u, std::thread::hardware_concurrency());
    const int iters = 500000;
    std::mutex mtx;
    auto worker = [&]() {
        for (int i = 0; i < iters; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
        }
    };
    double t = time_it([&]() {
        vector<thread> pool;
        for (int i = 0; i < threads; ++i)
            pool.emplace_back(worker);
        for (auto &th : pool) th.join();
    });
    cout << "Lock contention (" << threads << " threads): " << t << " ms" << endl;
}

// --------------------------------------------
// 🔹 7. Working Set Growth
// --------------------------------------------
void working_set_test() {
    cout << "\n[Working Set Growth]" << endl;
    for (int mb = 1; mb <= 512; mb *= 2) {
        size_t n = (mb * 1024 * 1024) / sizeof(float);
        vector<float> data(n, 1.0f);
        double t = time_it([&]() {
            volatile float sum = 0;
            for (auto x : data) sum += x;
        }, 5);
        cout << mb << " MB -> " << t << " ms" << endl;
    }
}

// --------------------------------------------
// 🔹 8. ONNX Runtime Benchmark
// --------------------------------------------
void onnx_benchmark() {
    cout << "\n[ONNX Runtime Benchmark]" << endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.EnableProfiling("onnxruntime_profile.json");

    string model_path = "../Models/yolo12n.onnx";
    Ort::Session session(env, model_path.c_str(), options);

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto &d : input_shape) if (d <= 0) d = 1;

    size_t tensor_size = 1;
    for (auto d : input_shape) tensor_size *= d; 

    vector<float> input_data(tensor_size);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto &x : input_data) x = dist(gen);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), tensor_size, input_shape.data(), input_shape.size()
    );

    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();
    vector<const char*> in_c, out_c;
    for (auto &s : input_names) in_c.push_back(s.c_str());
    for (auto &s : output_names) out_c.push_back(s.c_str());

    // Warmup
    for (int i = 0; i < 3; ++i)
        session.Run(Ort::RunOptions{nullptr}, in_c.data(), &input_tensor, 1, out_c.data(), out_c.size());

    const int num_runs = 20;
    vector<double> times;
    for (int i = 0; i < num_runs; ++i) {
        auto t1 = clk::now();
        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_c.data(), &input_tensor, 1, out_c.data(), out_c.size());
        auto t2 = clk::now();
        times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    double avg = accumulate(times.begin(), times.end(), 0.0) / times.size();
    cout << "Average latency: " << avg << " ms" << endl;
    cout << "Throughput: " << 1000.0 / avg << " FPS" << endl;
}

// --------------------------------------------
// MAIN
// --------------------------------------------
int main() {
    cout << "🔥 METALBENCH Microarchitecture Benchmark 🔥\n";
    system_info();
    cache_pressure();
    memory_bandwidth();
    ilp_test();
    tlb_pressure();
    contention_test();
    working_set_test();
    onnx_benchmark();
    cout << "\n✅ Benchmarking complete.\n";
}
