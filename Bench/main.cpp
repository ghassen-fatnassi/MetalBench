#include <onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4); // CPU threads
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::string model_path = "/mnt/USEFUL/pro/GetCracked/REPOs/MetalBench/Models/yolo12n.onnx";
    Ort::Session session(env, model_path.c_str(), options);

    // Prepare input tensor
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto &d : input_shape) if (d <= 0) d = 1;

    size_t tensor_size = 1;
    for (auto d : input_shape) tensor_size *= d;

    std::vector<float> input_data(tensor_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto &x : input_data) x = dist(gen);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), tensor_size, input_shape.data(), input_shape.size()
    );

    // Input/output names
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::string> output_names = session.GetOutputNames();
    std::vector<const char*> input_names_c, output_names_c;
    for (auto &s : input_names) input_names_c.push_back(s.c_str());
    for (auto &s : output_names) output_names_c.push_back(s.c_str());

    // Benchmark loop
    const int num_runs = 100;
    std::vector<double> times;
    times.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(), &input_tensor, 1,
            output_names_c.data(), output_names_c.size()
        );
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = t2 - t1;
        times.push_back(duration.count());
    }

    // Compute average latency
    double sum = 0.0;
    for (auto t : times) sum += t;
    double avg_latency = sum / times.size();
    std::cout << "Average latency over " << num_runs << " runs: " << avg_latency << " ms" << std::endl;
    std::cout << "Throughput: " << 1000.0 / avg_latency << " FPS" << std::endl;

    return 0;
}
