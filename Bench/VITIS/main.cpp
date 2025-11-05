#include <onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <unordered_map>

int main() {
    // ----------------------------
    // 1️⃣ Initialize ONNX Runtime
    // ----------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vitis_benchmark");

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.EnableProfiling("vitisai_profile.json");

    // ----------------------------
    // 2️⃣ Append Vitis AI Execution Provider
    // ----------------------------
    std::unordered_map<std::string, std::string> vitis_options;
    vitis_options["cache_dir"] = "/tmp/vitisai_cache";   // cache folder
    vitis_options["cache_key"] = "yolo12n";              // your model identifier
    vitis_options["log_level"] = "info";

    options.AppendExecutionProvider_VitisAI(vitis_options);

    std::string model_path = "../Models/yolo12n.onnx";
    Ort::Session session(env, model_path.c_str(), options);

    // ----------------------------
    // 3️⃣ Query input/output names and shapes
    // ----------------------------
    Ort::AllocatorWithDefaultOptions allocator;
    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();

    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    for (size_t i = 0; i < input_count; ++i) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes.emplace_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    std::vector<std::string> output_names;
    for (size_t i = 0; i < output_count; ++i)
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());

    std::vector<const char*> input_names_c, output_names_c;
    for (auto &s : input_names) input_names_c.push_back(s.c_str());
    for (auto &s : output_names) output_names_c.push_back(s.c_str());

    // ----------------------------
    // 4️⃣ Prepare dummy input tensor
    // ----------------------------
    auto input_shape = input_shapes[0];
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

    // ----------------------------
    // 5️⃣ Warm-up runs
    // ----------------------------
    const int warmup_runs = 5;
    for (int i = 0; i < warmup_runs; ++i) {
        session.Run(Ort::RunOptions{nullptr},
                    input_names_c.data(), &input_tensor, 1,
                    output_names_c.data(), output_names_c.size());
    }

    // ----------------------------
    // 6️⃣ Benchmark loop
    // ----------------------------
    const int num_runs = 50;
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

    // ----------------------------
    // 7️⃣ Compute statistics
    // ----------------------------
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (auto t : times) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    double avg_latency = sum / times.size();

    std::cout << "✅ Vitis AI Benchmark (" << num_runs << " runs)\n";
    std::cout << "Average latency: " << avg_latency << " ms\n";
    std::cout << "Min latency: " << min_time << " ms\n";
    std::cout << "Max latency: " << max_time << " ms\n";
    std::cout << "Throughput: " << 1000.0 / avg_latency << " FPS\n";
    std::cout << "Profiling file: vitisai_profile.json\n";

    return 0;
}
