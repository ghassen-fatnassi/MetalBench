#include <onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

int main() {
    // ----------------------------
    // 1️⃣ Initialize ONNX Runtime with OpenVINO EP
    // ----------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark_openvino");
    Ort::SessionOptions options;

    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.EnableProfiling("onnxruntime_openvino_profile.json");

    // 🧠 Set OpenVINO Execution Provider
    // Choose device: "CPU_FP32", "GPU_FP32", "AUTO", "HETERO:CPU,GPU", etc.
    const char* device_id = "CPU_FP32";
    OrtOpenVINOProviderOptions openvino_options;
    openvino_options.device_type = device_id;
    openvino_options.enable_opencl_throttling = 0;
    openvino_options.num_of_threads = 4;
    openvino_options.infer_num_of_threads = 4;

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(options, &openvino_options));

    std::string model_path = "../Models/yolov12n.onnx";
    Ort::Session session(env, model_path.c_str(), options);

    // ----------------------------
    // 2️⃣ Prepare random input tensor
    // ----------------------------
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    for (auto &d : input_shape)
        if (d <= 0) d = 1;  // dynamic -> 1

    size_t tensor_size = 1;
    for (auto d : input_shape) tensor_size *= d;

    std::vector<float> input_data(tensor_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto &x : input_data)
        x = dist(gen);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), tensor_size,
        input_shape.data(), input_shape.size()
    );

    // Input/output names
    std::vector<std::string> input_names_str = session.GetInputNames();
    std::vector<std::string> output_names_str = session.GetOutputNames();
    std::vector<const char*> input_names, output_names;
    for (auto &s : input_names_str) input_names.push_back(s.c_str());
    for (auto &s : output_names_str) output_names.push_back(s.c_str());

    // ----------------------------
    // 3️⃣ Warm-up runs
    // ----------------------------
    const int warmup_runs = 5;
    for (int i = 0; i < warmup_runs; ++i) {
        session.Run(Ort::RunOptions{nullptr},
                    input_names.data(), &input_tensor, 1,
                    output_names.data(), output_names.size());
    }

    // ----------------------------
    // 4️⃣ Benchmark loop
    // ----------------------------
    const int num_runs = 50;
    std::vector<double> times;
    times.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   input_names.data(), &input_tensor, 1,
                                   output_names.data(), output_names.size());

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = t2 - t1;
        times.push_back(duration.count());
    }

    // ----------------------------
    // 5️⃣ Compute stats
    // ----------------------------
    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (auto t : times) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    double avg_latency = sum / times.size();

    std::cout << "Benchmark results (" << device_id << ") over " << num_runs << " runs:\n";
    std::cout << "Average latency: " << avg_latency << " ms\n";
    std::cout << "Min latency: " << min_time << " ms\n";
    std::cout << "Max latency: " << max_time << " ms\n";
    std::cout << "Throughput: " << 1000.0 / avg_latency << " FPS\n";
    std::cout << "Profiling saved to: onnxruntime_openvino_profile.json\n";

    return 0;
}
