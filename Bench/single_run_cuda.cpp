// single_run_cuda.cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>

using namespace std;
using clock_t = std::chrono::high_resolution_clock;

const char* MODEL_PATH = "Models/yolo12n_op12.onnx";
const int IMG_C = 3;
const int IMG_H = 128;
const int IMG_W = 128;
const int BATCH_SIZE = 1;

// Create deterministic input like your python script
static vector<float> generate_input(int batch) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    vector<float> data(batch * IMG_C * IMG_H * IMG_W);
    for (auto &v : data) v = dist(rng);
    return data;
}

bool file_exists(const string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

Ort::Session create_cuda_session(Ort::Env& env, const char* model_path) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);
    so.SetInterOpNumThreads(4);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    // Append CUDA EP: this symbol is provided when you build ORT with CUDA EP linked.
    // Signature (C API): OrtStatus* OrtSessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions* options, int device_id);
    // The C++ SessionOptions can be cast to OrtSessionOptions*
    OrtSessionOptions* raw = so;
    // device_id = 0
    OrtSessionOptionsAppendExecutionProvider_CUDA(raw, 0);

    // Create the session
    return Ort::Session(env, model_path, so);
}

int main() {
    if (!file_exists(MODEL_PATH)) {
        std::cerr << "Error: Model not found at " << MODEL_PATH << '\n';
        return 1;
    }
    cout << "Loading model: " << MODEL_PATH << '\n';

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "single_run_cuda");
    Ort::Session session = create_cuda_session(env, MODEL_PATH);

    // Generate input
    vector<int64_t> input_shape = {BATCH_SIZE, IMG_C, IMG_H, IMG_W};
    vector<float> input_data = generate_input(BATCH_SIZE);

    // Allocate tensor
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
    );

    // Get input name
    char* input_name = session.GetInputNameAllocated(0, allocator).release();
    vector<const char*> input_names = {input_name};
    vector<const char*> output_names;
    size_t out_count = session.GetOutputCount();
    for (size_t i = 0; i < out_count; ++i) {
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).release());
    }

    // Run inference
    auto t0 = clock_t::now();
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
    auto t1 = clock_t::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cout << "Single GPU (CUDA) inference completed in " << ms << " ms\n";

    // Free names (allocator will release automatically when going out of scope)
    // Note: if using GetInputNameAllocated you used allocator.release() above; freed on process exit or free manually.

    return 0;
}
