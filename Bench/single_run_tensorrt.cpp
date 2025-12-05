// single_run_tensorrt.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

using namespace std;
using hr_clock = std::chrono::high_resolution_clock;

const char* MODEL_PATH = "Models/yolo12n_op12.onnx";
const int IMG_C = 3;
const int IMG_H = 128;
const int IMG_W = 128;
const int BATCH_SIZE = 1;

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

void safe_mkdir(const std::string &p) {
    mkdir(p.c_str(), 0755); // ignore errors if exists
}

Ort::Session create_tensorrt_session(Ort::Env& env, const char* model_path) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);
    so.SetInterOpNumThreads(4);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    // Create caching directory for TRT engines
    safe_mkdir("./trt_engine_cache");

    // Append TensorRT EP then CUDA EP
    OrtSessionOptions* raw = so;
    // Note: Append function arguments vary by ORT version / build.
    // Typical signatures (C API):
    //  OrtSessionOptionsAppendExecutionProvider_Tensorrt(OrtSessionOptions* options, int device_id, size_t max_workspace_size_bytes, bool fp16_enable);
    //  OrtSessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions* options, int device_id);
    // Below we call common prototypes — adjust if your ORT build uses different signatures.
    // Many Jetson builds expose: OrtSessionOptionsAppendExecutionProvider_Tensorrt(raw, 0, (size_t)(1<<28), true);
    // If your header doesn't match, replace with the proper call from your build.
    #if 1
    // Try to append TRT - this may require you to link provider libs (done in CMake)
    // If your build doesn't expose this symbol exactly, you'll need to adapt the call.
    // Example prototype variations exist — check your onnxruntime_c_api.h
    extern OrtStatus* OrtSessionOptionsAppendExecutionProvider_Tensorrt(OrtSessionOptions* options, int device_id, size_t trt_max_workspace_size, bool trt_fp16_enable);
    OrtSessionOptionsAppendExecutionProvider_Tensorrt(raw, 0, (size_t)(1 << 28), true);
    //OrtSessionOptionsAppendExecutionProvider_CUDA(raw, 0);
    #else
    // If not available, fallback to using CUDA EP only:
    OrtSessionOptionsAppendExecutionProvider_CUDA(raw, 0);
    #endif

    return Ort::Session(env, model_path, so);
}

int main() {
    if (!file_exists(MODEL_PATH)) {
        std::cerr << "Error: Model not found at " << MODEL_PATH << '\n';
        return 1;
    }
    cout << "Loading model: " << MODEL_PATH << '\n';

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "single_run_tensorrt");
    Ort::Session session = create_tensorrt_session(env, MODEL_PATH);

    vector<int64_t> input_shape = {BATCH_SIZE, IMG_C, IMG_H, IMG_W};
    vector<float> input_data = generate_input(BATCH_SIZE);

    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
    );

    char* input_name = session.GetInputName(0, allocator);
    vector<const char*> input_names = {input_name};
    vector<const char*> output_names;
    size_t out_count = session.GetOutputCount();
    for (size_t i = 0; i < out_count; ++i) {
        output_names.push_back(session.GetOutputName(i, allocator));
    }

    auto t0 = hr_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
    auto t1 = hr_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cout << "Single TensorRT inference completed in " << ms << " ms\n";

    return 0;
}
