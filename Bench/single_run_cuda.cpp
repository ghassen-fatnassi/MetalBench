#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>

// Required if using the specific C-API function for CUDA directly
// Ensure you link against onnxruntime_providers_cuda or onnxruntime
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h> 

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

Ort::Session create_cuda_session(Ort::Env& env, const char* model_path) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);
    so.SetInterOpNumThreads(4);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    so.EnableCpuMemArena();
    so.EnableMemPattern();

    // Fix: Explicitly cast C++ wrapper to C-style pointer
    OrtSessionOptions* raw_opts = so; 
    
    // Note: If this function is unresolved, you may need to use the OrtApi approach
    // or ensure you are linking 'onnxruntime_providers_cuda'
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(raw_opts, 0);
    
    if (status != nullptr) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        std::cerr << "Failed to append CUDA EP: " << api->GetErrorMessage(status) << "\n";
        api->ReleaseStatus(status);
    }

    return Ort::Session(env, model_path, so);
}

int main() {
    if (!file_exists(MODEL_PATH)) {
        std::cerr << "Error: Model not found at " << MODEL_PATH << '\n';
        return 1;
    }
    cout << "Loading model: " << MODEL_PATH << '\n';

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "single_run_cuda");
    
    try {
        Ort::Session session = create_cuda_session(env, MODEL_PATH);

        vector<int64_t> input_shape = {BATCH_SIZE, IMG_C, IMG_H, IMG_W};
        vector<float> input_data = generate_input(BATCH_SIZE);

        Ort::AllocatorWithDefaultOptions allocator;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
        );

        // --- FIXED NAME HANDLING (Modern ORT API) ---
        // We must keep the smart pointers (AllocatedStringPtr) alive 
        // while we use the raw char* in the Run call.
        
        // 1. Get Input Name
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        vector<const char*> input_names = {input_name};

        // 2. Get Output Names
        size_t out_count = session.GetOutputCount();
        vector<Ort::AllocatedStringPtr> output_name_ptrs;
        vector<const char*> output_names;
        
        for (size_t i = 0; i < out_count; ++i) {
            auto ptr = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(ptr.get());
            output_name_ptrs.push_back(std::move(ptr)); // Keep alive!
        }

        auto t0 = hr_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                          input_names.data(), &input_tensor, 1, 
                                          output_names.data(), output_names.size());
        auto t1 = hr_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cout << "Single GPU (CUDA) inference completed in " << ms << " ms\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}