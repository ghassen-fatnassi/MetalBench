#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <dlfcn.h>               // POSIX Dynamic Library Loading
#include "onnxruntime/core/framework/customregistry.h" 

// --- Configuration Constants ---
const std::string MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";
// ** YOU MUST UPDATE THIS PATH to your compiled .so file **
const std::string CUSTOM_OP_LIBRARY_PATH = "./build/libort_custom_fused_attn.so"; 

const int BATCH_SIZE = 1;
const int IMG_SIZE = 640;
const int NUM_WARMUP = 5;
const int NUM_RUNS   = 50;
// ------------------------------------------

// --- Custom Op Registration Helpers (POSIX Only) ---

// Define the signature of the function exported from your shared library
using RegisterCustomOpsFn = std::shared_ptr<onnxruntime::CustomRegistry>(*)();

// Function to load the library and find the registration function
void* LoadLibraryAndGetFunction(const char* library_path, const char* function_name, RegisterCustomOpsFn& out_func) {
    
    // Load the shared library
    // RTLD_NOW ensures all undefined symbols are resolved immediately
    void* library_handle = dlopen(library_path, RTLD_NOW);
    
    if (!library_handle) {
        std::cerr << "ERROR: Failed to load custom op library: " << library_path << std::endl;
        std::cerr << "dlerror: " << dlerror() << std::endl;
        return nullptr;
    }

    // Find the symbol (function pointer) inside the loaded library
    out_func = (RegisterCustomOpsFn)dlsym(library_handle, function_name);

    if (!out_func) {
        std::cerr << "ERROR: Failed to find registration function: " << function_name << std::endl;
        // Clean up the library since we couldn't find the function
        dlclose(library_handle); 
        return nullptr;
    }

    return library_handle;
}
// ------------------------------------------


int main() {
    void* custom_op_library_handle = nullptr; 

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Bench");
        Ort::SessionOptions session_options;

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();
        session_options.SetIntraOpNumThreads(4);
        session_options.SetInterOpNumThreads(4);

        // --- CUSTOM OP INTEGRATION START (POSIX) ---
        
        RegisterCustomOpsFn register_func = nullptr;
        
        // 1. Load the shared library and get the registration function pointer
        custom_op_library_handle = LoadLibraryAndGetFunction(
            CUSTOM_OP_LIBRARY_PATH.c_str(), 
            "RegisterFusedAttnOps", // The name of the C-linkage function you exported
            register_func
        );

        if (!custom_op_library_handle) {
            std::cerr << "FATAL: Custom Op registration failed. Exiting.\n";
            return -1;
        }

        // 2. Get the CustomRegistry object
        std::shared_ptr<onnxruntime::CustomRegistry> registry = register_func();
        
        // 3. CRITICAL STEP: Register the custom registry with SessionOptions
        if (!session_options.RegisterCustomRegistry(registry).IsOK()) {
            std::cerr << "FATAL: Could not register custom op registry. Exiting.\n";
            dlclose(custom_op_library_handle);
            return -1;
        }
        std::cout << "Custom Op 'FusedAttnOp' successfully registered.\n";

        // --- CUSTOM OP INTEGRATION END ---

        // CUDA provider (Your existing code)
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options); 
            std::cout << "CUDA Provider Appended.\n";
        } catch (...) {
            std::cerr << "WARNING: Could not append CUDA provider.\n";
        }

        std::cout << "Loading model: " << MODEL_PATH << std::endl;
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);

        // ... [Rest of your existing code for data preparation and benchmarking] ...

        // Prepare data
        std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
        size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
        std::vector<float> input_data(input_size);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& v : input_data) v = dis(gen);

        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name_str = session.GetInputName(0, allocator);
        const char* input_names[] = {input_name_str.c_str()};

        std::string output_name_str = session.GetOutputName(0, allocator);
        const char* output_names[] = {output_name_str.c_str()};

        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_size,
            input_dims.data(), input_dims.size()
        );

        // Warmup
        std::cout << "Warmup...\n";
        for (int i = 0; i < NUM_WARMUP; i++) {
            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);
        }

        // Benchmark
        std::vector<double> times;
        times.reserve(NUM_RUNS);

        std::cout << "Benchmarking...\n";
        for (int i = 0; i < NUM_RUNS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            session.Run(Ort::RunOptions{nullptr},
                        input_names, &input_tensor, 1,
                        output_names, 1);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            times.push_back(ms);
        }

        // Stats
        double sum = 0, mn = 1e9, mx = 0;
        for (auto v : times) {
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double avg = sum / times.size();

        std::cout << "\n---- RESULTS (" << NUM_RUNS << " runs) ----\n";
        std::cout << "Avg: " << avg << " ms\n";
        std::cout << "Min: " << mn << " ms\n";
        std::cout << "Max: " << mx << " ms\n";
        std::cout << "FPS: " << 1000.0 / avg << "\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << std::endl;
        // Clean up the library if an ORT exception occurs
        if (custom_op_library_handle) {
            dlclose(custom_op_library_handle);
        }
        return -1;
    }

    // --- Cleanup Library Handle ---
    if (custom_op_library_handle) {
        dlclose(custom_op_library_handle);
    }
    
    return 0;
}