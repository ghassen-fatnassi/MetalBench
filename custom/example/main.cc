// main.cc
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

// Define the name of the custom op library file created by CMake
#if defined(_WIN32)
static constexpr const char* CUSTOM_OP_LIBRARY_FILENAME = "custom_attn.dll";
#elif defined(__APPLE__)
static constexpr const char* CUSTOM_OP_LIBRARY_FILENAME = "libcustom_attn.dylib";
#else
static constexpr const char* CUSTOM_OP_LIBRARY_FILENAME = "./libcustom_attn.so";
#endif

// Define the path to the fused model created by your Python script
static constexpr const char* FUSED_MODEL_PATH = "../model_fused.onnx";

void RunInferenceTest() {
    // --- 1. Setup ONNX Runtime Environment ---
    // Create an environment handle
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CustomOpTest");

    // --- 2. Setup Session Options and Register Custom Op Library ---
    Ort::SessionOptions session_options;
    void* library_handle = nullptr;

    try {
        // Register the custom library built by CMake
        std::cout << "-> Registering custom op library: " << CUSTOM_OP_LIBRARY_FILENAME << std::endl;
        Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(
            (OrtSessionOptions*)session_options, 
            CUSTOM_OP_LIBRARY_FILENAME, 
            &library_handle));

        // Use CPU provider for simplicity, assuming your op is CPU-only for this test
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during setup/registration: " << e.what() << std::endl;
        return;
    }

    // --- 3. Load the Fused Model ---
    std::unique_ptr<Ort::Session> session;
    try {
        std::cout << "-> Loading fused model: " << FUSED_MODEL_PATH << std::endl;
        session = std::make_unique<Ort::Session>(env, FUSED_MODEL_PATH, session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading model (Did you run the Python fusion script?): " << e.what() << std::endl;
        return;
    }

    // --- 4. Prepare Dummy Input Data ---
    const std::vector<int64_t> input_shape = {1, 3, 2, 2}; // Example shape: [Batch, Channel, H, W]
    const size_t input_size = 1 * 3 * 2 * 2; // Total elements: 12

    // Create a simple float array input
    std::vector<float> input_values(input_size);
    std::iota(input_values.begin(), input_values.end(), 1.0f); // Fill with 1.0, 2.0, 3.0, ...

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Create the input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_values.data(), 
        input_size, 
        input_shape.data(), 
        input_shape.size());

    const char* input_names[] = {"X"}; // Assume your fused model's graph input is named "X"
    const char* output_names[] = {"Y"}; // Assume your fused model's graph output is named "Y"

    // --- 5. Run Inference ---
    std::cout << "-> Running inference..." << std::endl;
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1);
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during session run: " << e.what() << std::endl;
        return;
    }

    // --- 6. Verify Output ---
    assert(output_tensors.size() == 1);
    const float* output_data = output_tensors[0].GetTensorData<float>();
    
    bool match = std::equal(input_values.begin(), input_values.end(), output_data);

    if (match) {
        std::cout << "\n✅ SUCCESS: Custom FusedAttnOp ran successfully and returned an Identity (Copy) result!" << std::endl;
        std::cout << "   The custom operator pipeline is confirmed working." << std::endl;
    } else {
        std::cout << "\n❌ FAILURE: Output data does not match input data." << std::endl;
    }
}

int main() {
    RunInferenceTest();
    return 0;
}