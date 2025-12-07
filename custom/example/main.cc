#include <iostream>
#include <vector>
#include <numeric>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "custom_op.h" 

// --- Helper to check ORT Status and throw C++ exception ---
#define ORT_THROW_ON_ERROR(status) \
    do { \
        if (status) { \
            const char* msg = Ort::GetApi().GetErrorMessage(status); \
            std::cerr << "ORT Exception: " << msg << std::endl; \
            Ort::GetApi().ReleaseStatus(status); \
            throw std::runtime_error("ORT operation failed."); \
        } \
    } while(0)

int main() try {
    // --- 1. Setup Environment ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestOrtCustomOp");
    Ort::SessionOptions session_options;

    // Register the custom op library using the class method
    // NOTE: We pass the OrtSessionOptions by pointer (&session_options) 
    // because the ORT C API expects it.
    std::cout << "Attempting to register custom operators..." << std::endl;
    ORT_THROW_ON_ERROR(CustomOpLibrary().RegisterOps(session_options, OrtGetApiBase()));

    // Ensure CUDA Provider is enabled
    OrtCUDAProviderOptions cuda_options{};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    
    std::cout << "CUDA Execution Provider enabled." << std::endl;

    // --- 2. Model Loading and Inference ---
    
    // !!! IMPORTANT !!!
    // The path below must point to an actual ONNX model file that contains
    // a node defined with your custom operator:
    // op_type="SimpleReLUAdd", domain="com.your.custom"
    
    const char* model_path = "model_with_custom_op.onnx"; 
    
    std::cout << "Attempting to create ORT session from model: " << model_path << std::endl;
    
    // NOTE: This line will fail if the model file doesn't exist or is invalid.
    // Replace "model_with_custom_op.onnx" with the path to your actual ONNX model.
    // Ort::Session session(env, model_path, session_options); 
    
    std::cout << "\nTest compiled successfully. The custom op registration logic is now verified (Layer 2)." << std::endl;
    std::cout << "To complete the test, you must create and load an ONNX model file that uses 'SimpleReLUAdd' in the 'com.your.custom' domain." << std::endl;
    
    return 0;

} catch (const std::exception& e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
    return 1;
}