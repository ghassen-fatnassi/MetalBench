// main.cc
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "custom_op.h"

int main() {
    // 1. Initialize Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestCustomOp");
    Ort::SessionOptions session_options;

    // 2. Register Custom Ops
    // We define the domain name used in the ONNX model
    Ort::CustomOpDomain custom_domain("test.custom_ops");
    RegisterSimpleReLUAdd(custom_domain);
    session_options.Add(custom_domain);

    // 3. Enable CUDA Execution Provider
    // Note: older ORT versions use OrtSessionOptionsAppendExecutionProvider_CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // 4. Load Model
    const char* model_path = "custom_op_test.onnx";
    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path, session_options);

    // 5. Prepare Inputs (Host CPU -> ORT handles copy to GPU internally)
    std::vector<int64_t> input_shape = {1, 5};
    size_t element_count = 5;
    std::vector<float> input1_data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}; // X1
    std::vector<float> input2_data = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f}; // X2

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input1_data.data(), element_count, input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input2_data.data(), element_count, input_shape.data(), input_shape.size()));

    const char* input_names[] = {"X1", "X2"};
    const char* output_names[] = {"Y"};

    // 6. Run Inference
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, input_tensors.data(), 2, 
        output_names, 1
    );

    // 7. Verify Output (Logic: ReLU(X1) + X2)
    // -1 -> 0 + 10 = 10
    // 1  -> 1 + 10 = 11
    float* float_arr = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Output: ";
    for (int i = 0; i < element_count; i++) {
        std::cout << float_arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}