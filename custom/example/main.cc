#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "custom_op.h"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestCustomOp");
    Ort::SessionOptions session_options;

    // Register custom op
    Ort::CustomOpDomain custom_domain("test.custom_ops");
    RegisterSimpleReLUAdd(custom_domain);
    session_options.Add(custom_domain);

    // Enable CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Load model
    const char* model_path = "custom_op_test.onnx";
    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path, session_options);

    // Prepare inputs
    std::vector<int64_t> input_shape = {1, 5};
    size_t element_count = 5;
    std::vector<float> input1_data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    std::vector<float> input2_data = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input1_data.data(), element_count, input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input2_data.data(), element_count, input_shape.data(), input_shape.size()));

    const char* input_names[] = {"X1", "X2"};
    const char* output_names[] = {"Y"};

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, input_tensors.data(), 2,
                                      output_names, 1);

    float* out_arr = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Output: ";
    for (size_t i = 0; i < element_count; ++i)
        std::cout << out_arr[i] << " ";
    std::cout << std::endl;

    return 0;
}
