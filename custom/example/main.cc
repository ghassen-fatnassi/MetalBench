#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include "custom_op.h"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    SimpleReLUAddOp my_op;
    Ort::CustomOpDomain custom_domain("test.domain");
    custom_domain.Add(&my_op);
    session_options.Add(custom_domain);

    // Minimal dummy ONNX model file path
    const char* model_path = "dummy_model.onnx";
    Ort::Session session(env, model_path, session_options);

    size_t size = 8;
    std::vector<float> input1(size);
    std::vector<float> input2(size);
    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i) - 2.0f;
        input2[i] = static_cast<float>(i) * 0.5f;
    }

    // Create input tensors
    Ort::AllocatorWithDefaultOptions allocator;
    std::array<int64_t,1> dims{static_cast<int64_t>(size)};
    Ort::Value input1_tensor = Ort::Value::CreateTensor<float>(allocator, input1.data(), size, dims.data(), 1);
    Ort::Value input2_tensor = Ort::Value::CreateTensor<float>(allocator, input2.data(), size, dims.data(), 1);

    const char* input_names[] = {"input1","input2"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names,
                                      &input1_tensor,
                                      2,
                                      output_names,
                                      1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "Custom op output: ";
    for (size_t i = 0; i < size; ++i) std::cout << output_data[i] << " ";
    std::cout << std::endl;

    return 0;
}
