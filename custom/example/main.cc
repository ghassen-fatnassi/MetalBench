// main.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include "custom_op.h"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    SimpleReLUAddOp my_op;                      // your custom op
    Ort::CustomOpDomain custom_domain("test.domain");
    custom_domain.Add(&my_op);
    session_options.Add(custom_domain);

    const char* model_path = "../yolo12n_op12.onnx";

    // create session
    Ort::Session session(env, model_path, session_options);

    // Query model inputs/outputs
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    std::vector<std::string> input_names_str;
    std::vector<const char*> input_names_c;
    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = session.GetInputName(i, allocator);
        input_names_str.emplace_back(name);
        input_names_c.push_back(input_names_str.back().c_str());
        allocator.Free(name);
    }

    std::vector<std::string> output_names_str;
    std::vector<const char*> output_names_c;
    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = session.GetOutputName(i, allocator);
        output_names_str.emplace_back(name);
        output_names_c.push_back(output_names_str.back().c_str());
        allocator.Free(name);
    }

    std::cout << "Model has " << num_inputs << " inputs:\n";
    for (size_t i = 0; i < num_inputs; ++i) std::cout << "  [" << i << "] " << input_names_str[i] << "\n";
    std::cout << "Model has " << num_outputs << " outputs:\n";
    for (size_t i = 0; i < num_outputs; ++i) std::cout << "  [" << i << "] " << output_names_str[i] << "\n";

    // --- Simple test: require at least 2 model inputs and 1 output ---
    if (num_inputs < 2) {
        std::cerr << "Model has fewer than 2 inputs. Adjust the test to match model inputs.\n";
        return 1;
    }
    if (num_outputs < 1) {
        std::cerr << "Model has no outputs. Can't run.\n";
        return 1;
    }

    // Create simple data (you said you'll fix shapes later)
    size_t size = 8;
    std::vector<float> input1(size);
    std::vector<float> input2(size);
    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i) - 2.0f;
        input2[i] = static_cast<float>(i) * 0.5f;
    }

    // CPU memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // NOTE: dims must match model expectations. This example uses a 1-D tensor of length `size`.
    std::array<int64_t, 1> dims{ static_cast<int64_t>(size) };

    Ort::Value input1_tensor = Ort::Value::CreateTensor<float>(memory_info, input1.data(), input1.size(), dims.data(), dims.size());
    Ort::Value input2_tensor = Ort::Value::CreateTensor<float>(memory_info, input2.data(), input2.size(), dims.data(), dims.size());

    // Build array of Ort::Value (must be contiguous in memory so we can pass pointer)
    std::array<Ort::Value, 2> input_tensors = { std::move(input1_tensor), std::move(input2_tensor) };

    // Run using the model's actual input names (first two inputs)
    const char* input_names_ptrs[2] = { input_names_c[0], input_names_c[1] };
    const char* output_names_ptrs[1] = { output_names_c[0] };

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                     input_names_ptrs,
                                     input_tensors.data(),
                                     2,
                                     output_names_ptrs,
                                     1);
    } catch (const Ort::Exception& e) {
        std::cerr << "Runtime error while running session: " << e.what() << std::endl;
        std::cerr << "Common causes: input shapes/types do not match model's expected input(s)." << std::endl;
        return 1;
    }

    if (output_tensors.size() == 0) {
        std::cerr << "No outputs returned.\n";
        return 1;
    }

    // Access first output as float tensor (be careful: type must match)
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    // to print shape/size you can query type info (omitted for brevity)
    std::cout << "Custom op output: ";
    for (size_t i = 0; i < size; ++i) std::cout << output_data[i] << " ";
    std::cout << std::endl;

    return 0;
}
