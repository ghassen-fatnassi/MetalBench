#include <iostream>
#include <vector>
#include "custom_op.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::AllocatorWithDefaultOptions allocator;

    // Input sizes
    const size_t size = 8;

    // Create input data
    std::vector<float> input1(size);
    std::vector<float> input2(size);
    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i) - 2.0f; // some negative values
        input2[i] = static_cast<float>(i) * 0.5f;
    }

    // Output buffer
    std::vector<float> output(size, 0.0f);

    // Create kernel API
    Ort::CustomOpDomain custom_domain("test_domain");
    SimpleReLUAddOp simple_op;

    // Normally ONNX Runtime calls CreateKernel for you; here we do it manually
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    SimpleReLUAddOpKernel* kernel = new SimpleReLUAddOpKernel(*api, nullptr);

    // Directly launch the kernel (CPU version)
    kernel->Compute(nullptr);

    // For simplicity, manually copy input data and call CUDA kernel
    SimpleReLUAddKernelLaunch(0, input1.data(), input2.data(), output.data(), size);

    // Print results
    std::cout << "SimpleReLUAdd output:\n";
    for (size_t i = 0; i < size; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    delete kernel;
    return 0;
}
