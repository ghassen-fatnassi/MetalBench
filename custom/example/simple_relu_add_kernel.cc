#include "custom_op.h"
#include <cuda_runtime.h>
#include <stdexcept>

extern "C" void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                                          const float* input1,
                                          const float* input2,
                                          float* output,
                                          size_t size);

void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi* api = &api_;

    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;
    api->KernelContext_GetInput(context, 0, &in1_val);
    api->KernelContext_GetInput(context, 1, &in2_val);

    float* input1 = nullptr;
    float* input2 = nullptr;
    api->GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1));
    api->GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2));

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api->GetTensorTypeAndShape(in1_val, &shape_info);

    size_t dim_count = 0;
    api->GetDimensionsCount(shape_info, &dim_count);

    std::vector<int64_t> shape(dim_count);
    api->GetDimensions(shape_info, shape.data(), dim_count);
    api->ReleaseTensorTypeAndShapeInfo(shape_info);

    size_t size = 1;
    for (auto d : shape) size *= d;

    OrtValue* out_val = nullptr;
    api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val);
    float* output = nullptr;
    api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output));

    // Launch CUDA kernel
    SimpleReLUAddKernelLaunch(0, input1, input2, output, size);
}
