#include "custom_op.h"
#include <cuda_runtime.h>

void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // -----------------------------
    // Get input tensors
    // -----------------------------
    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;

    api->KernelContext_GetInput(context, 0, &in1_val);
    api->KernelContext_GetInput(context, 1, &in2_val);

    // Raw data (ORT 1.6 has no template helpers)
    float* in1 = nullptr;
    float* in2 = nullptr;

    api->GetTensorMutableData(in1_val, (void**)&in1);
    api->GetTensorMutableData(in2_val, (void**)&in2);

    // -----------------------------
    // Get tensor shape
    // -----------------------------
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api->GetTensorTypeAndShape(in1_val, &shape_info);

    size_t dim_count = 0;
    api->GetDimensionsCount(shape_info, &dim_count);

    std::vector<int64_t> shape(dim_count);
    api->GetDimensions(shape_info, shape.data(), dim_count);

    size_t total = 1;
    for (size_t i = 0; i < dim_count; i++) {
        total *= shape[i];
    }

    // -----------------------------
    // Create output tensor
    // -----------------------------
    OrtValue* out_val = nullptr;
    api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val);

    float* out = nullptr;
    api->GetTensorMutableData(out_val, (void**)&out);

    // -----------------------------
    // CUDA kernel launch
    // -----------------------------
    simple_add_relu_cuda(in1, in2, out, total);
}
