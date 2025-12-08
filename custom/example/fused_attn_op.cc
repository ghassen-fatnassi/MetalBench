// fused_attn_op.cc
#include "fused_attn_op.h"
#include <iostream>

FusedAttnOp::FusedAttnOp() {
    // Constructor (empty for now)
}

void FusedAttnOp::Compute(OrtKernelContext* context) {
    Ort::CustomOpApi api(OrtGetApiBase()->GetApi(ORT_API_VERSION));

    // Get input tensor
    const OrtValue* input_tensor = api.KernelContext_GetInput(context, 0);
    const float* input_data = api.GetTensorData<float>(input_tensor);

    OrtTensorDimensions dims(api, input_tensor);

    // Create output tensor with same shape
    OrtValue* output_tensor = api.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    float* output_data = api.GetTensorMutableData<float>(output_tensor);

    // No-op: just copy input to output
    size_t total_len = 1;
    for (auto d : dims) total_len *= d;
    for (size_t i = 0; i < total_len; ++i)
        output_data[i] = input_data[i]; // replace with GPU kernel later
}
