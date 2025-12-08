// fused_attn_op.cc
#include "fused_attn_op.h"
#include <iostream>
#include <cuda_runtime.h> // Include for CUDA functions

// Helper struct for getting dimensions (as seen in ORT documentation)
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};


FusedAttnOp::FusedAttnOp() {
    // Constructor (empty for now)
}

void FusedAttnOp::Compute(OrtKernelContext* context) {
    // Use the CustomOpApi to interact with tensors and memory
    Ort::CustomOpApi api(OrtGetApiBase()->GetApi(ORT_API_VERSION));

    // Get input tensor
    const OrtValue* input_tensor = api.KernelContext_GetInput(context, 0);
    // Since the EP is CUDA, 'input_data' points to GPU memory
    const float* input_data = api.GetTensorData<float>(input_tensor);

    OrtTensorDimensions dims(api, input_tensor);

    // Create output tensor with same shape
    OrtValue* output_tensor = api.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    // 'output_data' also points to GPU memory
    float* output_data = api.GetTensorMutableData<float>(output_tensor);

    // No-op: just copy input to output (MUST be done via CUDA, or Ort will manage it)
    size_t total_len = 1;
    for (auto d : dims) total_len *= d;
    
    // Note: If this kernel is truly for CUDA, you must replace the loop with a GPU kernel launch.
    // For now, if you are testing that the Op runs at all, this operation will implicitly rely on 
    // ONNX Runtime's memory copy infrastructure to handle the host/device transfer if needed.
    // If you intend to call a CUDA kernel here:
    // launch_fused_attn_kernel(total_len, output_data, input_data);
    
    // For a temporary CPU copy (which should NOT be used in a production CUDA op):
    // for (size_t i = 0; i < total_len; ++i)
    //     output_data[i] = input_data[i];

    // For a minimal working example on CUDA, you might use cudaMemcpy:
    // cudaMemcpy(output_data, input_data, total_len * sizeof(float), cudaMemcpyDeviceToDevice);
}
// Removed the closing '}' here.