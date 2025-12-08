#include "fused_attn_op.h"
#include <iostream>
#include <cuda_runtime.h>

// --- Helper for dimension retrieval (keep this) ---
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(const OrtApi& api, const OrtValue* value) {
    Ort::CustomOpApi ort(api);
    OrtTensorTypeAndShapeInfo* info = ort->GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort->GetTensorShape(info));
    ort->ReleaseTensorTypeAndShapeInfo(info);
  }
};

// --- FusedAttnKernel Implementation ---

FusedAttnKernel::FusedAttnKernel(const OrtApi& api, const OrtKernelInfo* /*info*/) : api_(api) {
    // Constructor logic, if needed (e.g., reading node attributes from info)
}

void FusedAttnKernel::Compute(OrtKernelContext* context) {
    // The Ort::CustomOpApi member is already initialized in the constructor
    // Ort::CustomOpApi api = this->api_; 
    
    // Using the member 'api_' for all calls:
    
    const OrtValue* input_tensor = api_->KernelContext_GetInput(context, 0);
    const float* input_data = api_->GetTensorData<float>(input_tensor);

    // Note: Use 'api_' when creating the dimension helper
    OrtTensorDimensions dims(api_, input_tensor);

    OrtValue* output_tensor = api_->KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    float* output_data = api_->GetTensorMutableData<float>(output_tensor);

    // No-op: just copy input to output (using CUDA)
    size_t total_len = 1;
    for (auto d : dims) total_len *= d;
    
    // Perform CUDA computation here
    cudaMemcpy(output_data, input_data, total_len * sizeof(float), cudaMemcpyDeviceToDevice);
}

// --- C-Style API Implementations ---

// Create an instance of the op definition struct (only done once during registration)
OrtCustomOp* CreateFusedAttnOp() {
    OrtCustomOp* op = new OrtCustomOp{};
    
    op->version = ORT_API_VERSION; // Match the API version

    // Assign C-style function pointers
    op->CreateKernel = FusedAttnOp_CreateKernel;
    op->GetName = FusedAttnOp_GetName;
    op->GetExecutionProviderType = FusedAttnOp_GetExecutionProviderType;
    op->GetInputTypeCount = FusedAttnOp_GetInputTypeCount;
    op->GetInputType = FusedAttnOp_GetInputType;
    op->GetOutputTypeCount = FusedAttnOp_GetOutputTypeCount;
    op->GetOutputType = FusedAttnOp_GetOutputType;
    op->KernelCompute = FusedAttnOp_KernelCompute;
    op->KernelDestroy = FusedAttnOp_KernelDestroy;

    return op;
}


const char* ORT_API_CALL FusedAttnOp_GetName(const void* /*op*/) { return "FusedAttnOp"; }
const char* ORT_API_CALL FusedAttnOp_GetExecutionProviderType(const void* /*op*/) { return "CUDA"; }
size_t ORT_API_CALL FusedAttnOp_GetInputTypeCount(const void* /*op*/) { return 1; }
ONNXTensorElementDataType ORT_API_CALL FusedAttnOp_GetInputType(const void* /*op*/, size_t /*index*/) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t ORT_API_CALL FusedAttnOp_GetOutputTypeCount(const void* /*op*/) { return 1; }
ONNXTensorElementDataType ORT_API_CALL FusedAttnOp_GetOutputType(const void* /*op*/, size_t /*index*/) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }


// This function creates an instance of the C++ kernel class
void* ORT_API_CALL FusedAttnOp_CreateKernel(const void* /*op*/, const OrtApi* api, const OrtKernelInfo* info) {
    // Pass the dereferenced API pointer (const OrtApi&) to the C++ constructor
    return new FusedAttnKernel(*api, info); 
}

// This function calls the C++ kernel's Compute method
void ORT_API_CALL FusedAttnOp_KernelCompute(void* op_kernel, OrtKernelContext* context) {
    // Cast the void* back to the C++ kernel class
    static_cast<FusedAttnKernel*>(op_kernel)->Compute(context);
}

// This function calls the C++ kernel's destructor
void ORT_API_CALL FusedAttnOp_KernelDestroy(void* op_kernel) {
    // Cast the void* back to the C++ kernel class and delete
    delete static_cast<FusedAttnKernel*>(op_kernel);
}