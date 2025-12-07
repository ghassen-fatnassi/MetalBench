#include "custom_op.h"
#include <vector>
#include <iostream>

void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi& api = api_;  // api_ is already a reference
    
    // Get inputs
    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;
    OrtStatus* status = nullptr;
    
    status = api.KernelContext_GetInput(context, 0, &in1_val);
    if (status != nullptr) {
        std::cerr << "Error getting input 0" << std::endl;
        api.ReleaseStatus(status);
        return;
    }
    
    status = api.KernelContext_GetInput(context, 1, &in2_val);
    if (status != nullptr) {
        std::cerr << "Error getting input 1" << std::endl;
        api.ReleaseStatus(status);
        return;
    }
    
    // Get input data pointers
    float* input1 = nullptr;
    float* input2 = nullptr;
    api.GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1));
    api.GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2));
    
    // Get shape information
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api.GetTensorTypeAndShape(in1_val, &shape_info);
    
    size_t dim_count = 0;
    api.GetDimensionsCount(shape_info, &dim_count);
    
    std::vector<int64_t> shape(dim_count);
    api.GetDimensions(shape_info, shape.data(), dim_count);
    
    // Calculate total size
    size_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    
    // Clean up shape info BEFORE getting output (important!)
    api.ReleaseTensorTypeAndShapeInfo(shape_info);
    
    // Get output tensor
    OrtValue* out_val = nullptr;
    status = api.KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val);
    if (status != nullptr) {
        std::cerr << "Error getting output" << std::endl;
        api.ReleaseStatus(status);
        return;
    }
    
    float* output = nullptr;
    api.GetTensorMutableData(out_val, reinterpret_cast<void**>(&output));
    
    // Get CUDA stream from context
    void* cuda_stream = nullptr;
    api.KernelContext_GetGPUComputeStream(context, &cuda_stream);
    
    // Launch kernel with proper stream
    SimpleReLUAddKernelLaunch(static_cast<cudaStream_t>(cuda_stream), 
                              input1, input2, output, size);
    
    // Optional: Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}