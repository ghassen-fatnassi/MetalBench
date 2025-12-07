#include "custom_op.h"
#include <vector>

void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi* api = &api_;

    // 1. Get Inputs
    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetInput(context, 0, &in1_val));
    ORT_CHECK(api, api->KernelContext_GetInput(context, 1, &in2_val));

    // 2. Get Input Data Pointers
    float* input1 = nullptr;
    float* input2 = nullptr;
    // Note: This relies on the memory being on the GPU (as per CUDAExecutionProvider)
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1)));
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2)));

    // 3. Get Input Shape
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    ORT_CHECK(api, api->GetTensorTypeAndShape(in1_val, &shape_info));

    size_t dim_count = 0;
    ORT_CHECK(api, api->GetDimensionsCount(shape_info, &dim_count));

    std::vector<int64_t> shape(dim_count);
    ORT_CHECK(api, api->GetDimensions(shape_info, shape.data(), dim_count));

    size_t size = 1;
    for (auto d : shape) size *= d;

    // Cleanup shape info
    api->ReleaseTensorTypeAndShapeInfo(shape_info); 

    // 4. Allocate Output (ORT allocates memory on the GPU)
    OrtValue* out_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val));

    float* output = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output)));

    // 5. Get CUDA Stream and Launch Kernel
    // Get the stream managed by the ORT CUDA Execution Provider
    cudaStream_t stream = nullptr; 
    api->KernelContext_GetGPUComputeStream(context, &stream);
    
    // Launch wrapper (The actual kernel call is inside this function)
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}