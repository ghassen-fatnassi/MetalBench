#include "custom_op.h"
#include <vector>
#include <iostream>

// Helper to check ORT return status
#define ORT_CHECK(api, status) \
    do { \
        OrtStatus* _s = (status); \
        if (_s != nullptr) { \
            const char* msg = api->GetErrorMessage(_s); \
            std::cerr << "ORT Error: " << msg << std::endl; \
            api->ReleaseStatus(_s); \
            /* In a real scenario, you might throw or handle better */ \
            abort(); \
        } \
    } while(0)

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
    
    // Note: Generally for inputs we use GetTensorData (const), but CustomOps often allow Mutable 
    // if you cast away const or if the API version allows it. Keeping your logic:
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

    // 4. Allocate Output
    OrtValue* out_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val));

    float* output = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output)));

    // 5. Launch CUDA Kernel
    // In a real ORT execution, we should get the compute stream from context, 
    // but for ORT 1.6 / custom ops, passing 0 (default stream) is common for simple tests.
    // If you need the specific ORT stream, you would use api->KernelContext_GetGPUComputeStream(context)
    cudaStream_t stream = nullptr; 
    
    // Launch
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}