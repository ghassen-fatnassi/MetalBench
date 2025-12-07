#include "custom_op.h"
#include <vector>
#include <iostream>
#include <algorithm> 
#include <string.h> 

// Helper to check ORT return status
#define ORT_CHECK(api, status) \
    do { \
        OrtStatus* _s = (status); \
        if (_s != nullptr) { \
            const char* msg = api->GetErrorMessage(_s); \
            std::cerr << "ORT Error: " << msg << std::endl; \
            api->ReleaseStatus(_s); \
            abort(); \
        } \
    } while(0)

// ----------------------------------------------------
// 1. ORT Kernel Implementation (Compute method)
// ... (The Compute method remains the same)
// ----------------------------------------------------
void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi* api = &api_;

    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetInput(context, 0, &in1_val));
    ORT_CHECK(api, api->KernelContext_GetInput(context, 1, &in2_val));

    float* input1_mutable = nullptr;
    float* input2_mutable = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1_mutable)));
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2_mutable)));
    const float* input1 = input1_mutable;
    const float* input2 = input2_mutable;

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    ORT_CHECK(api, api->GetTensorTypeAndShape(in1_val, &shape_info));

    size_t dim_count = 0;
    ORT_CHECK(api, api->GetDimensionsCount(shape_info, &dim_count));

    std::vector<int64_t> shape(dim_count);
    ORT_CHECK(api, api->GetDimensions(shape_info, shape.data(), dim_count));

    size_t size = 1;
    for (auto d : shape) size *= d;

    api->ReleaseTensorTypeAndShapeInfo(shape_info); 

    OrtValue* out_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val));

    float* output = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output)));

    cudaStream_t stream = 0; 
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}

// ----------------------------------------------------
// 2. Dedicated Test Runner (Simulates ORT Environment)
// ----------------------------------------------------

// Data payload: The actual data required by the mock functions
struct MockKernelContextData {
    const OrtApi* api;
    std::vector<const OrtValue*> inputs;
    OrtValue* output = nullptr;
    OrtAllocator* allocator;
};

// **NEW MOCK STRUCTURE** 
// This structure explicitly follows the expected function pointer layout of OrtKernelContext
// and holds a pointer back to our actual MockKernelContextData.
struct OrtKernelContext_Mock {
    // These pointers MUST be in the exact order the ORT API expects them.
    OrtStatus* (*GetInput)(const OrtKernelContext* context, size_t index, const OrtValue** input);
    OrtStatus* (*GetOutput)(OrtKernelContext* context, size_t index, const int64_t* dim_values, size_t dim_count, OrtValue** output);
    OrtStatus* (*GetGPUComputeStream)(const OrtKernelContext* context, void** stream);
    
    // Pointer to the actual data used by the mock functions
    MockKernelContextData* data;
};


// Helper function to create an OrtValue (Tensor) on the Device (CUDA)
OrtValue* CreateDeviceTensor(const OrtApi* api, OrtAllocator* allocator, 
                             const std::vector<float>& data, const std::vector<int64_t>& shape) {
    // ... (unchanged)
    size_t size = data.size();
    OrtValue* value = nullptr;
    
    ORT_CHECK(api, api->CreateTensorAsOrtValue(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value));
    
    void* buffer_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(value, &buffer_ptr));
    
    cudaMemcpy(buffer_ptr, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    return value;
}


// Mock function for KernelContext_GetInput
OrtStatus* Mock_KernelContext_GetInput(const OrtKernelContext* context, size_t index, const OrtValue** input) {
    // Use the context pointer to find the data structure
    auto mock_ctx_data = reinterpret_cast<const OrtKernelContext_Mock*>(context)->data;
    
    if (index < mock_ctx_data->inputs.size()) {
        *input = mock_ctx_data->inputs[index];
        return nullptr;
    }
    return mock_ctx_data->api->CreateStatus(ORT_FAIL, "MockContext: Input index out of bounds.");
}

// Mock function for KernelContext_GetOutput
OrtStatus* Mock_KernelContext_GetOutput(OrtKernelContext* context, size_t index, 
                                        const int64_t* dim_values, size_t dim_count, OrtValue** output) {
    // Use the context pointer to find the data structure
    auto mock_ctx_data = reinterpret_cast<OrtKernelContext_Mock*>(context)->data;
    
    if (index == 0) {
        // Allocate a new output tensor on the GPU using the mock allocator
        ORT_CHECK(mock_ctx_data->api, mock_ctx_data->api->CreateTensorAsOrtValue(mock_ctx_data->allocator, dim_values, dim_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output));
        mock_ctx_data->output = *output;
        return nullptr;
    }
    return mock_ctx_data->api->CreateStatus(ORT_FAIL, "MockContext: Output index out of bounds (Only index 0 supported).");
}

// Simple stream function for older ORT versions (returns 0/nullptr)
OrtStatus* Mock_KernelContext_GetGPUComputeStream(const OrtKernelContext* context, void** stream) {
    *stream = nullptr; 
    return nullptr;
}


void SimpleReLUAdd_ORT_Test(const std::vector<float>& input1_data, 
                             const std::vector<float>& input2_data, 
                             std::vector<float>& output_data, 
                             size_t size) {
    
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env_c;
    ORT_CHECK(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SimpleORTTest", &env_c));

    std::vector<int64_t> shape = { (int64_t)size };
    
    OrtMemoryInfo* cuda_info;
    ORT_CHECK(api, api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cuda_info));

    OrtAllocator* cuda_allocator;
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(&cuda_allocator));
    
    // 1. Create Input Tensors (OrtValues) on CUDA memory
    OrtValue* in1_val = CreateDeviceTensor(api, cuda_allocator, input1_data, shape);
    OrtValue* in2_val = CreateDeviceTensor(api, cuda_allocator, input2_data, shape);
    
    // 2. Setup the Mock Kernel Context Data (the actual data payload)
    MockKernelContextData mock_data = {};
    mock_data.api = api;
    mock_data.inputs = {in1_val, in2_val};
    mock_data.allocator = cuda_allocator;

    // 3. Setup the Mock Context Structure (the object passed to Compute)
    OrtKernelContext_Mock mock_context = {};
    mock_context.GetInput = Mock_KernelContext_GetInput;
    mock_context.GetOutput = Mock_KernelContext_GetOutput;
    mock_context.GetGPUComputeStream = Mock_KernelContext_GetGPUComputeStream;
    mock_context.data = &mock_data; // Link the function pointers to the data payload

    // 4. Instantiate the Kernel and call its Compute method
    SimpleReLUAddOpKernel kernel(*api, nullptr);
    
    // Pass the structure with the function pointers as the required OrtKernelContext*
    kernel.Compute(reinterpret_cast<OrtKernelContext*>(&mock_context));
    
    // 5. Get the result from the output tensor
    float* output_dev_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(mock_data.output, reinterpret_cast<void**>(&output_dev_ptr)));
    
    // 6. Copy result back to host
    cudaMemcpy(output_data.data(), output_dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Cleanup
    api->ReleaseValue(in1_val);
    api->ReleaseValue(in2_val);
    api->ReleaseValue(mock_data.output); 
    api->ReleaseAllocator(cuda_allocator);
    api->ReleaseMemoryInfo(cuda_info);
    api->ReleaseEnv(env_c);
}