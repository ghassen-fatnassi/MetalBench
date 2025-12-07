#include "custom_op.h"
#include <vector>
#include <iostream>
#include <algorithm> // Added for std::max in the test runner

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
// 1. ORT Kernel Implementation (Called by ORT session)
// ----------------------------------------------------

void SimpleReLUAddOpKernel::Compute(OrtKernelContext* context) {
    const OrtApi* api = &api_;

    // 1. Get Inputs
    const OrtValue* in1_val = nullptr;
    const OrtValue* in2_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetInput(context, 0, &in1_val));
    ORT_CHECK(api, api->KernelContext_GetInput(context, 1, &in2_val));

    // 2. Get Input Data Pointers
    float* input1_mutable = nullptr;
    float* input2_mutable = nullptr;
    
    // FIX: Use GetTensorMutableData for older ORT versions, casting away const
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1_mutable)));
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2_mutable)));

    // Since the kernel expects const float*, we cast back to const here
    const float* input1 = input1_mutable;
    const float* input2 = input2_mutable;

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

    // 5. Get CUDA Stream 
    // FIX: Removed KernelContext_GetGPUComputeStream in the previous step, using default stream (0)
    cudaStream_t stream = 0; 

    // 6. Launch CUDA Kernel
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}

// ----------------------------------------------------
// 2. Dedicated Test Runner (Simulates ORT Environment)
// ----------------------------------------------------

// Custom structure to hold context data 
struct MockKernelContextData {
    const OrtApi* api;
    std::vector<const OrtValue*> inputs;
    OrtValue* output = nullptr;
    OrtAllocator* allocator;
};

// Forward declarations for the mock functions (since we cannot use the struct definition directly)
OrtStatus* Mock_KernelContext_GetInput(const OrtKernelContext* context, size_t index, const OrtValue** input);
OrtStatus* Mock_KernelContext_GetOutput(OrtKernelContext* context, size_t index, 
                                        const int64_t* dim_values, size_t dim_count, OrtValue** output);
OrtStatus* Mock_KernelContext_GetGPUComputeStream(const OrtKernelContext* context, void** stream); 

// Helper function to create an OrtValue (Tensor) on the Device (CUDA)
OrtValue* CreateDeviceTensor(const OrtApi* api, OrtAllocator* allocator, 
                             const std::vector<float>& data, const std::vector<int64_t>& shape) {
    
    size_t size = data.size();
    OrtValue* value = nullptr;
    
    ORT_CHECK(api, api->CreateTensorAsOrtValue(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value));
    
    void* buffer_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(value, &buffer_ptr));
    
    cudaMemcpy(buffer_ptr, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    return value;
}


// FIX: Implement the mock functions to use the custom MockKernelContextData 
OrtStatus* Mock_KernelContext_GetInput(const OrtKernelContext* context, size_t index, const OrtValue** input) {
    auto mock_ctx_data = reinterpret_cast<const MockKernelContextData*>(context);
    if (index < mock_ctx_data->inputs.size()) {
        *input = mock_ctx_data->inputs[index];
        return nullptr;
    }
    return mock_ctx_data->api->CreateStatus(ORT_FAIL, "MockContext: Input index out of bounds.");
}

OrtStatus* Mock_KernelContext_GetOutput(OrtKernelContext* context, size_t index, 
                                        const int64_t* dim_values, size_t dim_count, OrtValue** output) {
    auto mock_ctx_data = reinterpret_cast<MockKernelContextData*>(context);
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
    
    // Setup CUDA Memory Allocation 
    OrtMemoryInfo* cuda_info;
    
    // FIX 1: Use the compiler suggestion OrtDeviceAllocator as the device type constant
    // This is common in some older ORT custom builds.
    ORT_CHECK(api, api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cuda_info));

    OrtAllocator* cuda_allocator;
    // FIX 2: Use GetAllocatorWithDefaultOptions as suggested by the compiler
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(cuda_info, &cuda_allocator));
    
    // 1. Create Input Tensors (OrtValues) on CUDA memory
    OrtValue* in1_val = CreateDeviceTensor(api, cuda_allocator, input1_data, shape);
    OrtValue* in2_val = CreateDeviceTensor(api, cuda_allocator, input2_data, shape);
    
    // 2. Setup the Mock Kernel Context Data
    MockKernelContextData mock_ctx_data = {};
    mock_ctx_data.api = api;
    mock_ctx_data.inputs = {in1_val, in2_val};
    mock_ctx_data.allocator = cuda_allocator;
    
    // 3. Instantiate the Kernel and call its Compute method
    SimpleReLUAddOpKernel kernel(*api, nullptr);
    
    // Create a temporary structure that mimics the OrtKernelContext function pointers
    struct OrtKernelContext_mock {
        OrtStatus* (*GetInput)(const OrtKernelContext* context, size_t index, const OrtValue** input);
        OrtStatus* (*GetOutput)(OrtKernelContext* context, size_t index, const int64_t* dim_values, size_t dim_count, OrtValue** output);
        OrtStatus* (*GetGPUComputeStream)(const OrtKernelContext* context, void** stream);
    };

    OrtKernelContext_mock mock_base_ctx_functions = {};
    mock_base_ctx_functions.GetInput = Mock_KernelContext_GetInput;
    mock_base_ctx_functions.GetOutput = Mock_KernelContext_GetOutput;
    mock_base_ctx_functions.GetGPUComputeStream = Mock_KernelContext_GetGPUComputeStream;

    // Overlay the function pointers onto the beginning of our mock data structure
    // This allows the kernel's Compute function to call the mock functions.
    // NOTE: This relies on the memory layout of OrtKernelContext, which is non-standard but required for this mock.
    memcpy(&mock_ctx_data, &mock_base_ctx_functions, sizeof(OrtKernelContext_mock));

    // Call Compute, passing the data structure as the required OrtKernelContext*
    kernel.Compute(reinterpret_cast<OrtKernelContext*>(&mock_ctx_data));
    
    // 4. Get the result from the output tensor created by the mock context (from mock_ctx_data.output)
    float* output_dev_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(mock_ctx_data.output, reinterpret_cast<void**>(&output_dev_ptr)));
    
    // 5. Copy result back to host
    cudaMemcpy(output_data.data(), output_dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Cleanup
    api->ReleaseValue(in1_val);
    api->ReleaseValue(in2_val);
    api->ReleaseValue(mock_ctx_data.output); 
    api->ReleaseAllocator(cuda_allocator);
    api->ReleaseMemoryInfo(cuda_info);
    api->ReleaseEnv(env_c);
}