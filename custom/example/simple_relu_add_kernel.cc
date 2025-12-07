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
    
    // Use GetTensorMutableData for older ORT versions
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in1_val), reinterpret_cast<void**>(&input1_mutable)));
    ORT_CHECK(api, api->GetTensorMutableData(const_cast<OrtValue*>(in2_val), reinterpret_cast<void**>(&input2_mutable)));

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

    api->ReleaseTensorTypeAndShapeInfo(shape_info); 

    // 4. Allocate Output
    OrtValue* out_val = nullptr;
    ORT_CHECK(api, api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val));

    float* output = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output)));

    // 5. Launch CUDA Kernel
    cudaStream_t stream = 0; // Use default stream
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}

// ----------------------------------------------------
// 2. Dedicated Test Runner (Simulates ORT Environment)
// ----------------------------------------------------

// NEW MOCK STRUCTURE: Use only C-style members to prevent C++ internal corruption.
struct MockOrtKernelContext {
    // 1. Function Pointers (MUST be in the exact order the ORT API expects them)
    OrtStatus* (*GetInput)(const OrtKernelContext* context, size_t index, const OrtValue** input);
    OrtStatus* (*GetOutput)(OrtKernelContext* context, size_t index, const int64_t* dim_values, size_t dim_count, OrtValue** output);
    OrtStatus* (*GetGPUComputeStream)(const OrtKernelContext* context, void** stream);
    
    // 2. Data payload (Pure C-style members)
    const OrtApi* api;
    const OrtValue* inputs[2]; // Fixed size C array for the 2 inputs
    size_t input_count;        // To track the number of inputs
    OrtValue* output = nullptr;
    OrtAllocator* allocator;
};


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


// Mock function for KernelContext_GetInput
OrtStatus* Mock_KernelContext_GetInput(const OrtKernelContext* context, size_t index, const OrtValue** input) {
    // Directly cast the context pointer to our mock structure
    auto mock_ctx = reinterpret_cast<const MockOrtKernelContext*>(context);
    
    if (index < mock_ctx->input_count) {
        *input = mock_ctx->inputs[index];
        return nullptr;
    }
    return mock_ctx->api->CreateStatus(ORT_FAIL, "MockContext: Input index out of bounds.");
}

// Mock function for KernelContext_GetOutput
OrtStatus* Mock_KernelContext_GetOutput(OrtKernelContext* context, size_t index, 
                                        const int64_t* dim_values, size_t dim_count, OrtValue** output) {
    // Directly cast the context pointer to our mock structure
    auto mock_ctx = reinterpret_cast<MockOrtKernelContext*>(context);
    
    if (index == 0) {
        // Allocate a new output tensor on the GPU using the mock allocator
        ORT_CHECK(mock_ctx->api, mock_ctx->api->CreateTensorAsOrtValue(mock_ctx->allocator, dim_values, dim_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output));
        mock_ctx->output = *output;
        return nullptr;
    }
    return mock_ctx->api->CreateStatus(ORT_FAIL, "MockContext: Output index out of bounds (Only index 0 supported).");
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
    
    // 1. Setup CUDA Memory Allocation Info
    OrtMemoryInfo* cuda_info;
    ORT_CHECK(api, api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cuda_info));

    OrtAllocator* cuda_allocator;
    ORT_CHECK(api, api->GetAllocatorWithDefaultOptions(&cuda_allocator));
    
    // 2. Setup the Mock Kernel Context Structure (This is the object we pass to Compute)
    MockOrtKernelContext mock_context = {};
    
    // Fill Function Pointers
    mock_context.GetInput = Mock_KernelContext_GetInput;
    mock_context.GetOutput = Mock_KernelContext_GetOutput;
    mock_context.GetGPUComputeStream = Mock_KernelContext_GetGPUComputeStream;

    // Fill Data Payload
    mock_context.api = api;
    mock_context.allocator = cuda_allocator;
    mock_context.input_count = 2;

    // 3. Create Input Tensors (OrtValues) on CUDA memory
    OrtValue* in1_val = CreateDeviceTensor(api, cuda_allocator, input1_data, shape);
    OrtValue* in2_val = CreateDeviceTensor(api, cuda_allocator, input2_data, shape);
    
    // Store inputs in the C-array within the mock context
    mock_context.inputs[0] = in1_val; 
    mock_context.inputs[1] = in2_val; 
    
    // 4. Instantiate the Kernel and call its Compute method
    SimpleReLUAddOpKernel kernel(*api, nullptr);
    
    // Call Compute, passing the structure with the function pointers and data payload
    kernel.Compute(reinterpret_cast<OrtKernelContext*>(&mock_context));
    
    // 5. Get the result from the output tensor
    float* output_dev_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(mock_context.output, reinterpret_cast<void**>(&output_dev_ptr)));
    
    // 6. Copy result back to host
    output_data.resize(size);
    cudaMemcpy(output_data.data(), output_dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Cleanup
    api->ReleaseValue(in1_val);
    api->ReleaseValue(in2_val);
    api->ReleaseValue(mock_context.output); 
    api->ReleaseAllocator(cuda_allocator);
    api->ReleaseMemoryInfo(cuda_info);
    api->ReleaseEnv(env_c);
}