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
    const float* input1 = nullptr;
    const float* input2 = nullptr;
    // GetTensorData for read-only access (const float*)
    ORT_CHECK(api, api->GetTensorData(in1_val, reinterpret_cast<const void**>(&input1)));
    ORT_CHECK(api, api->GetTensorData(in2_val, reinterpret_cast<const void**>(&input2)));

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
    // OrtKernelContext_GetOutput allocates the output tensor for the given shape/type/location
    ORT_CHECK(api, api->KernelContext_GetOutput(context, 0, shape.data(), dim_count, &out_val));

    float* output = nullptr;
    // GetTensorMutableData is used here because we are writing to the output tensor
    ORT_CHECK(api, api->GetTensorMutableData(out_val, reinterpret_cast<void**>(&output)));

    // 5. Get CUDA Stream (required for proper ORT execution)
    // NOTE: If ORT is older/simpler, this might return nullptr, so we use 0 as fallback.
    cudaStream_t stream = nullptr; 
    ORT_CHECK(api, api->KernelContext_GetGPUComputeStream(context, reinterpret_cast<void**>(&stream)));

    // 6. Launch CUDA Kernel
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
}

// ----------------------------------------------------
// 2. Dedicated Test Runner (Simulates ORT Environment)
// ----------------------------------------------------

// Helper function to create an OrtValue (Tensor) on the Device (CUDA)
OrtValue* CreateDeviceTensor(const OrtApi* api, OrtAllocator* allocator, 
                             const std::vector<float>& data, const std::vector<int64_t>& shape) {
    
    size_t size = data.size();
    OrtValue* value = nullptr;
    
    // Create the OrtValue header with the correct shape/type/location
    ORT_CHECK(api, api->CreateTensorAsOrtValue(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value));
    
    // Get the pointer to the device buffer
    void* buffer_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(value, &buffer_ptr));
    
    // Copy data from host to device
    cudaMemcpy(buffer_ptr, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    return value;
}

// Minimal implementation of OrtKernelContext for testing the Compute function
struct MockKernelContext {
    OrtKernelContext ctx_base;
    const OrtApi* api;
    std::vector<const OrtValue*> inputs;
    OrtValue* output;
    OrtAllocator* allocator;
};

// Mock function for KernelContext_GetInput (required by Compute)
OrtStatus* Mock_KernelContext_GetInput(const OrtKernelContext* context, size_t index, const OrtValue** input) {
    auto mock_ctx = reinterpret_cast<const MockKernelContext*>(context);
    if (index < mock_ctx->inputs.size()) {
        *input = mock_ctx->inputs[index];
        return nullptr;
    }
    return mock_ctx->api->CreateStatus(ORT_FAIL, "MockContext: Input index out of bounds.");
}

// Mock function for KernelContext_GetOutput (required by Compute)
OrtStatus* Mock_KernelContext_GetOutput(OrtKernelContext* context, size_t index, 
                                        const int64_t* dim_values, size_t dim_count, OrtValue** output) {
    auto mock_ctx = reinterpret_cast<MockKernelContext*>(context);
    if (index == 0) {
        // Allocate a new output tensor on the GPU using the mock allocator and the shape provided
        ORT_CHECK(mock_ctx->api, mock_ctx->api->CreateTensorAsOrtValue(mock_ctx->allocator, dim_values, dim_count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output));
        mock_ctx->output = *output;
        return nullptr;
    }
    return mock_ctx->api->CreateStatus(ORT_FAIL, "MockContext: Output index out of bounds (Only index 0 supported).");
}

// Mock function for KernelContext_GetGPUComputeStream
OrtStatus* Mock_KernelContext_GetGPUComputeStream(const OrtKernelContext* context, void** stream) {
    // Return the default stream (0) for the test
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
    
    // Setup CUDA Memory Allocation (Use ORT's CUDA allocator)
    OrtMemoryInfo* cuda_info;
    ORT_CHECK(api, api->CreateMemoryInfo("Cuda", OrtCuda, 0, OrtMemTypeDefault, &cuda_info));

    OrtAllocator* cuda_allocator;
    ORT_CHECK(api, api->CreateAllocator(env_c, cuda_info, &cuda_allocator));
    
    // 1. Create Input Tensors (OrtValues) on CUDA memory
    OrtValue* in1_val = CreateDeviceTensor(api, cuda_allocator, input1_data, shape);
    OrtValue* in2_val = CreateDeviceTensor(api, cuda_allocator, input2_data, shape);
    
    // 2. Setup the Mock Kernel Context
    MockKernelContext mock_ctx = {};
    mock_ctx.api = api;
    mock_ctx.inputs = {in1_val, in2_val};
    mock_ctx.allocator = cuda_allocator;

    // Manually override the function pointers in the base context for the mock
    mock_ctx.ctx_base.GetInput = Mock_KernelContext_GetInput;
    mock_ctx.ctx_base.GetOutput = Mock_KernelContext_GetOutput;
    mock_ctx.ctx_base.GetGPUComputeStream = Mock_KernelContext_GetGPUComputeStream;
    
    // 3. Instantiate the Kernel and call its Compute method
    SimpleReLUAddOpKernel kernel(*api, nullptr);
    kernel.Compute(reinterpret_cast<OrtKernelContext*>(&mock_ctx));
    
    // 4. Get the result from the output tensor created by the mock context
    float* output_dev_ptr = nullptr;
    ORT_CHECK(api, api->GetTensorMutableData(mock_ctx.output, reinterpret_cast<void**>(&output_dev_ptr)));
    
    // 5. Copy result back to host
    cudaMemcpy(output_data.data(), output_dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Cleanup
    api->ReleaseValue(in1_val);
    api->ReleaseValue(in2_val);
    api->ReleaseValue(mock_ctx.output); // Output is released via mock_ctx
    api->ReleaseAllocator(cuda_allocator);
    api->ReleaseMemoryInfo(cuda_info);
    api->ReleaseEnv(env_c);
}