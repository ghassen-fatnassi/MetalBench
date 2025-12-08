#include "FusedAttnOpKernel.h"
#include <algorithm> // For std::copy

void FusedAttnOpKernel::Compute(OrtKernelContext* context) {
    // 1. Setup Input (X at index 0)
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X_data = ort_.GetTensorData<float>(input_X);

    // 2. Setup Output (Z at index 0): Dimensions come from the input.
    OrtTensorDimensions dimensions(ort_, input_X);
    OrtValue* output_Z = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* Z_data = ort_.GetTensorMutableData<float>(output_Z);

    // Get the total number of elements
    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output_Z);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // 3. Do Computation: Copy X_data to Z_data (Identity)
    std::copy(X_data, X_data + size, Z_data);
    
    // This is equivalent to:
    // for (int64_t i = 0; i < size; i++) {
    //   Z_data[i] = X_data[i];
    // }
}