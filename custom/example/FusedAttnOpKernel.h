#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <vector>

// Utility to get tensor dimensions
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct FusedAttnOpKernel {
    // CHANGE: Remove 'const' from the API parameter
    FusedAttnOpKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {}
    void Compute(OrtKernelContext* context);
private:
    // CHANGE: Remove 'const' from the member type
    Ort::CustomOpApi ort_; 
};