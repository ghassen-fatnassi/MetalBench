#include <iostream>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "custom_op.h"

int main(int argc, char** argv) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SimpleCustomOpTest");
  Ort::SessionOptions session_options;

  // Register custom op domain
  Ort::CustomOpDomain custom_domain("test.custom_ops");
  RegisterSimpleReLUAdd(custom_domain);
  session_options.Add(custom_domain);

#ifdef USE_CUDA
  // enable CUDA EP (ORT built with CUDA)
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif

  const char* model_path = "custom_op_test.onnx";
  std::cout << "Loading model: " << model_path << std::endl;

  try {
    Ort::Session session(env, model_path, session_options);

    // Prepare inputs on CPU (ORT will move to EP memory if needed)
    std::vector<int64_t> input_shape = {1, 5};
    size_t element_count = 5;
    std::vector<float> input1_data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    std::vector<float> input2_data(5, 10.0f);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input1_data.data(), element_count, input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input2_data.data(), element_count, input_shape.data(), input_shape.size()));

    const char* input_names[] = {"X1", "X2"};
    const char* output_names[] = {"Y"};

    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, input_tensors.data(), 2,
                                      output_names, 1);

    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Output: ";
    for (size_t i = 0; i < element_count; ++i) std::cout << out_data[i] << " ";
    std::cout << std::endl;

  } catch (const Ort::Exception& e) {
    std::cerr << "ORT exception: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "std::exception: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
