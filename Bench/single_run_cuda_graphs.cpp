#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

const std::string MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx";
const int BATCH_SIZE = 1;
const int IMG_SIZE = 640;

int main() {
    // ----------------------
    // ONNX Runtime Setup
    // ----------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CUDA_Graphs");
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetIntraOpNumThreads(2);

    // ----------------------
    // CUDA provider (ORT 1.6)
    // ----------------------
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // ----------------------
    // Create session
    // ----------------------
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name_str = session.GetInputName(0, allocator);
    std::string output_name_str = session.GetOutputName(0, allocator);
    const char* input_names[] = { input_name_str.c_str() };
    const char* output_names[] = { output_name_str.c_str() };

    // ----------------------
    // Prepare input tensor (PINNED memory for faster async copy)
    // ----------------------
    std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
    size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
    size_t input_bytes = input_size * sizeof(float);

    // Use pinned memory for host data (if possible/needed for best perf)
    float* h_input_data;
    cudaMallocHost((void**)&h_input_data, input_bytes);

    // Fill host input data (same as your original input_data logic)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < input_size; ++i) h_input_data[i] = dis(gen);

    // ----------------------
    // CUDA stream + graph
    // ----------------------
    cudaStream_t stream;
    // Create a non-blocking stream to avoid implicit sync with default stream
    // This is good practice for concurrency but not strictly required for the fix, 
    // as we are manually managing the copy.
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); 

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // ----------------------
    // Prepare GPU input and output buffers
    // ----------------------
    float* d_input_data = nullptr;
    cudaMalloc((void**)&d_input_data, input_bytes);

    // Determine output size (You will need to know this from your model)
    // Assuming output is BATCH_SIZE * 25200 * 85 (standard YOLOv5/8 output)
    size_t output_size = BATCH_SIZE * 25200 * 85; 
    size_t output_bytes = output_size * sizeof(float);
    float* d_output_data = nullptr;
    cudaMalloc((void**)&d_output_data, output_bytes);

    // ----------------------
    // ONNX Runtime I/O Binding Setup
    // ----------------------
    Ort::IoBinding io_binding(session);
    Ort::MemoryInfo mem_info_device = Ort::MemoryInfo::CreateDevice(
        "CUDA", 0 // The device ID
    );

    // Bind the GPU input buffer
    Ort::Value input_tensor_gpu = Ort::Value::CreateTensor<float>(
        mem_info_device, d_input_data, input_size, input_dims.data(), input_dims.size()
    );
    io_binding.BindInput(input_name_str.c_str(), input_tensor_gpu);

    // Bind the GPU output buffer
    std::vector<int64_t> output_dims = {BATCH_SIZE, 25200, 85}; // Placeholder output dims
    Ort::Value output_tensor_gpu = Ort::Value::CreateTensor<float>(
        mem_info_device, d_output_data, output_size, output_dims.data(), output_dims.size()
    );
    io_binding.BindOutput(output_name_str.c_str(), output_tensor_gpu);

    // Set the stream for the execution
    OrtCUDAProviderOptionsV2* cuda_options_v2 = nullptr;
    OrtApi::GetApi().GetExecutionProviderApi("CUDA", ORT_API_VERSION)->Get:onnxruntime:core:session:onnxruntime_cxx_api.h:OrtCUDAProviderOptionsV2::SetCudaStream(io_binding.Get<Ort::IoBinding::Impl>()->GetOrtBinding(), stream);


    // ----------------------
    // Warmup run (perform copy and execution)
    // ----------------------
    std::cout << "Warmup run..." << std::endl;

    // HtoD copy OUTSIDE the capture block
    cudaMemcpyAsync(d_input_data, h_input_data, input_bytes, cudaMemcpyHostToDevice, stream);

    // Run with I/O Binding and the custom stream
    Ort::RunOptions run_options{nullptr};
    session.Run(run_options, io_binding);

    cudaStreamSynchronize(stream); // Sync with custom stream

    // ----------------------
    // Capture CUDA graph
    // ----------------------
    std::cout << "Capturing CUDA graph..." << std::endl;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // --- The operations to capture ---

    // HtoD copy (Must use Async and a captured stream)
    cudaMemcpyAsync(d_input_data, h_input_data, input_bytes, cudaMemcpyHostToDevice, stream);

    // Execute the model (now using the custom stream)
    session.Run(run_options, io_binding);

    // --- End of captured operations ---
    cudaStreamEndCapture(stream, &graph);


    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    std::cout << "Capture completed." << std::endl;

    // ----------------------
    // Replay the graph
    // ----------------------
    std::cout << "Replaying CUDA graph 10 times..." << std::endl;
    for (int i = 0; i < 10; i++) {
        cudaGraphLaunch(graph_exec, stream);
        cudaStreamSynchronize(stream);
    }

    std::cout << "Done!" << std::endl;

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
