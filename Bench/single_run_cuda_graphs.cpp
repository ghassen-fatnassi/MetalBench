#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

const std::string MODEL_PATH = "Models/yolo12n_op12.onnx";
const int BATCH_SIZE = 1;
const int IMG_SIZE = 128;

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

    // CUDA provider (ORT 1.6)
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // ----------------------
    // Create session
    // ----------------------
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);
    session_options.DisableFallback();
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name_str = session.GetInputName(0, allocator);
    std::string output_name_str = session.GetOutputName(0, allocator);
    const char* input_names[] = { input_name_str.c_str() };
    const char* output_names[] = { output_name_str.c_str() };

    std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
    size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
    std::vector<float> input_data(input_size);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for(auto& v : input_data) v = dis(gen);

    Ort::MemoryInfo mem_info_cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemCUDAPinned);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_cpu, input_data.data(), input_size, input_dims.data(), input_dims.size()
    );

    // Allocate GPU output once on first run
    std::vector<Ort::Value> output_tensors;

    // ----------------------
    // CUDA stream + graph
    // ----------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // ----------------------
    // WARMUP: ensures buffers allocated
    // ----------------------
    std::cout << "Warmup run..." << std::endl;
    output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                input_names, &input_tensor, 1,
                                output_names, 1);
    cudaDeviceSynchronize();

    std::cout << "Capturing CUDA graph..." << std::endl;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                input_names, &input_tensor, 1,
                                output_names, 1);

    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

    std::cout << "Capture completed." << std::endl;

    // ----------------------
    // REPLAY
    // ----------------------
    std::cout << "Replaying CUDA graph 10 times..." << std::endl;

    for (int i = 0; i < 10; i++) {
        cudaGraphLaunch(graph_exec, stream);
        cudaStreamSynchronize(stream);
    }

    std::cout << "Done!" << std::endl;

    // Cleanup
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaStreamDestroy(stream);

    return 0;
}
