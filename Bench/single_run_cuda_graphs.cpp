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

    // Optional: prevent fallback to CPU for debugging (will throw if a node can't run on GPU)
    // session_options.DisableFallback();

    // CUDA provider (ORT 1.6)
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
    // Prepare input tensor (pinned memory)
    // ----------------------
    std::vector<int64_t> input_dims = {BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE};
    size_t input_size = BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
    std::vector<float> input_data(input_size);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : input_data) v = dis(gen);

    Ort::MemoryInfo mem_info_cpu = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_cpu, input_data.data(), input_size, input_dims.data(), input_dims.size()
    );

    // ----------------------
    // CUDA stream + graph
    // ----------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // ----------------------
    // Warmup run (full model, CPU fallback allowed)
    // ----------------------
    std::cout << "Warmup run..." << std::endl;
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1,
        output_names, 1  // number of outputs = 1
    );
    cudaDeviceSynchronize();

    // ----------------------
    // Capture CUDA graph (GPU-only nodes)
    // ----------------------
    std::cout << "Capturing CUDA graph..." << std::endl;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Run the same model; CPU fallback nodes will execute normally
    output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1,
        output_names, 1
    );

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
