#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <random>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Logger
class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// Deleter
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const { if (obj) obj->destroy(); }
};

// Random data generator
void create_random_data(float* data, size_t size, unsigned int seed = 1U)
{
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
        data[i] = dis(eng);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <plugin_library_path> <engine_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string plugin_library_path{argv[1]};
    std::string engine_file_path{argv[2]};

    CustomLogger logger{};

    // 1. Load Plugin Library
    auto dlclose_deleter = [](void* handle){ dlclose(handle); };
    std::unique_ptr<void, decltype(dlclose_deleter)> plugin_handle{
        dlopen(plugin_library_path.c_str(), RTLD_LAZY), dlclose_deleter};
    if (!plugin_handle)
    {
        std::cerr << "Failed to load plugin library: " << dlerror() << std::endl;
        return EXIT_FAILURE;
    }

    // 2. Create Runtime
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) return EXIT_FAILURE;

    // 3. Deserialize Engine
    std::ifstream engine_file(engine_file_path, std::ios::binary);
    if (!engine_file)
    {
        std::cerr << "Failed to open engine file." << std::endl;
        return EXIT_FAILURE;
    }
    engine_file.seekg(0, std::ios::end);
    size_t fsize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engineData(fsize);
    engine_file.read(engineData.data(), fsize);

    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine{
        runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr)};
    if (!engine) return EXIT_FAILURE;

    // 4. Create Context
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context{engine->createExecutionContext()};
    if (!context) return EXIT_FAILURE;

    // 5. Prepare Buffers
    int nbBindings = engine->getNbBindings();
    std::vector<void*> buffers(nbBindings);
    std::vector<float*> hostInputs;
    std::vector<float*> hostOutputs;
    std::vector<size_t> sizes;

    for (int i = 0; i < nbBindings; ++i)
    {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];
        sizes.push_back(vol);

        CHECK_CUDA_ERROR(cudaMalloc(&buffers[i], vol * sizeof(float)));

        float* hostMem = new float[vol];
        if (engine->bindingIsInput(i))
        {
            create_random_data(hostMem, vol);
            CHECK_CUDA_ERROR(cudaMemcpy(buffers[i], hostMem, vol * sizeof(float), cudaMemcpyHostToDevice));
            hostInputs.push_back(hostMem);
        }
        else
        {
            hostOutputs.push_back(hostMem);
        }

        std::cout << "Binding " << i << ": " << (engine->bindingIsInput(i) ? "Input" : "Output")
                  << ", Volume: " << vol << std::endl;
    }

    // 6. Run Inference
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    context->enqueueV2(buffers.data(), stream, nullptr);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 7. Copy back outputs
    int outIdx = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (!engine->bindingIsInput(i))
        {
            CHECK_CUDA_ERROR(cudaMemcpy(hostOutputs[outIdx], buffers[i], sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
            outIdx++;
        }
    }

    // 8. Verify scalar diff
    if (!hostOutputs.empty())
    {
        float max_diff = hostOutputs[0][0]; // attn_max_diff
        std::cout << "attn_max_diff = " << max_diff << std::endl;
        if (max_diff >= 1e-6f)
        {
            std::cout << "Verification FAILED: Difference >= 1e-6" << std::endl;
            return EXIT_FAILURE;
        }
        else
        {
            std::cout << "Verification PASSED: Difference below 1e-6" << std::endl;
        }
    }

    // 9. Cleanup
    cudaStreamDestroy(stream);
    for (auto* p : hostInputs) delete[] p;
    for (auto* p : hostOutputs) delete[] p;
    for (auto* p : buffers) cudaFree(p);

    return EXIT_SUCCESS;
}
