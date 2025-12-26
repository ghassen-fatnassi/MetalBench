/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
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
    void operator()(T* obj) const
    {
        if (obj) obj->destroy();
    }
};

void create_random_data(float* data, size_t const size, unsigned int seed = 1U)
{
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = dis(eng);
    }
}

bool all_close(float const* a, float const* b, size_t size, float rtol = 1e-5f, float atol = 1e-5f)
{
    bool passed = true;
    for (size_t i = 0; i < size; ++i)
    {
        float const diff = std::abs(a[i] - b[i]);
        if (diff > (atol + rtol * std::abs(b[i])))
        {
            std::cout << "Mismatch at [" << i << "]: " << a[i] << " vs " << b[i] << std::endl;
            passed = false;
            // return false; // Uncomment to stop at first error
        }
    }
    return passed;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <plugin_library_path> <engine_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const plugin_library_path{argv[1]};
    std::string const engine_file_path{argv[2]};

    CustomLogger logger{};

    // 1. Load Plugin Library
    auto dlclose_deleter = [](void* handle) { dlclose(handle); };
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

    // 5. Prepare Buffers (Legacy API: getNbBindings)
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

        // Allocate Device Memory
        CHECK_CUDA_ERROR(cudaMalloc(&buffers[i], vol * sizeof(float)));

        // Allocate Host Memory
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

    // enqueueV2 is the correct API for explicit batch in older TRT
    context->enqueueV2(buffers.data(), stream, nullptr);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 7. Verify Results (Identity Check)
    // Copy back outputs
    int outIdx = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (!engine->bindingIsInput(i))
        {
            CHECK_CUDA_ERROR(cudaMemcpy(hostOutputs[outIdx], buffers[i], sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
            outIdx++;
        }
    }

    // Assuming 1 Input and 1 Output for verification
    if (!hostInputs.empty() && !hostOutputs.empty())
    {
        std::cout << "Verifying output..." << std::endl;
        if (all_close(hostInputs[0], hostOutputs[0], sizes[0]))
        {
            std::cout << "Verification PASSED: Output matches Input (Identity)." << std::endl;
        }
        else
        {
            std::cout << "Verification FAILED." << std::endl;
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    for (auto* p : hostInputs) delete[] p;
    for (auto* p : hostOutputs) delete[] p;
    for (auto* p : buffers) cudaFree(p);

    return EXIT_SUCCESS;
}