#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <dlfcn.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

// FIX 1: TensorRT objects must be destroyed via destroy(), not delete
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <onnx_file_path> <plugin_library_path> <engine_file_path>"
            << std::endl;
        return EXIT_FAILURE;
    }

    std::string const onnx_file_path{argv[1]};
    std::string const plugin_library_path{argv[2]};
    std::string const engine_file_path{argv[3]};

    std::cout << "ONNX file path: " << onnx_file_path << std::endl;
    std::cout << "Plugin library path: " << plugin_library_path << std::endl;
    std::cout << "Engine file path: " << engine_file_path << std::endl;

    CustomLogger logger{};

    // Create the builder.
    std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder{
        nvinfer1::createInferBuilder(logger)};
    if (builder == nullptr)
    {
        std::cerr << "Failed to create the builder." << std::endl;
        return EXIT_FAILURE;
    }

    // dlopen the plugin library using RAII.
    // The plugin will be registered automatically when the library is loaded.
    auto dlclose_deleter = [](void* handle) { dlclose(handle); };
    std::unique_ptr<void, decltype(dlclose_deleter)> plugin_handle{
        dlopen(plugin_library_path.c_str(), RTLD_LAZY), dlclose_deleter};
    
    if (plugin_handle == nullptr)
    {
        std::cerr << "Failed to load the plugin library: " << dlerror()
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Create the network.
    uint32_t flag{0U};
    if (getInferLibVersion() < 100000)
    {
        flag |= 1U << static_cast<uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    }
    std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network{
        builder->createNetworkV2(flag)};
    if (network == nullptr)
    {
        std::cerr << "Failed to create the network." << std::endl;
        return EXIT_FAILURE;
    }

    // Create the parser.
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{
        nvonnxparser::createParser(*network, logger)};
    if (parser == nullptr)
    {
        std::cerr << "Failed to create the parser." << std::endl;
        return EXIT_FAILURE;
    }
    parser->parseFromFile(
        onnx_file_path.c_str(),
        static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // Check input/output validity before setting formats
    if (network->getNbInputs() > 0 && network->getNbOutputs() > 0)
    {
        // Set the allowed IO tensor formats.
        uint32_t const formats{
            1U << static_cast<uint32_t>(nvinfer1::TensorFormat::kLINEAR)};
        nvinfer1::DataType const dtype{nvinfer1::DataType::kFLOAT};
        network->getInput(0)->setAllowedFormats(formats);
        network->getInput(0)->setType(dtype);
        network->getOutput(0)->setAllowedFormats(formats);
        network->getOutput(0)->setType(dtype);
    }
    else
    {
        std::cerr << "Network has no inputs or outputs!" << std::endl;
        // Proceeding might crash, but we'll let TRT handle the error
    }

    // Build the engine.
    std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config{
        builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "Failed to create the builder config." << std::endl;
        return EXIT_FAILURE;
    }

    // FIX 2: Use setMaxWorkspaceSize instead of setMemoryPoolLimit for older TRT compatibility
    config->setMaxWorkspaceSize(1U << 30); // 1 MB workspace, increase if needed (e.g., 1U << 30 for 1GB)
    
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // FIX 3: Use buildEngineWithConfig + serialize for older TRT compatibility
    // instead of buildSerializedNetwork
    std::cout << "Building engine..." << std::endl;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine{
        builder->buildEngineWithConfig(*network, *config)};
        
    if (engine == nullptr)
    {
        std::cerr << "Failed to build the engine." << std::endl;
        return EXIT_FAILURE;
    }

    std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serializedModel{
        engine->serialize()};

    if (serializedModel == nullptr)
    {
        std::cerr << "Failed to serialize the engine." << std::endl;
        return EXIT_FAILURE;
    }

    // Write the serialized engine to a file.
    std::ofstream engineFile{engine_file_path.c_str(), std::ios::binary};
    if (!engineFile.is_open())
    {
        std::cerr << "Failed to open the engine file." << std::endl;
        return EXIT_FAILURE;
    }
    engineFile.write(static_cast<char const*>(serializedModel->data()),
                     serializedModel->size());
    engineFile.close();

    std::cout << "Successfully serialized the engine to the file: "
              << engine_file_path << std::endl;

    return EXIT_SUCCESS;
}