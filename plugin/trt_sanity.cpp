#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <iostream>

using namespace nvinfer1;

// Dummy plugin just to test ABI + headers
class SanityPlugin : public IPluginV2IOExt {
public:
    const char* getPluginType() const noexcept override { return "SanityPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    Dims getOutputDimensions(int, const Dims*, int) noexcept override { return Dims3{1,1,1}; }
    bool supportsFormatCombination(int, const PluginTensorDesc*, int, int) noexcept override { return true; }
    void configurePlugin(const PluginTensorDesc*, int, const PluginTensorDesc*, int) noexcept override {}
    size_t getWorkspaceSize(int) const noexcept override { return 0; }
    int enqueue(int, const void* const*, void* const*, void*, cudaStream_t) noexcept override { return 0; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void destroy() noexcept override { delete this; }
    IPluginV2IOExt* clone() const noexcept override { return new SanityPlugin(); }
    void setPluginNamespace(const char*) noexcept override {}
    const char* getPluginNamespace() const noexcept override { return ""; }
    DataType getOutputDataType(int, const DataType*, int) const noexcept override { return DataType::kFLOAT; }
    bool isOutputBroadcastAcrossBatch(int, const bool*, int) const noexcept override { return false; }
    bool canBroadcastInputAcrossBatch(int) const noexcept override { return false; }
};

int main() {
    std::cout << "TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    // Check plugin registry exists (article depends on this)
    auto* registry = getPluginRegistry();
    if (!registry) {
        std::cerr << "Plugin registry NOT available\n";
        return 1;
    }

    std::cout << "Plugin registry OK\n";

    // Check ONNX parser symbols exist
    auto builder = createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(0);
    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser) {
        std::cerr << "ONNX parser NOT available\n";
        return 1;
    }

    std::cout << "ONNX parser OK\n";

    parser->destroy();
    network->destroy();
    builder->destroy();

    std::cout << "TRT 7 plugin ABI is usable\n";
    return 0;
}
