#ifndef TRT_FUSED_ATTN_PLUGIN_H
#define TRT_FUSED_ATTN_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace custom
{

// Updated launcher signature - matches the CUDA kernel
int computeFusedAttn(
    cudaStream_t stream,
    int batchSize,      // N
    int height,         // H (e.g., 40)
    int width,          // W (e.g., 40)
    const float* input,         // (N, 64, H, W)
    const float* qkv_weights,   // (192, 64, 1, 1)
    const float* qkv_bias,      // (192,)
    const float* pe_weights,    // (64, 1, 7, 7)
    const float* pe_bias,       // (64,)
    const float* proj_weights,  // (64, 64, 1, 1)
    const float* proj_bias,     // (64,)
    float* output,              // (N, 64, H, W)
    void* workspace);

// Helper function to calculate required workspace size
size_t getWorkspaceSize(int batchSize, int height, int width);

class FusedAttnPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedAttnPlugin(const std:: string name);
    FusedAttnPlugin(const std::string name, const void* data, size_t length);

    FusedAttnPlugin() = delete;

    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1:: PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1:: PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1:: PluginTensorDesc* inputDesc, const nvinfer1:: PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    nvinfer1:: DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private: 
    const std:: string mLayerName;
    std:: string mNamespace;
};

class FusedAttnPluginCreator :  public nvinfer1::IPluginCreator
{
public:
    FusedAttnPluginCreator();

    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const nvinfer1::PluginFieldCollection* getFieldNames() override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std:: string mNamespace;
};

} // namespace custom

#endif // TRT_FUSED_ATTN_PLUGIN_H