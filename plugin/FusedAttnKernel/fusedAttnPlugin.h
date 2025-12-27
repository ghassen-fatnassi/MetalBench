#ifndef TRT_FUSED_ATTN_PLUGIN_H
#define TRT_FUSED_ATTN_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace custom
{

// Updated launcher signature
int computeFusedAttn(cudaStream_t stream, 
    int batchSize, int height,int width, int hiddenDim, 
    const float* input, 
    const float* w1, const float* b1, // Proj 1 (QKV)
    const float* w2, const float* b2, // DW Conv
    const float* w3, const float* b3, // Proj 2 (Out)
    float* output, 
    void* workspace, 
    float attnScale);

class FusedAttnPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    FusedAttnPlugin(const std::string name, float attnScale);
    FusedAttnPlugin(const std::string name, const void* data, size_t length);

    FusedAttnPlugin() = delete;

    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

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
    const std::string mLayerName;
    std::string mNamespace;
    float mAttnScale;
};

class FusedAttnPluginCreator : public nvinfer1::IPluginCreator
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
    std::string mNamespace;
};

} // namespace custom

#endif // TRT_FUSED_ATTN_PLUGIN_H