#include "fused_attention_plugin.h"
#include <cstdio>
#include <cuda_runtime.h>

int FusedAttentionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc*,
    const nvinfer1::PluginTensorDesc*,
    const void* const*,
    void* const*,
    void*,
    cudaStream_t)
{
    printf("🔥 FusedAttention enqueue called\n");
    return 0;
}

class FusedAttentionPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const override { return "FusedAttnOp"; }
    const char* getPluginVersion() const override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() override { return &mFC; }

    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) override {
        return new FusedAttentionPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, size_t length) override {
        return new FusedAttentionPlugin(data, length);
    }

    void setPluginNamespace(const char* ns) override { mNamespace = ns ? ns : ""; }
    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
};

REGISTER_TENSORRT_PLUGIN(FusedAttentionPluginCreator);