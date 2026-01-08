#include "fusedAttnPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

using namespace nvinfer1;

namespace custom
{

namespace
{
static const char* FUSED_ATTN_PLUGIN_VERSION{"1"};
static const char* FUSED_ATTN_PLUGIN_NAME{"FusedAttnOp"}; 

// Constants matching the CUDA kernel
constexpr int NUM_HEADS = 2;
constexpr int HEAD_DIM = 32;
constexpr int HIDDEN_DIM = 64;  // NUM_HEADS * HEAD_DIM
constexpr int QKV_DIM = 192;    // 3 * HIDDEN_DIM

} // namespace

// Static class fields initialization
PluginFieldCollection FusedAttnPluginCreator::mFC{};
std::vector<PluginField> FusedAttnPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedAttnPluginCreator);

FusedAttnPlugin::FusedAttnPlugin(const std::string name)
    : mLayerName(name)
{
}

FusedAttnPlugin::FusedAttnPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // No parameters to deserialize since attnScale is computed from HEAD_DIM
}

nvinfer1::IPluginV2DynamicExt* FusedAttnPlugin::clone() const
{
    auto plugin = new FusedAttnPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace. c_str());
    return plugin;
}

nvinfer1::DimsExprs FusedAttnPlugin:: getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1:: IExprBuilder& exprBuilder)
{
    // Output shape matches input shape (N, C, H, W)
    return inputs[0];
}

bool FusedAttnPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // 7 inputs (Data + 6 weights/bias), 1 output. 
    // Ensure everything is Float32 and Linear
    return (inOut[pos].type == DataType::kFLOAT) && (inOut[pos]. format == TensorFormat::kLINEAR);
}

void FusedAttnPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
}

size_t FusedAttnPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1:: PluginTensorDesc* outputs, int nbOutputs) const
{
    // Input is N, C, H, W
    int n = inputs[0]. dims. d[0];
    int h = inputs[0]. dims.d[2];
    int w = inputs[0]. dims.d[3];
    
    // Safety check for dynamic dims (-1)
    if (n < 0 || h < 0 || w < 0) return 0; // Runtime will call again with real dims

    // Use the helper function from the CUDA kernel
    return custom::getWorkspaceSize(n, h, w);
}

int FusedAttnPlugin::enqueue(const nvinfer1:: PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    int n = inputDesc[0].dims. d[0];
    int h = inputDesc[0].dims. d[2];
    int w = inputDesc[0].dims.d[3];

    // Inputs layout: 
    // 0: Data          (N, 64, H, W)
    // 1: qkv_weights   (192, 64, 1, 1)
    // 2: qkv_bias      (192,)
    // 3: pe_weights    (64, 1, 7, 7)
    // 4: pe_bias       (64,)
    // 5: proj_weights  (64, 64, 1, 1)
    // 6: proj_bias     (64,)
    
    return computeFusedAttn(stream, 
        n, h, w,
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        static_cast<const float*>(inputs[4]),
        static_cast<const float*>(inputs[5]),
        static_cast<const float*>(inputs[6]),
        static_cast<float*>(outputs[0]),
        workspace);
}

nvinfer1::DataType FusedAttnPlugin::getOutputDataType(
    int index, const nvinfer1:: DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

const char* FusedAttnPlugin::getPluginType() const { return FUSED_ATTN_PLUGIN_NAME; }
const char* FusedAttnPlugin::getPluginVersion() const { return FUSED_ATTN_PLUGIN_VERSION; }
int FusedAttnPlugin::getNbOutputs() const { return 1; }
int FusedAttnPlugin::initialize() { return 0; }
void FusedAttnPlugin:: terminate() {}
size_t FusedAttnPlugin::getSerializationSize() const { return 0; } // No parameters to serialize
void FusedAttnPlugin::serialize(void* buffer) const {} // No parameters to serialize
void FusedAttnPlugin::destroy() { delete this; }
void FusedAttnPlugin::setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }
const char* FusedAttnPlugin::getPluginNamespace() const { return mNamespace.c_str(); }

// Creator Implementation
FusedAttnPluginCreator::FusedAttnPluginCreator()
{
    mPluginAttributes.clear();
    // No plugin attributes needed - attnScale is derived from HEAD_DIM
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FusedAttnPluginCreator::getPluginName() const { return FUSED_ATTN_PLUGIN_NAME; }
const char* FusedAttnPluginCreator::getPluginVersion() const { return FUSED_ATTN_PLUGIN_VERSION; }
const PluginFieldCollection* FusedAttnPluginCreator:: getFieldNames() { return &mFC; }

IPluginV2* FusedAttnPluginCreator:: createPlugin(const char* name, const PluginFieldCollection* fc)
{
    return new FusedAttnPlugin(name);
}

IPluginV2* FusedAttnPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new FusedAttnPlugin(name, serialData, serialLength);
}

void FusedAttnPluginCreator:: setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }
const char* FusedAttnPluginCreator::getPluginNamespace() const { return mNamespace. c_str(); }

} // namespace custom