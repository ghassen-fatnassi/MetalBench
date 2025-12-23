#ifndef TENSORRT_FUSED_ATTN_OP_PLUGIN_H
#define TENSORRT_FUSED_ATTN_OP_PLUGIN_H

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <NvInferRuntimePlugin.h>

constexpr char const* const kFUSED_ATTN_OP_PLUGIN_NAME{"FusedAttnOp"};
// "constexpr char const* const" means : compile time constant , basically a constant pointer that points to a constant string "FusedAttnOp"
constexpr char const* const kFUSED_ATTN_OP_PLUGIN_VERSION{"1"};

namespace nvinfer1
{
namespace plugin
{

struct FusedAttnOpParameters

} // namespace plugin
} // namespace nvinfer1
#endif //TENSORRT_FUSED_ATTN_OP_PLUGIN_H