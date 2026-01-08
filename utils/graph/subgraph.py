import onnx
from onnx import helper, TensorProto
import os

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx"
REF_MODEL_PATH = "model_ref.onnx"
FUSED_MODEL_PATH = "model_fused.onnx"

ATTN_SCALE = 0.1767766922712326
TARGET_BLOCK_INDEX = 0
INPUT_SHAPE = [1, 64, 40, 40]  # Input shape for the block

# ================= LOAD MODEL =================
print(f"Loading {MODEL_PATH}...")
model = onnx.load(MODEL_PATH)
graph = model.graph
nodes = list(graph.node)
initializer_map = {i.name: i for i in graph.initializer}

# ================= DETECT ATTENTION BLOCKS =================
blocks = []
current_block = []
in_attn = False
for node in nodes:
    if "attn" in node.name.lower():
        if not in_attn:
            current_block = [node]
            in_attn = True
        else:
            current_block.append(node)
    else:
        if in_attn:
            blocks.append(current_block)
            current_block = []
            in_attn = False
if current_block:
    blocks.append(current_block)

target_block = blocks[TARGET_BLOCK_INDEX]
print(f"Found {len(blocks)} attention blocks. Targeting index {TARGET_BLOCK_INDEX} with {len(target_block)} nodes.")

# ================= BLOCK IO =================
start_node = target_block[0]
end_node = target_block[-1]

block_input_name = start_node.input[0]
block_output_name = end_node.output[0]

# ================= GATHER INITIALIZERS =================
# We need the weights specifically for this block
used_input_names = set()
for n in target_block:
    for inp in n.input:
        used_input_names.add(inp)

block_initializers = [initializer_map[name] for name in used_input_names if name in initializer_map]

# ================= 1. GENERATE REFERENCE MODEL =================
# This model contains the original decomposed graph
ref_graph = helper.make_graph(
    nodes=target_block,
    name="reference_graph",
    inputs=[helper.make_tensor_value_info(block_input_name, TensorProto.FLOAT, INPUT_SHAPE)],
    outputs=[helper.make_tensor_value_info(block_output_name, TensorProto.FLOAT, INPUT_SHAPE)],
    initializer=block_initializers
)

ref_model = helper.make_model(ref_graph, producer_name="ref_generator")
ref_model.opset_import.extend(model.opset_import)

onnx.save(ref_model, REF_MODEL_PATH)
print(f"✅ Saved Reference Model: {REF_MODEL_PATH}")


# ================= 2. GENERATE FUSED MODEL =================
# This model replaces the block with FusedAttnOp

# Extract Conv weights in order (QKV, PE, Proj)
conv_params = []
for n in target_block:
    if n.op_type == "Conv":
        w = b = None
        for inp in n.input:
            if inp in initializer_map:
                t = initializer_map[inp]
                if len(t.dims) == 4:
                    w = t.name
                elif len(t.dims) == 1:
                    b = t.name
        conv_params.append((w, b))

# Construct inputs for FusedOp: [Input, W1, B1, W2, B2, W3, B3]
fused_inputs = [block_input_name]
for w, b in conv_params:
    fused_inputs.extend([w, b])

fused_node = helper.make_node(
    "FusedAttnOp",
    fused_inputs,
    ["fused_output"],
    name="custom.attn",
    attn_scale=ATTN_SCALE
)

fused_graph = helper.make_graph(
    nodes=[fused_node],
    name="fused_graph",
    inputs=[helper.make_tensor_value_info(block_input_name, TensorProto.FLOAT, INPUT_SHAPE)],
    outputs=[helper.make_tensor_value_info("fused_output", TensorProto.FLOAT, INPUT_SHAPE)],
    initializer=block_initializers # We can pass the same initializers, unused ones are ignored usually
)

fused_model = helper.make_model(fused_graph, producer_name="fused_generator")
fused_model.opset_import.extend(model.opset_import)

onnx.save(fused_model, FUSED_MODEL_PATH)
print(f"✅ Saved Fused Model: {FUSED_MODEL_PATH}")