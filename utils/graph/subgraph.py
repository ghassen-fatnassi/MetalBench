import onnx
from onnx import helper, TensorProto
import json

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12_static_1_640.onnx"
OUTPUT_MODEL_PATH = "model_attn_validation.onnx"
REPORT_PATH = "fusion_report.json"

ATTN_SCALE = 0.1767766922712326
TARGET_BLOCK_INDEX = 0
INPUT_SHAPE = [1, 64, 40, 40]  # fixed shape

# ================= LOAD MODEL =================
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

# ================= BLOCK IO =================
start = target_block[0]
end = target_block[-1]

block_input = start.input[0]
block_output = end.output[0]

ref_output = block_output + "_ref"
fused_output = block_output + "_fused"
end.output[0] = ref_output

# ================= CONV PARAMS =================
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

# ================= FUSED ATTENTION NODE =================
fused_inputs = [block_input]
for w, b in conv_params:
    fused_inputs.extend([w, b])

fused_node = helper.make_node(
    "FusedAttnOp",
    fused_inputs,
    [fused_output],
    name="custom.attn"
)
fused_node.attribute.append(helper.make_attribute("attn_scale", ATTN_SCALE))

# ================= DIFF + REDUCE =================
sub_node = helper.make_node("Sub", [ref_output, fused_output], ["attn_diff_raw"], name="attn_diff_sub")
abs_node = helper.make_node("Abs", ["attn_diff_raw"], ["attn_diff_abs"], name="attn_diff_abs")
reduce_node = helper.make_node("ReduceMax", ["attn_diff_abs"], ["attn_max_diff"], name="attn_diff_max", keepdims=0)

# ================= BUILD MINIMAL GRAPH =================
# collect only the initializers used by the block
used_input_names = []
for n in target_block:
    used_input_names.extend(list(n.input))

used_initializers = [initializer_map[i] for i in initializer_map if i in used_input_names]

new_graph = helper.make_graph(
    nodes=[*target_block, fused_node, sub_node, abs_node, reduce_node],
    name="attn_validation_graph",
    inputs=[helper.make_tensor_value_info(block_input, TensorProto.FLOAT, INPUT_SHAPE)],
    outputs=[helper.make_tensor_value_info("attn_max_diff", TensorProto.FLOAT, [])],
    initializer=used_initializers
)

# ================= ADD VALUE INFO FOR VISUALIZATION =================
# Fused output same shape as input
new_graph.value_info.extend([
    helper.make_tensor_value_info(fused_output, TensorProto.FLOAT, INPUT_SHAPE),
    helper.make_tensor_value_info(ref_output, TensorProto.FLOAT, INPUT_SHAPE),
    helper.make_tensor_value_info("attn_diff_raw", TensorProto.FLOAT, INPUT_SHAPE),
    helper.make_tensor_value_info("attn_diff_abs", TensorProto.FLOAT, INPUT_SHAPE),
    helper.make_tensor_value_info("attn_max_diff", TensorProto.FLOAT, [])
])

new_model = helper.make_model(new_graph, producer_name="attn_validation")

# preserve original opset safely
while len(new_model.opset_import) > 0:
    del new_model.opset_import[0]
for o in model.opset_import:
    new_model.opset_import.append(o)

# ================= SAVE =================
onnx.save(new_model, OUTPUT_MODEL_PATH)

fusion_report = {
    "target_block_index": TARGET_BLOCK_INDEX,
    "fused_op": fused_node.name,
    "inputs": fused_inputs,
    "ref_output": ref_output,
    "fused_output": fused_output,
    "attn_scale": ATTN_SCALE
}

with open(REPORT_PATH, "w") as f:
    json.dump(fusion_report, f, indent=4)

print("✅ Minimal attention validation model generated")
print("📤 Output tensor: attn_max_diff")
