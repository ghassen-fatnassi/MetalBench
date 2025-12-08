import onnx
from onnx import helper, TensorProto
import json

# ------------------------------------------------------------
# Utility: Get tensor shape + dtype with fallback
# ------------------------------------------------------------
def get_tensor_info(name, model):
    dtype = "FLOAT"
    channels = None

    def inspect(v):
        nonlocal dtype, channels
        t = v.type.tensor_type
        dtype = TensorProto.DataType.Name(t.elem_type)
        dims = [d.dim_value for d in t.shape.dim]
        for d in dims:
            if d > 1:
                channels = d
                break

    for v in model.graph.input:
        if v.name == name:
            inspect(v)
            break
    for v in model.graph.output:
        if v.name == name:
            inspect(v)
            break
    for v in model.graph.value_info:
        if v.name == name:
            inspect(v)
            break

    if channels is None:
        channels = 3

    return {
        "shape": ["batch", channels, "res", "res"],
        "dtype": dtype
    }

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model_path = "Models/yolo12n_op12.onnx"
model = onnx.load(model_path)
graph = model.graph
nodes = list(graph.node)

# ------------------------------------------------------------
# Step 1: Detect attention blocks
# ------------------------------------------------------------
blocks = []
current_block = []
in_attn = False

for node in nodes:
    if "attn" in node.name.lower():
        if not in_attn:
            if current_block:
                blocks.append(current_block)
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

# ------------------------------------------------------------
# Step 2: Fuse blocks into single nodes
# ------------------------------------------------------------
new_nodes = []
fusion_report = []

for i, block in enumerate(blocks):
    start_node = block[0]
    end_node = block[-1]

    input_names = list(start_node.input)
    output_names = list(end_node.output)

    fused_name = f"custom.attn_{i}"  # unique name
    fused_node = helper.make_node(
        op_type="FusedAttnOp",
        inputs=input_names,
        outputs=output_names,
        name=fused_name,
        domain="custom.attn"  # match your C++ registration
    )

    new_nodes.append((start_node.name, fused_node))  # store insertion point

    record = {
        "fused_op_name": fused_name,
        "original_nodes": [n.name for n in block],
        "inputs": [
            {
                "name": inp,
                **get_tensor_info(inp, model)
            }
            for inp in input_names
        ],
        "outputs": [
            {
                "name": out,
                **get_tensor_info(out, model)
            }
            for out in output_names
        ]
    }
    fusion_report.append(record)

# ------------------------------------------------------------
# Step 3: Replace nodes in graph preserving order
# ------------------------------------------------------------
all_block_names = {n.name for block in blocks for n in block}
final_nodes = []

for n in nodes:
    if n.name in all_block_names:
        # insert fused node at the position of the first node in the block
        for start_name, fused_node in new_nodes:
            if n.name == start_name:
                final_nodes.append(fused_node)
    elif n.name not in all_block_names:
        final_nodes.append(n)

graph.ClearField("node")
graph.node.extend(final_nodes)

# ------------------------------------------------------------
# Step 4: Save fused model
# ------------------------------------------------------------
onnx.save(model, "model_fused.onnx")
print("Fused model saved → model_fused.onnx")

# ------------------------------------------------------------
# Step 5: Save JSON metadata
# ------------------------------------------------------------
with open("fusion_report.json", "w") as f:
    json.dump(fusion_report, f, indent=4)

print("Fusion report saved → fusion_report.json")
