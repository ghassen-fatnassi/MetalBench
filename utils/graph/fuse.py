import onnx
from onnx import helper, TensorProto, numpy_helper
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
model_path = "Models/yolo12n_op12_static_1_640.onnx"
model = onnx.load(model_path)
graph = model.graph
nodes = list(graph.node)

# Build initializer map ONCE
initializer_map = {init.name: init for init in graph.initializer}

# ------------------------------------------------------------
# Step 1: Detect attention blocks (unchanged)
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
# Step 2: Fuse blocks + FIX ATTRIBUTES
# ------------------------------------------------------------
new_nodes = []
fusion_report = []
used_initializers = set()

existing_names = set(node.name for node in nodes)
attn_counter = 0

for block in blocks:
    start_node = block[0]
    end_node = block[-1]

    input_names = list(start_node.input)
    output_names = list(end_node.output)

    fused_name = f"custom.attn_{attn_counter}"
    while fused_name in existing_names:
        attn_counter += 1
        fused_name = f"custom.attn_{attn_counter}"
    existing_names.add(fused_name)
    attn_counter += 1

    fused_node = helper.make_node(
        op_type="FusedAttnOp",
        inputs=input_names,
        outputs=output_names,
        name=fused_name,
        domain=""  # TRT plugin domain
    )

    # --------------------------------------------------------
    # 🔧 FIX: collect ALL weights/constants from the block
    # --------------------------------------------------------
    block_initializers = {}

    for n in block:
        for inp in n.input:
            if inp in initializer_map:
                block_initializers[inp] = initializer_map[inp]
                used_initializers.add(inp)

    # Attach them as REAL tensor attributes
    for name, tensor in block_initializers.items():
        fused_node.attribute.append(
            helper.make_attribute(name, tensor)
        )

    new_nodes.append((start_node.name, fused_node))

    fusion_report.append({
        "fused_op_name": fused_name,
        "original_nodes": [n.name for n in block],
        "weights": list(block_initializers.keys()),
        "inputs": [
            {"name": inp, **get_tensor_info(inp, model)}
            for inp in input_names
        ],
        "outputs": [
            {"name": out, **get_tensor_info(out, model)}
            for out in output_names
        ]
    })

# ------------------------------------------------------------
# Step 3: Replace nodes (unchanged logic)
# ------------------------------------------------------------
all_block_names = {n.name for block in blocks for n in block}
final_nodes = []
replaced = set()

for n in nodes:
    if n.name in all_block_names:
        if n.name not in replaced:
            for start_name, fused_node in new_nodes:
                if n.name == start_name:
                    final_nodes.append(fused_node)
                    for block in blocks:
                        if n in block:
                            replaced.update(x.name for x in block)
                            break
    else:
        final_nodes.append(n)

graph.ClearField("node")
graph.node.extend(final_nodes)

# ------------------------------------------------------------
# 🔧 FIX: remove consumed initializers from graph
# ------------------------------------------------------------
remaining_initializers = [
    init for init in graph.initializer
    if init.name not in used_initializers
]

graph.ClearField("initializer")
graph.initializer.extend(remaining_initializers)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
onnx.save(model, "model_fused.onnx")
print("✅ Fused model saved → model_fused.onnx")

with open("fusion_report.json", "w") as f:
    json.dump(fusion_report, f, indent=4)

print("✅ Fusion report saved → fusion_report.json")
print(f"Total blocks fused: {len(blocks)}")
