import onnx
from onnx import helper, TensorProto
import json

ATTN_SCALE = 0.1767766922712326

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model_path = "Models/yolo12n_op12_static_1_640.onnx"
model = onnx.load(model_path)
graph = model.graph
nodes = list(graph.node)

initializer_map = {i.name: i for i in graph.initializer}

# ------------------------------------------------------------
# Step 1: Detect attention blocks (UNCHANGED)
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
# Step 2: Fuse blocks + STRICT attribute extraction
# ------------------------------------------------------------
new_nodes = []
fusion_report = []
used_initializers = set()
existing_names = set(n.name for n in nodes)

attn_counter = 0

for block in blocks:
    start = block[0]
    end = block[-1]

    fused_name = f"custom.attn_{attn_counter}"
    while fused_name in existing_names:
        attn_counter += 1
        fused_name = f"custom.attn_{attn_counter}"
    existing_names.add(fused_name)
    attn_counter += 1

    fused_node = helper.make_node(
        op_type="FusedAttnOp",
        inputs=list(start.input),
        outputs=list(end.output),
        name=fused_name,
        domain=""
    )

    # --------------------------------------------------------
    # 🔧 Extract ONLY 3 Conv weights + biases
    # --------------------------------------------------------
    conv_weights = []
    conv_biases = []

    for n in block:
        if n.op_type == "Conv":
            for inp in n.input:
                if inp in initializer_map:
                    tensor = initializer_map[inp]
                    used_initializers.add(inp)

                    if len(tensor.dims) == 4:
                        conv_weights.append(tensor)
                    elif len(tensor.dims) == 1:
                        conv_biases.append(tensor)

    assert len(conv_weights) == 3, "Expected exactly 3 Conv weights"
    assert len(conv_biases) == 3, "Expected exactly 3 Conv biases"

    # Attach attributes in deterministic order
    for i in range(3):
        fused_node.attribute.append(
            helper.make_attribute(f"conv{i}_weight", conv_weights[i])
        )
        fused_node.attribute.append(
            helper.make_attribute(f"conv{i}_bias", conv_biases[i])
        )

    # --------------------------------------------------------
    # 🔧 Attention scaling factor
    # --------------------------------------------------------
    fused_node.attribute.append(
        helper.make_attribute("attn_scale", ATTN_SCALE)
    )

    new_nodes.append((start.name, fused_node))

    fusion_report.append({
        "fused_op": fused_name,
        "convs": 3,
        "attn_scale": ATTN_SCALE
    })

# ------------------------------------------------------------
# Step 3: Replace nodes (UNCHANGED)
# ------------------------------------------------------------
all_block_names = {n.name for b in blocks for n in b}
final_nodes = []
replaced = set()

for n in nodes:
    if n.name in all_block_names:
        if n.name not in replaced:
            for start_name, fused in new_nodes:
                if n.name == start_name:
                    final_nodes.append(fused)
                    for b in blocks:
                        if n in b:
                            replaced.update(x.name for x in b)
                            break
    else:
        final_nodes.append(n)

graph.ClearField("node")
graph.node.extend(final_nodes)

# ------------------------------------------------------------
# Remove consumed initializers
# ------------------------------------------------------------
graph.ClearField("initializer")
graph.initializer.extend(
    i for i in initializer_map.values()
    if i.name not in used_initializers
)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
onnx.save(model, "model_fused.onnx")
with open("fusion_report.json", "w") as f:
    json.dump(fusion_report, f, indent=4)

print("✅ Fusion done correctly")
print(f"Fused blocks: {len(blocks)}")
