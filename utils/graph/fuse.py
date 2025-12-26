import onnx
from onnx import helper
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
# Step 2: Fuse blocks (CORRECT Conv handling)
# ------------------------------------------------------------
new_nodes = []
fusion_report = []
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

    # --------------------------------------------------------
    # Extract Conv params PER CONV (NO DUPLICATES)
    # --------------------------------------------------------
    conv_params = []

    for n in block:
        if n.op_type == "Conv":
            weight = None
            bias = None

            for inp in n.input:
                if inp in initializer_map:
                    tensor = initializer_map[inp]
                    if len(tensor.dims) == 4:
                        weight = tensor.name
                    elif len(tensor.dims) == 1:
                        bias = tensor.name

            assert weight is not None, f"{n.name} missing weight"
            assert bias is not None, f"{n.name} missing bias"

            conv_params.append((weight, bias))

    assert len(conv_params) == 3, "Expected exactly 3 Conv layers"

    # --------------------------------------------------------
    # Build fused inputs (EXACTLY 7)
    # --------------------------------------------------------
    fused_inputs = [start.input[0]]
    for w, b in conv_params:
        fused_inputs.append(w)
        fused_inputs.append(b)

    assert len(fused_inputs) == 7

    fused_node = helper.make_node(
        op_type="FusedattnopIPluginV2DynamicExt",
        inputs=fused_inputs,
        outputs=list(end.output),
        name=fused_name,
        domain=""
    )

    fused_node.attribute.append(
        helper.make_attribute("attn_scale", ATTN_SCALE)
    )

    new_nodes.append((start.name, fused_node))

    fusion_report.append({
        "fused_op": fused_name,
        "inputs": fused_inputs,
        "attn_scale": ATTN_SCALE
    })

# ------------------------------------------------------------
# Step 3: Replace nodes
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
# Step 4: SAFE initializer pruning
# ------------------------------------------------------------
used_inputs = set()
for n in graph.node:
    used_inputs.update(n.input)

graph.ClearField("initializer")
graph.initializer.extend(
    i for i in initializer_map.values()
    if i.name in used_inputs
)

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
onnx.save(model, "model_fused.onnx")
with open("fusion_report.json", "w") as f:
    json.dump(fusion_report, f, indent=4)

print("✅ Fusion done correctly")
print(f"Fused blocks: {len(blocks)}")
