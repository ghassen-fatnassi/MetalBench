import onnx
from onnx import helper, TensorProto

# Load model
model_path = "Models/yolo12n_op12.onnx"
model = onnx.load(model_path)
graph = model.graph

# === Step 1: Identify blocks (reuse previous code) ===
nodes = list(graph.node)
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

# === Step 2: Fuse each block into a single node ===
new_nodes = []
for i, block in enumerate(blocks):
    start_node = block[0]
    end_node = block[-1]

    # Capture inputs from first node
    input_names = list(start_node.input)
    # Capture outputs from last node
    output_names = list(end_node.output)

    # Fuse into a single custom node
    fused_node = helper.make_node(
        op_type="FusedAttnOp",
        inputs=input_names,
        outputs=output_names,
        name=f"{start_node.name}_fused"
    )

    new_nodes.append(fused_node)

# === Step 3: Build new node list for the graph ===
all_block_node_names = [n.name for block in blocks for n in block]
final_nodes = [n for n in nodes if n.name not in all_block_node_names]
final_nodes.extend(new_nodes)

# Replace graph nodes
graph.ClearField("node")
graph.node.extend(final_nodes)

# Optional: you can attach input/output shape info as attributes if needed
for fused_node in new_nodes:
    # For simplicity, just store number of inputs/outputs
    fused_node.attribute.extend([
        helper.make_attribute("num_inputs", len(fused_node.input)),
        helper.make_attribute("num_outputs", len(fused_node.output))
    ])

# === Step 4: Save fused model ===
onnx.save(model, "model_fused.onnx")
print("Fused ONNX model saved as model_fused.onnx")
