import onnx
from collections import defaultdict

# Load model
model_path = "Models/yolo12n_op12.onnx"
model = onnx.load(model_path)
graph = model.graph

# Step 1: Sort nodes in topological order (they're usually in order, but just in case)
nodes = list(graph.node)

# Step 2: Scan sequentially to detect attention blocks
blocks = []
current_block = []
in_attn = False

for node in nodes:
    if "attn" in node.name.lower():
        if not in_attn:
            # Starting a new attention block
            if current_block:
                blocks.append(current_block)
            current_block = [node]
            in_attn = True
        else:
            # Continue current attention block
            current_block.append(node)
    else:
        if in_attn:
            # Leaving attention region
            blocks.append(current_block)
            current_block = []
            in_attn = False
        # else: still outside, ignore

# If last block still exists
if current_block:
    blocks.append(current_block)

# Step 3: Assign custom op labels
thing_mapping = {}
for i, block_nodes in enumerate(blocks):
    thing_mapping[f"attn_block_{i}"] = [node.name for node in block_nodes]

# Step 4: Print summary
for block_name, nodes_list in thing_mapping.items():
    print(f"{block_name}: {len(nodes_list)} nodes, first node: {nodes_list[0]}")

# Optional: Save mapping
import json
with open("attn_block_mapping.json", "w") as f:
    json.dump(thing_mapping, f, indent=2)
