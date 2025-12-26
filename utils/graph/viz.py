import json
from graphviz import Digraph

# Load JSON
with open("yolo12n_engine.json") as f:
    data = json.load(f)

dot = Digraph(comment="TensorRT Layer Graph", format="png")
dot.attr(rankdir='LR')  # left to right

# Add nodes
for d in data:
    if "name" in d and "averageMs" in d:
        name = d["name"].replace("/", "_").replace(" ", "_")[:60]  # safe for Graphviz
        label = f"{d['name']}\n{d['averageMs']:.2f} ms"
        # Color slow layers red
        color = "red" if d['averageMs'] > 1.0 else "lightblue"
        dot.node(name, label, style="filled", fillcolor=color)

# Optional: Add edges by naming convention
# This is crude: split names by '/' and connect sequentially
for d in data:
    if "name" in d:
        parts = d["name"].split("/")
        for i in range(len(parts)-1):
            src = "_".join(parts[:i+1])
            dst = "_".join(parts[:i+2])
            dot.edge(src, dst)

dot.render("trt_layer_graph")  # opens PNG automatically
