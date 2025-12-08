import onnx

# Load the ONNX model
model_path = "Models/yolo12n_op12.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

# Define the substring(s) to filter
search_terms = ["attn"]

print("Filtered nodes related to '/attn/':")
for i, node in enumerate(model.graph.node):
    combined = " ".join([node.name, node.op_type, *node.input, *node.output])
    if any(term.lower() in combined.lower() for term in search_terms):
        print(f"{i+1:03d}: {node.op_type} | name: {node.name}\n")
