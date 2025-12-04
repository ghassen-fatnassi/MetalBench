import onnx

# Load the ONNX model
model_path = "Models/yolo12n_op12.onnx"
model = onnx.load(model_path)

# Check the model
onnx.checker.check_model(model)

# Define the substring(s) to search for
search_terms = ["attn"]

print(f"Filtered nodes containing {search_terms}")
for i, node in enumerate(model.graph.node):
    # Combine node name, op_type, and input/output names into a single string
    combined = node.name + " " + node.op_type 
    
    # If any search term appears in the combined string, print it
    if any(term.lower() in combined.lower() for term in search_terms):
        print(f"{i+1:03d}: {node.op_type} | name: {node.name}")
