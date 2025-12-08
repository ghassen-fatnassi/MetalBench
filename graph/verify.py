import onnx
import numpy as np

def check_fused_model(model_path="Models/model_fused.onnx"):
    # Load the fused model
    model = onnx.load(model_path)
    
    # Find all custom ops in the model
    custom_ops = set()
    
    print("Looking for custom ops in the model...")
    for node in model.graph.node:
        if node.op_type != "AttentionBlock":  # Change this to what you expect
            if node.domain != "":  # Custom ops have non-empty domain
                custom_ops.add((node.op_type, node.domain))
    
    print("\nFound the following custom ops:")
    for op_type, domain in custom_ops:
        print(f"  Op Type: {op_type}, Domain: {domain}")
    
    # Also check what the actual attention block nodes are called
    print("\n\nLooking for attention-related nodes:")
    attention_nodes = []
    for node in model.graph.node:
        if "attn" in node.name.lower() or "attention" in node.name.lower():
            attention_nodes.append(node)
    
    if attention_nodes:
        print(f"Found {len(attention_nodes)} attention-related nodes")
        print("\nFirst few nodes:")
        for i, node in enumerate(attention_nodes[:5]):
            print(f"  {i}: {node.name} (op_type: {node.op_type}, domain: {node.domain})")
            print(f"     Inputs: {node.input}")
            print(f"     Outputs: {node.output}")
            print()
    else:
        print("No attention nodes found by name search")
    
    # Check model's opset imports
    print("\nModel opset imports:")
    for imp in model.opset_import:
        print(f"  Domain: '{imp.domain}', Version: {imp.version}")

if __name__ == "__main__":
    check_fused_model()