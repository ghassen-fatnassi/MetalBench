import onnx
from onnx import shape_inference
import sys

# Usage: python3 fix_model.py Models/yolo12n_op12.onnx
if len(sys.argv) < 2:
    print("Usage: python3 fix_model.py <path_to_onnx_model>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = input_path.replace(".onnx", "_shaped.onnx")

print(f"Loading {input_path}...")
model = onnx.load(input_path)

# 1. Check the model
print("Checking model consistency...")
onnx.checker.check_model(model)

# 2. Apply Shape Inference (The Fix)
print("Applying shape inference...")
# This forces the calculation of all intermediate tensor shapes
inferred_model = shape_inference.infer_shapes(model)

# 3. Save the new model
onnx.save(inferred_model, output_path)
print(f"Success! Saved fixed model to: {output_path}")