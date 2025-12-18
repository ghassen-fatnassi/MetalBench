import onnx
import os

def fix_onnx_shapes(input_model_path, batch, res):
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph

    # 1. Fix the Input Shape
    # Most YOLO models have one input. We'll find it and overwrite it.
    for input_node in graph.input:
        dim = input_node.type.tensor_type.shape.dim
        dim[0].dim_value = batch   # Set Batch
        dim[1].dim_value = 3       # Set Channels (RGB)
        dim[2].dim_value = res     # Set Height
        dim[3].dim_value = res     # Set Width
        print(f"Fixed input '{input_node.name}' to {batch}x3x{res}x{res}")

    # 2. Fix the Output Shapes
    # TensorRT 7 needs the output shapes defined to allocate memory
    for output_node in graph.output:
        dim = output_node.type.tensor_type.shape.dim
        # Note: YOLO outputs usually look like [Batch, Boxes, Elements] 
        # or [Batch, Channels, H, W]. We only fix the Batch (index 0).
        if len(dim) > 0:
            dim[0].dim_value = batch
        print(f"Fixed output '{output_node.name}' batch to {batch}")

    # 3. Generate filename
    output_name = f"yolo12n_op12_static_{batch}_{res}.onnx"
    output_path = os.path.join("Models", output_name)
    
    # Ensure directory exists
    os.makedirs("Models", exist_ok=True)

    # Save the model
    onnx.save(model, output_path)
    print(f"Successfully saved: {output_path}")

# ================= CONFIGURATION =================
INPUT_MODEL = "Models/yolo12n_op12.onnx"
BATCHES = [1, 2, 4, 8]
RESOLUTIONS = [128, 256, 384, 512, 640]

if __name__ == "__main__":
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: {INPUT_MODEL} not found!")
    else:
        for b in BATCHES:
            for r in RESOLUTIONS:
                fix_onnx_shapes(INPUT_MODEL, b, r)
        print("\nAll static models generated successfully.")