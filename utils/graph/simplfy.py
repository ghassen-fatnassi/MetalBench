import onnx
import onnxoptimizer  # separate package for optimizer

def simplify_reshape_concat_reshape(model_path: str, output_path: str):
    # Load model
    model = onnx.load(model_path)


    # Optimizer passes
    passes = [
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "fuse_consecutive_transposes",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_unsqueezes",
        "eliminate_deadend",
        "eliminate_identity",
        "fuse_add_bias_into_conv"
    ]

    # Apply optimizer
    optimized_model = onnxoptimizer.optimize(model, passes)

    # Keep opset 12
    if optimized_model.opset_import[0].version != 12:
        optimized_model.opset_import[0].version = 12

    # Validate model
    onnx.checker.check_model(optimized_model)

    # Save optimized model
    onnx.save(optimized_model, output_path)
    print(f"Simplified model saved to {output_path}, opset {optimized_model.opset_import[0].version}")

if __name__ == "__main__":
    simplify_reshape_concat_reshape("Models/yolo12n_op12.onnx", "model_simplified.onnx")
