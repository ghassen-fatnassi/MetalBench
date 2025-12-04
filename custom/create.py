import onnx
from onnx import helper, TensorProto

def build_custom_model(path="custom_model.onnx"):
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    # Custom node (no attributes for simplicity)
    custom_node = helper.make_node(
        "MyCustomOp",          # operator name
        ["input"],             # inputs
        ["output"],            # outputs
        domain="mydomain"      # REQUIRED: namespace for custom ops
    )

    graph = helper.make_graph(
        [custom_node],
        "CustomGraph",
        [input],
        [output]
    )

    model = helper.make_model(graph, producer_name="custom-op-test")
    onnx.save(model, path)
    print("Saved:", path)

build_custom_model()
