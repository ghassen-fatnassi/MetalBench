import onnx
from onnx import helper


def list_ops(model_path: str):
    model = onnx.load(model_path)
    ops = []
    for node in model.graph.node:
        ops.append((node.op_type, node.domain, node.name))
    return ops


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python list_ops.py model.onnx")
        exit(1)

    model_path = 'Models/model_fused.onnx'
    ops = list_ops(model_path)
    for op_type, domain, name in ops:
        if "attn" in domain:
            print(f"op_type: {op_type} | name: {name} | domain: {domain if domain else 'ai.onnx'}")
