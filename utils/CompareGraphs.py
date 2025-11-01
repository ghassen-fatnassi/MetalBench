import os, json, csv
from pathlib import Path
from collections import Counter
import onnx, pandas as pd

root = Path("./Models")
out = Path("model_summaries"); out.mkdir(exist_ok=True)

def summarize(name, path, nodes, params):
    ops = [n.get("op_type", "Unknown") for n in nodes]
    return {
        "model": name,
        "file": path.name,
        "size_MB": round(path.stat().st_size/1024/1024,3),
        "nodes": len(nodes),
        "unique_ops": len(set(ops)),
        "top_ops": ", ".join([f"{k}:{v}" for k,v in Counter(ops).most_common(5)]),
        "params": params
    }

def onnx_summary(path):
    m = onnx.load(path)
    nodes = [{"op_type":n.op_type} for n in m.graph.node]
    pcount = sum([int(__import__("numpy").prod(t.dims)) for t in m.graph.initializer])
    return summarize(path.stem, path, nodes, pcount)

def xml_summary(xmlpath):
    from openvino.runtime import Core
    core = Core()
    model = core.read_model(xmlpath)
    nodes = [{"op_type":op.get_type_name()} for op in model.get_ops()]
    return summarize(xmlpath.stem, xmlpath, nodes, 0)

def torch_summary(path):
    import torch
    m = torch.jit.load(path, map_location="cpu")
    nodes = [{"op_type": str(n.kind())} for n in m.graph.nodes()]
    params = sum(p.numel() for p in m.parameters())
    return summarize(path.stem, path, nodes, params)

def tflite_summary(path):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()

    ops_details = interp._get_ops_details()
    ops = []

    for op in ops_details:
        # Handle different possible key names
        if "builtin_code" in op:
            op_type = op["builtin_code"]
        elif "op_name" in op:
            op_type = op["op_name"]
        elif "opcode_index" in op:
            # Try to map index back to the operator name if available
            opcode_details = interp._get_tensor_details()
            op_type = f"opcode_{op['opcode_index']}"
        else:
            op_type = "unknown"

        ops.append({"op_type": op_type})

    return {
        "type": "tflite",
        "path": str(path),
        "num_ops": len(ops),
        "unique_ops": len(set([o["op_type"] for o in ops])),
        "ops": ops[:10],  # preview
    }


def coreml_summary(path):
    import coremltools as ct
    m = ct.models.MLModel(path)
    spec = m.get_spec()
    nn = getattr(spec,"neuralNetwork",None) or getattr(spec,"neuralNetworkClassifier",None)
    ops=[{"op_type":l.WhichOneof("layer")} for l in nn.layers]
    return summarize(path.stem, path, ops, 0)

def ncnn_summary(param):
    lines = [l.strip() for l in open(param) if l.strip()]
    ops=[{"op_type":l.split()[0]} for l in lines if l[0].isalpha()]
    return summarize(param.stem, param, ops, 0)

summaries=[]

# torch
summaries.append(torch_summary(root/"yolo12n.torchscript"))

# onnx
summaries.append(onnx_summary(root/"yolo12n.onnx"))
summaries.append(onnx_summary(root/"yolo12n.pnnxsim.onnx"))

# openvino
summaries.append(xml_summary(root/"yolo12n_openvino_model"/"yolo12n.xml"))
summaries.append(xml_summary(root/"yolo12n_int8_openvino_model"/"yolo12n.xml"))

# tflite
summaries.append(tflite_summary(root/"yolo12n_TFLite_model"/"yolo12n_float32.tflite"))
summaries.append(tflite_summary(root/"yolo12n_TFLite_model"/"yolo12n_float16.tflite"))

# coreml
#summaries.append(coreml_summary(root/"yolo12n.coreml_model"/"Data"/"com.apple.CoreML"/"model.mlmodel"))

# ncnn
summaries.append(ncnn_summary(root/"yolo12n_ncnn_model"/"model.ncnn.param"))

pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
print(pd.DataFrame(summaries)[["model","size_MB","nodes","unique_ops","top_ops"]])
