#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent / "train"))
from model import LeNet5


def load_pytorch_model(model_path):
    device = torch.device("cpu")
    model = LeNet5()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_to_onnx(model, output_path):
    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=True,
    )


def remove_initializers_from_inputs(onnx_model):
    initializers = [init.name for init in onnx_model.graph.initializer]
    inputs_to_remove = []

    for i, input in enumerate(onnx_model.graph.input):
        if input.name in initializers:
            inputs_to_remove.append(i)

    for i in reversed(inputs_to_remove):
        del onnx_model.graph.input[i]

    return onnx_model


def cleanup_external_data(output_dir):
    onnx_data_file = output_dir / "model.onnx.data"
    if onnx_data_file.exists():
        onnx_data_file.unlink()


def validate_onnx_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def test_inference(model_path):
    session = ort.InferenceSession(model_path)
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    outputs = session.run(["output"], {"input": dummy_input})
    return outputs


def get_model_size_mb(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)


def convert():
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "best_model.pt"
    output_dir = project_root / "web"
    output_path = output_dir / "model.onnx"

    output_dir.mkdir(exist_ok=True, parents=True)

    model = load_pytorch_model(model_path)
    export_to_onnx(model, output_path)

    onnx_model = validate_onnx_model(output_path)
    onnx_model = remove_initializers_from_inputs(onnx_model)
    onnx.save(onnx_model, output_path)

    cleanup_external_data(output_dir)
    test_inference(output_path)

    size_mb = get_model_size_mb(output_path)

    print(f"✅ Model exported to {output_path}")
    print(f"📦 Model size: {size_mb:.2f} MB")
    print("✅ ONNX model is valid")
    print("✅ ONNX Runtime inference works")


if __name__ == "__main__":
    convert()
