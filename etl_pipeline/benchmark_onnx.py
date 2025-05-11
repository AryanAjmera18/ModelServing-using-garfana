import torch
import onnxruntime as ort
import time
import numpy as np
import os
import mlflow.pytorch

# Load PyTorch model from MLflow
model_uri = "runs:/a9f25199ee7b4d78a7bae9df958e39e7/resnet50_custom_model"
pt_model = mlflow.pytorch.load_model(model_uri)
pt_model.eval()

# Load ONNX model
onnx_model_path = "resnet50_custom_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# Generate random input
dummy_input = torch.randn(1, 3, 224, 224)
numpy_input = dummy_input.numpy()

# Benchmark PyTorch
with torch.no_grad():
    torch_times = []
    for _ in range(50):
        start = time.time()
        pt_model(dummy_input)
        torch_times.append(time.time() - start)

# Benchmark ONNX
onnx_times = []
for _ in range(50):
    start = time.time()
    session.run(None, {input_name: numpy_input})
    onnx_times.append(time.time() - start)

# Model sizes
pytorch_size = os.path.getsize("resnet50_custom_model.pth") / (1024 ** 2) if os.path.exists("resnet50_custom_model.pth") else 0
onnx_size = os.path.getsize(onnx_model_path) / (1024 ** 2)

# Print results
print("\n===== Benchmark Results =====")
print(f"PyTorch - Avg Inference Time: {np.mean(torch_times)*1000:.2f} ms")
print(f"ONNX    - Avg Inference Time: {np.mean(onnx_times)*1000:.2f} ms")
print(f"PyTorch Model Size: {pytorch_size:.2f} MB")
print(f"ONNX Model Size:    {onnx_size:.2f} MB")
