import torch
import mlflow.pytorch
import torchvision.models as models
import os

# Load model from MLflow
model_uri = "runs:/a9f25199ee7b4d78a7bae9df958e39e7/resnet50_custom_model"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
onnx_path = "resnet50_custom_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ Exported model to {onnx_path}")