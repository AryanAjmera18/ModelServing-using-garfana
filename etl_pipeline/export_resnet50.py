# export_resnet50.py
import torch
import torchvision.models as models

dummy_input = torch.randn(1, 3, 224, 224)
model = models.resnet50(pretrained=True)
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "resnet50_custom_model.onnx",  # <- Save directly in the current directory
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
