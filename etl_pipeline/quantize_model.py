from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Input/output paths
model_fp32 = "resnet50_custom_model.onnx"
model_int8 = "resnet50_custom_model_quantized.onnx"

# Apply dynamic quantization
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

print(f"✅ Quantized model saved to {model_int8}")
print(f"Original size: {os.path.getsize(model_fp32)/1e6:.2f} MB")
print(f"Quantized size: {os.path.getsize(model_int8)/1e6:.2f} MB")
