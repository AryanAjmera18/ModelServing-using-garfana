import torch
from src.dataset import EyeDataset
from src.model import build_model
from torch.utils.data import DataLoader
import mlflow

# Load model
model = build_model(num_classes=10)
model = mlflow.pytorch.load_model("runs:/a9f25199ee7b4d78a7bae9df958e39e7/resnet50_custom_model")
model.eval()

# Load sample data
dataset = EyeDataset("data/processed")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

with torch.no_grad():
    for i, (image, label) in enumerate(dataloader):
        output = model(image)
        predicted = torch.argmax(output, dim=1)
        print(f"Sample {i+1}: Predicted={predicted.item()}, Actual={label.item()}")
        if i == 9:
            break  # Show 10 samples only

print("✅ Evaluation completed.")