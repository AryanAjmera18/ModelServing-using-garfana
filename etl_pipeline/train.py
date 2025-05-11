
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import EyeDataset
from src.model import build_model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow

# Setup MLflow
mlflow.set_experiment("eye-disease-resnet50-custom")

# Dataset and DataLoader
dataset = EyeDataset("data/processed")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = build_model(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
with mlflow.start_run():
    mlflow.log_param("model", "resnet50_custom")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)

    # Save model for evaluation
    torch.save(model.state_dict(), "resnet50_custom_model.pth")
    mlflow.pytorch.log_model(model, "resnet50_custom_model")
