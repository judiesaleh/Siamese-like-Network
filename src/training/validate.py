import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import device


def validate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: device) -> (float, float):
    model.eval()  # Switch to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item() * img1.size(0)

            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc
