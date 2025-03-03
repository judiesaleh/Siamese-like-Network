import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import device, Tensor
from typing import Any, Callable, List

# For tracking loss for each epoch/batches
class LossMonitor:
    def __init__(self) -> None:
        self.losses = []

    def update(self, loss: float) -> None:
        self.losses.append(loss)

    def get_average(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

# For tracking accuracy for each epoch/batches
class AccuracyMonitor:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, prediction: Tensor, labels: Tensor) -> None:
        # (prediction == labels) produces a boolean True (1) or False (0)
        # That gets converted into a float e.g. 1.0
        # sum() adds up all 1.0s to count the correct predictions in the batch
        # .item() extracts the numerical value from the tensor
        self.correct += (prediction == labels).float().sum().item()
        self.total += labels.size(0)

    def get_accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

class WeightTracker:
    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        self.weight_means = []
        self.weight_stds = []

    def __call__(self, model: nn.Module, epoch: int, loss: float) -> None:
        for name, param in model.named_parameters():
            self.weight_means.append(param.mean().item())
            self.weight_stds.append(param.std().item())
        print(f"Epoch {epoch+1}: {self.layer_name} - Mean: {self.weight_means[-1]:.4f}, Std: {self.weight_stds[-1]:.4f}")

def train_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
        # criterion refers for the loss function (
        # measures the error between your model's predictions and the
        # actual labels, guiding how the model should update
        # its weights during training.)
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: device,
    callbacks: List[Callable] = None
) -> None:

    model.to(device)
    loss_monitor = LossMonitor()
    acc_monitor = AccuracyMonitor()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch in loader:
            # Unpack the batch data: two images and their binary label
            img1, img2, labels = batch

            # Move data to the device (CPU/GPU)
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute model outputs and loss
            outputs: Tensor = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy metrics
            running_loss += loss.item() * img1.size(0)
            # Convert model outputs to predictions using sigmoid thresholding
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            loss_monitor.update(loss.item())
            acc_monitor.update(predictions, labels)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = acc_monitor.get_accuracy()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if callbacks:
            for callback in callbacks:
                callback(model, epoch, epoch_loss)

