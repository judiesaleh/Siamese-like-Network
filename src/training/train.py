import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import device, Tensor
from typing import Any

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
    device: device
) -> None:

    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

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
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Save a checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
