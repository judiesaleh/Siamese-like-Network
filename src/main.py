import torch
import matplotlib.pyplot as plt
import os
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.paired_dataset import PairedDataset
from models.siamese_network import SiameseNetwork
from training.train import train_model, LossMonitor, AccuracyMonitor, WeightTracker
from training.validate import validate_model
from training.train import WeightTracker





# datasets from torchvision has prebuilt access to popular datasets
# Can be used for downloading, preprocessing, transformation or standardization

# Transforms provide preprocessing functions to modify the dataset
# for machine learning tasks

# Normalization pixel values scaling to a standard range of (like [-1, 1] or [1, 0])
# Match pretrained models like ResNet


# Tensors are Pytorch's primary data structure
# Enables efficient computations on GPUs
# Integrate seamlessly with Pytorch's neural networks layers and optimizers

# Define transforms for normalization and resizing


def main() -> None:
    # check of what devic is being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create paired datasets for training and testing
    paired_train_dataset = PairedDataset(train_dataset)
    paired_test_dataset = PairedDataset(test_dataset)

    # DataLoaders for training and testing
    train_loader = DataLoader(paired_train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(paired_test_dataset, batch_size=32, shuffle=False)

    # Initialize the Siamese network model
    model = SiameseNetwork(freeze_pretrained=True).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_monitor = LossMonitor()
    acc_monitor = AccuracyMonitor()
    weight_tracker = WeightTracker(layer_name="head.0")

    def save_checkpoint_callback(model, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


    callbacks = [save_checkpoint_callback, weight_tracker]

    # Train the model
    num_epochs = 5
    train_model(model, train_loader, criterion, optimizer, num_epochs, device, callbacks=callbacks, loss_monitor=loss_monitor,
        acc_monitor=acc_monitor, resume_from_checkpoint=None)

    # Validate/Test the model
    val_loss, val_acc = validate_model(model, test_loader, criterion, device)
    print("Accuracies per epoch:", acc_monitor.epoch_accuracies)
    print("Losses per epoch:", loss_monitor.epoch_losses)
    print("Validation Loss:", val_loss)
    print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_monitor.epoch_losses) + 1), loss_monitor.epoch_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_monitor.epoch_accuracies) + 1), loss_monitor.epoch_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
