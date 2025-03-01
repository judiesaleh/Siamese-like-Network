import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.paired_dataset import PairedDataset
from models.siamese_network import SiameseNetwork
from training.train import train_model
from training.validate import validate_model



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
    model = SiameseNetwork()

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    num_epochs = 5  # Adjust as needed
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Validate/Test the model
    val_loss, val_acc = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
