import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.siamese_network import SiameseNetwork
from src.data.paired_dataset import PairedDataset
from training.validate import validate_model

def test_model(checkpoint_path: str) -> None:
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Create the paired dataset for testing
    paired_test_dataset = PairedDataset(test_dataset)
    test_loader = DataLoader(paired_test_dataset, batch_size=32, shuffle=False)

    # Initialize the model and load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(freeze_pretrained=True).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define the loss function (same as during training)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Evaluate the model on the test set
    test_loss, test_acc = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Replace with the version (checkpoint) of the model you want to test
    test_model('checkpoint_epoch_5.pth')

