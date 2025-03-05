import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.paired_dataset import PairedDataset
from models.siamese_network import SiameseNetwork
from src.training.model_wrapper import ModelWrapper
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



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

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616] )
])



def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, start_epoch, loss




def collate_fn(batch):
    img1_list = [item[0] for item in batch]
    img2_list = [item[1] for item in batch]
    label_list = [item[2] for item in batch]
    batch_img1 = torch.stack(img1_list)  # Shape: (batch_size, C, H, W)
    batch_img2 = torch.stack(img2_list)  # Shape: (batch_size, C, H, W)
    batch_labels = torch.tensor(label_list)  # Shape: (batch_size,)
    return (batch_img1, batch_img2), batch_labels

# check of what devic is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create paired datasets for training and testing
paired_train_dataset = PairedDataset(train_dataset)
paired_test_dataset = PairedDataset(test_dataset)

# DataLoaders for training and testing
train_loader = DataLoader(paired_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(paired_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# Initialize the Siamese network model
model = SiameseNetwork(freeze_pretrained=True).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


wrapper = ModelWrapper(model, optimizer, loss, device=device)


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


callbacks = [save_checkpoint_callback]

# Train the model
num_epochs = 5

"""
checkpoint_path = 'checkpoint_epoch_5.pth'  # Path to saved checkpoint
model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
wrapper = ModelWrapper(model, optimizer, loss, device=device)

in fit you need to adjust the parameter epochs to num_epochs - start_epoch
"""
metrics = wrapper.fit(train_loader, epochs=num_epochs, data_eval=test_loader, evaluate=True, verbose=True, callbacks=callbacks)

# after training run tensorboard
print("Training Metrics:", metrics["training"])
print("Test Metrics:", metrics["test"])


def plot_metrics(metrics: dict, save_path: str = None) -> None:
    """Plot training and test loss/accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['training']['loss'], label='Training Loss', marker='o')
    if metrics['test']['loss']:
        plt.plot(metrics['test']['loss'], label='Test Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['training']['acc'], label='Training Accuracy', marker='o')
    if metrics['test']['acc']:
        plt.plot(metrics['test']['acc'], label='Test Accuracy', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# Call the plotting function after training
plot_metrics(metrics, save_path='training_metrics.png')






