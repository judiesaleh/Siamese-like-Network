import random
from typing import Tuple, Any
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL import Image

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

class PairedDataset(Dataset):
    def __init__(self, dataset: CIFAR10) -> None:
        self.dataset = dataset
        self.class_indices = {}  # Empty dictionary. key: class label, value: list of indices
        for i in range(len(dataset)):
            _, label = dataset[i] # _ is used in python as a throwaway variable
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img1, label1 = self.dataset[index]
        same_class = random.choice([True, False])
        if same_class:
            idx2 = random.choice(self.class_indices[label1])
            pair_label = 1
        else:
            other_classes = list(self.class_indices.keys())
            other_classes.remove(label1)
            random_class = random.choice(other_classes)
            idx2 = random.choice(self.class_indices[random_class])
            pair_label = 0

        img2, _ = self.dataset[idx2]
        return img1, img2, pair_label
