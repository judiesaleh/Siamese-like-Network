import random
from typing import Tuple, Any
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import Tensor



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PairedDataset(Dataset):
    def __init__(self, dataset: CIFAR10, transform: Any = None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.class_indices = {}
        for i in range(len(dataset)):
            _, label = dataset[i] # _ is used in python as a throwaway variable (ignore image, since we dont need it)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)

    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        image1, label1 = self.dataset[index]
        same_class = random.choice([True, False])
        if same_class:
            index2 = random.choice(self.class_indices[label1])
            pair_label = 1
        else:
            other_classes = list(self.class_indices.keys())
            other_classes.remove(label1)
            random_class = random.choice(other_classes)
            index2 = random.choice(self.class_indices[random_class])
            pair_label = 0

        image2, _ = self.dataset[index2]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, pair_label
