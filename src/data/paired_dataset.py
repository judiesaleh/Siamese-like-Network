import random
from typing import Tuple, Any
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

"""
"Datasets" from torchvision has prebuilt access to popular datasets
-> Can be used for downloading, preprocessing, transformation or standardization

"Transforms" provide preprocessing functions to modify the dataset
for machine learning tasks

"Normalizing" pixel values is scaling to a standard range of 
(like [-1, 1] or [1, 0]) to match pretrained models like ResNet

"Tensors" are Pytorch's primary data structure
-> enables efficient computations on GPUs
-> Integrate seamlessly with Pytorch's neural networks layers and optimizers

we define transforms for normalization and resizing
"""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PairedDataset(Dataset):
    def __init__(self, dataset: CIFAR10, transform: Any = None) -> None:
        self.dataset = dataset
        self.transform = transform
        """
        Empty dictionary. key: class label, value: list of indices
        efficient: speeds up pair generation by avoiding repeated searches
        for images of specific classes
        Dictionary Look ups in O(1)
        """
        self.class_indices = {}
        for i in range(len(dataset)):
            _, label = dataset[i] # _ is used in python as a throwaway variable (ignore image, since we dont need it)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)

        """
        Imagine dataset is CIFAR10 with labels 0 to 9. 
        After running this:
        self.class_indices[0] might be [0, 5, 12, ...] 
        (indices of all "airplane" images).
        self.class_indices[1] might be [1, 3, 9, ...] 
        (indices of all "automobile" images). And so on for all 10 classes.
        """

    # This len function determines how many pairs are
    # generated per epoch
    def __len__(self) -> int:
        return len(self.dataset)

    """
    defining __getitem__ enables indexing, with random
    selection of the second image plus choosing the right label accordingly
    (Check if both images belong to the same class or not)
    """
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img1, label1 = self.dataset[index]
        """
        random choice for balanced sampling, which ensures te model sees
        both cases, preventing bias
        """
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
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, pair_label
