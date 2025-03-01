import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor


class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the Siamese network using a pretrained ResNet18 model.
        The final fully connected layer of ResNet18 is replaced with an identity layer
        to extract features. A head network is defined to combine the embeddings from
        two images and output a binary prediction.
        """
        super(SiameseNetwork, self).__init__()
        # Load the pretrained ResNet18 model
        self.feature_extractor = models.resnet18(pretrained=True)
        # Remove the final classification layer so that the model outputs embeddings
        # ResNet18, with its configuration, ends with a fully
        # connected layer
        """
        nn.Identity() is a PyTorch layer that simply returns
        its input without any change
        By assigning it to self.feature_extractor.fc,
        you effectively remove the classification layer.
        """

        self.feature_extractor.fc = nn.Identity()

        """
        The model now outputs the 512-dimensional features directly.
        These features can then be used as input to another network 
        (the head in the Siamese architecture) for tasks like 
        comparing image pairs.
        """

        # Define a small neural network head to compare embeddings from two images.
        # Here we concatenate the two 512-dim embeddings from ResNet18.
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # Single output for binary classification (same/different)
        )

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Forward pass for the Siamese network.

        Args:
            img1 (Tensor): The first batch of images.
            img2 (Tensor): The second batch of images.

        Returns:
            Tensor: The raw output logits for each image pair.
        """
        # Get feature embeddings for both images using the pretrained ResNet18
        feat1: Tensor = self.feature_extractor(img1)
        feat2: Tensor = self.feature_extractor(img2)
        # Combine the two embeddings by concatenating them along the feature dimension
        combined: Tensor = torch.cat((feat1, feat2), dim=1)
        # Compute the output using the head network
        output: Tensor = self.head(combined)
        return output
