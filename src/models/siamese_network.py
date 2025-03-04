import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor

"""
Siamese Networks:
Designed to compare two inputs (e.g., images) using twin subnetworks
with shared weights. 
-> Used for verification tasks (e.g., "Are these two images from 
    the same class?").
-> Learns a similarity metric between inputs through contrastive 
    or binary cross-entropy loss.

Transfer Learning:
Reusing pretrained models (like ResNet18) as feature extractors.
Freezing weights prevents updates during training, 
preserving learned features.

Activation functions:
An activation function in a neural network introduces non-linearity 
to the model, enabling it to learn complex patterns and decision 
boundaries. Without activation functions, a neural network would 
behave like a simple linear transformation, no matter how many layers 
it has.
"""

class SiameseNetwork(nn.Module):
    def __init__(self, freeze_pretrained: bool = False) -> None:
        """
        Initializes the Siamese network using a pretrained ResNet18 model.
        The final fully connected layer of ResNet18 is replaced with an identity layer
        to extract features. A head network is defined to combine the embeddings from
        two images and output a binary prediction.
        """
        super(SiameseNetwork, self).__init__()
        # Load the pretrained(pretrained=True) ResNet18 model as feature extractor
        self.feature_extractor = models.resnet18(pretrained=True)
        # Remove the final classification layer so that the model outputs embeddings
        # ResNet18, with its configuration, ends with a fully connected layer

        # Optional freezing of the pretrained model
        if freeze_pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        """
        nn.Identity() is a PyTorch layer that simply returns
        its input without any change
        By assigning it to self.feature_extractor.fc,
        the classification layer is effectively removed.
        """

        # fc is the final (linear) layer of the nn
        self.feature_extractor.fc = nn.Identity()

        """
        The model now outputs the 512-dimensional features directly.
        These features can then be used as input to another network 
        (the head in the Siamese architecture) for tasks like 
        comparing image pairs.
        """

        # Define a neural network head to compare embeddings from two images.
        # Here we concatenate the two 512-dim embeddings from ResNet18.
        self.head = nn.Sequential(
            # Dimensionality reduction for efficiency
            nn.Linear(512 * 2, 256),

            # Rectified Linear Unit activation function
            # inplace=True: Saves memory by modifying the input tensor directly (no copy).
            nn.ReLU(inplace=True),

            # Single output for binary classification (same/different or 1/0)
            # torch.sigmoid(output) to convert to a probability (0-1)
            nn.Linear(256, 1)
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
        feat1: Tensor = self.feature_extractor(img1) # Shape:(batch_size, 512)
        feat2: Tensor = self.feature_extractor(img2) # Shape:(batch_size, 512)
        # Combine the two embeddings by concatenating them along the feature dimension
        combined: Tensor = torch.cat((feat1, feat2), dim=1)
        # Compute the output using the head network (Now combined will go through
        # all the layers inside nn.Sequential, because head is a sequential module)
        output: Tensor = self.head(combined)
        return output
