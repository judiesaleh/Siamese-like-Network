import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor



class SiameseNetwork(nn.Module):
    def __init__(self, freeze_pretrained: bool = False) -> None:
        """
        Initializes the Siamese network using a pretrained ResNet18 model.
        The final fully connected layer of ResNet18 is replaced with an identity layer
        to extract features. A head network is defined to combine the embeddings from
        two images and output a binary prediction.
        """
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)

        if freeze_pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False


        # fc is the final (linear) layer of the nn
        self.feature_extractor.fc = nn.Identity()
        self.head = nn.Sequential(
            # Dimensionality reduction for efficiency
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),

            # inplace=True: Saves memory by modifying the input tensor directly (no copy).
            nn.ReLU(inplace=True),

            nn.Linear(512, 2) # produces logits for score_diff and score_same
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
        combined = torch.abs(feat1 - feat2)
        output: Tensor = self.head(combined)
        return output
