# Project 6: Siamese Transfer Learning for Image Similarity  
**Advanced Python Programming Course**  

---

## Objective  
Build a **Siamese-like neural network** to determine if two images belong to the same class. The model uses a pretrained network (e.g., ResNet) for feature extraction and a custom head to compare image pairs.  

---

## Key Features  
- **Paired Dataset**: Generates image pairs (same/different classes) from CIFAR-10/FashionMNIST.  
- **Transfer Learning**: Uses pretrained models (ResNet/VGG) to extract features.  
- **Similarity Prediction**: Compares embeddings with a neural network head.  
- **Checkpointing**: Saves progress to resume training after interruptions.  
- **Visualization**: Plots training metrics and memory usage.  

---

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/siamese-transfer-learning.git  