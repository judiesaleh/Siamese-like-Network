# Siamese Network for Image Similarity Detection  
**Academic Project Submission**  
*Advanced Python Programming - Ruhr-Universität Bochum*  

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-%23EE4C2C.svg)](https://pytorch.org/)
[![Academic Use](https://img.shields.io/badge/License-Academic%20Use-blue.svg)](LICENSE)

<img src="docs/Siamese.jpg" width="600" alt="Siamese Network Architecture">

A PyTorch implementation of a Siamese Network for image pair classification using transfer learning with ResNet18 backbone. Developed for Advanced Python Programming at the Ruhr-Universität Bochum.

## Key Features

- **Balanced Pair Generation**: Dynamic pair sampling from CIFAR-10 with 50% same-class and 50% different-class pairs
- **Modified ResNet18 Backbone**: 
  - Pretrained feature extraction with optional freezing
  - Final FC layer replaced with identity mapping
  - Custom head network with dimensionality reduction
- **Training Infrastructure**:
  - Automatic checkpointing after each epoch
  - GPU acceleration support
  - Integrated metrics tracking (loss & accuracy)
  - Visualization of training curves
- **Modular Architecture**:
  - Separate components for data processing, model definition, and training logic
  - Custom collate function for pair handling


## Installation
```bash
git clone https://github.com/judiesaleh/siamese-transfer-learning.git
pip install -r requirements.txt
```

## Training
```
python main.py
```