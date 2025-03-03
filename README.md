# Siamese Network for Image Similarity Detection  
**Academic Project Submission**  
*Advanced Python Programming - Ruhr-Universität Bochum*  

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-%23EE4C2C.svg)](https://pytorch.org/)
[![Academic Use](https://img.shields.io/badge/License-Academic%20Use-blue.svg)](LICENSE)

<img src="docs/Siamese.jpg" width="600" alt="Siamese Network Architecture">

A PyTorch implementation of a Siamese Network for image pair classification using transfer learning with ResNet18 backbone. Developed for Advanced Python Programming at the Ruhr-Universität Bochum.

## Features

- **Pair Generation**: Balanced sampling of same/different class pairs from CIFAR-10
- **Transfer Learning**: Pretrained ResNet18 feature extraction with optional freezing
- **Training Pipeline**: Integrated with checkpointing and metrics tracking
- **Academic Compliance**: Proper CIFAR-10 dataset citation and usage terms

## Installation

```bash
git clone https://github.com/yourusername/siamese-transfer-learning.git
cd siamese-transfer-learning
pip install -r requirements.txt