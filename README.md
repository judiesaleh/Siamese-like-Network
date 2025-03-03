# Siamese Network for Image Similarity using Transfer Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-%23EE4C2C.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.21.0-%23EE4C2C.svg)](https://pytorch.org/vision/stable/index.html)

## Academic Use
This codebase was developed for educational purposes as part of Advanced Python Programming Lab at the Ruhr-Universität Bochum.  
It is provided without warranty and intended solely for academic evaluation.
A PyTorch implementation of a Siamese Network for image similarity detection using pretrained ResNet18 features. Designed for pair-wise classification tasks ("same class" vs "different class") on CIFAR-10 dataset.

This project uses the CIFAR-10 dataset under its original terms of use.  
Required citations can be found in:
- [CITATION.md](CITATION.md) - BibTeX format
- LICENSE file - Human-readable reference

## Features

- **PyTorch 2.6.0** with CUDA 11.8 support
- **TorchVision 0.21.0** for dataset transformations
- **NumPy 2.2.3** for numerical operations
- **Matplotlib 3.10.1** for visualization support
- **tqdm 4.67.1** for progress tracking

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/siamese-transfer-learning.git
cd siamese-transfer-learning
```

## Install Depandencies
```bash
pip install -r requirements.txt
```
