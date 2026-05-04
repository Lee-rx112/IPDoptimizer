# Interacting Particle Dynamics (IPD) for Neural Network Optimization

This repository contains the official implementation of the **Interacting Particle Dynamics (IPD)** optimization framework. The framework introduces parameter-interaction fields, specifically Diffusion-like and Coulomb-like variants, to optimize the traversal of non-convex loss landscapes during deep neural network training.

## Project Structure

The codebase is organized into the following functional modules:

*   **`main_new.py`**: The primary entry point for executing CIFAR-100 image classification tasks. It handles model initialization, training loops, and performance logging for both baselines and IPD variants.
*   **`IPD_opt.py`**: Implementation of the IPD optimizer framework. It contains the logic for **IPD-Diff** and **IPD-Coul** bias operators integrated with SGD and AdamW base optimizers.
*   **`map_contribution.py`**: Defines neighborhood mapping functions and structural contribution mechanisms used to calculate the interaction fields.
*   **`config.py`**: Centralized configuration management for all hyperparameters, including interaction strengths and architecture-specific settings.
*   **`util.py`**: Utility functions for dataset handling (CIFAR-100), training scheduling, and general logging procedures.

## Environment Requirements

The experiments were conducted using high-performance hardware (NVIDIA RTX 5090 Blackwell architecture). To ensure faithful reproduction, we recommend the following software stack:

*   **Python**: 3.10+
*   **PyTorch**: 2.5+ (Optimized for CUDA 13.0+)
*   **CUDA**: 13.0 or higher
*   **Hardware**: Recommended 24GB+ VRAM (e.g., RTX 5090) for optimal throughput using FP8 quantization or large batch sizes.

## Usage

### Training with Base Optimizers
To train a baseline model (ResNet-50 or DenseNet-121) using standard SGD or AdamW:
```bash
python main_new.py --model resnet50 --optimizer sgd --lr 0.1
python main_new.py --model densenet121 --optimizer adamw --lr 0.001
python main_new.py --model resnet50 --optimizer ipd_diff --strength 0.001 --ratio 0.4
python main_new.py --model densenet121 --optimizer ipd_coul --strength 0.005 --ratio 0.15

