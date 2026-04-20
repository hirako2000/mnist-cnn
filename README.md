[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white")](https://pytorch.org/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005A9C?style=for-the-badge)](https://onnxruntime.ai/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-FF6B6B?style=for-the-badge)](https://www.w3.org/TR/webgpu/)
[![Hugging Face](https://img.shields.io/badge/🤗-Dataset-FFD21E?style=for-the-badge)](https://huggingface.co/ylecun/mnist)
[![License](https://img.shields.io/badge/License-CC--BY--NC-blue?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

[Live demo](https://mnist.mtassoumt.uk)


# MNIST CNN

<p align="center">
  <img src="./img/mnist.avif" alt="MNIST CNN logo">
</p>

---

A Convolutional Neural Network for handwritten digit recognition, built with PyTorch. This project is designed for understanding.

This architecture closely follows [LeNet-5](https://en.wikipedia.org/wiki/LeNet). The network introduced by Yann LeCun in 1998 for vision recognition. The design, convolutional layers followed by fully connected layers remains unchanged, demonstrating how this structure learns to read digits, already over twenty-five years ago.

## The what

Takes a 28x28 grayscale image of a handwritten digit (0-9) and predicts which digit it is. The model achieves ~99% accuracy on the MNIST test set after 5-10 epochs.

## Sample images from the dataset

![Dataset samples](./visualisations/dataset_samples.avif)

## The structure

The code is organized into modules in the [src](./src/) folder:

- `data` – Downloads MNIST from Hugging Face and saves as Parquet files using Polars
- `viz` – Generates visualizations (sample grids, class distributions, confusion matrices) saved as AVIF images
- `train` – Trains the CNN model with configurable epochs, batch size, and device
- `infer` – Runs predictions on single images, directories, or random test samples

Training and inference supports MPS (Apple Silicon), CUDA (Nvidia GPU) or CPU.

The web folder contains a simple web application to run inference via WebGPU.
## Setup

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/), a fast Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and set up the project:

```bash
git clone https://github.com/hirako2000/mnist-cnn
cd mnist-cnn
uv sync
```

`uv sync` creates a virtual environment and installs all dependencies. You can either:

- Use `uv run` for individual commands (activates environment automatically)
- Activate the environment manually with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)

Then fetch the dataset and run a quick test:

```bash
just data fetch  # Download and prepare the dataset
just viz all     # Generate all visualizations
```

```bash
just train      # Default training (5 epochs, CPU)
just train mps  # 5 epochs on Apple GPU
```

```bash
just infer image path/to/digit.png  # Predict a single image
just infer random 5                 # Predict random samples from the test set
```

## Commands

Commands are organized by module, just enter the `just` command to see all modules:
![just](./img/just.avif)

And just <module> to see a particular module's commands:
![just-data](./img/just-data.avif)


## Dataset exploration

The dataset is balanced across all digits, with 6,000 training samples and 1,000 test samples per class.

![Class distribution](./visualisations/class_distribution.avif)

The heatmap below shows the average pixel intensity for each digit, revealing the "typical" shape the model learns to recognize.

![Pixel heatmap](./visualisations/pixel_heatmap.avif)

## Model performance

After training, the confusion matrix shows where the model occasionally makes mistakes — most commonly confusing 4 and 9, or 3 and 8.

![Confusion matrix](./visualisations/confusion_matrix.avif)

Here are example predictions from a test set with particularly poorly written digit. Green titles indicate correct predictions, red titles show where the model was wrong.

![Prediction examples](./visualisations/prediction_examples.avif)

## 🧪 ReLU Activation

> **This section references the [`improvement/relu`](https://github.com/hirako2000/mnist-cnn/tree/improvement/relu) branch**

The original LeNet-5 uses `tanh` activations (state-of-the-art in 1998). Modern CNNs replaced this with [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectified_linear_unit) (ReLU) in 2010, enabling much deeper networks.

Even on shallow LeNet-5, ReLU provides measurable benefits:

- **+0.12% peak accuracy** (98.99% → 99.11%)
- **23% lower test loss** (0.0448 → 0.0345) - more confident predictions
- **Faster convergence** to 99% accuracy
- **Better generalization** (reduced overfitting)

<div align="center">
  <img src="https://raw.githubusercontent.com/hirako2000/mnist-cnn/improvement/relu/visualisations/training_comparison_relu_vs_tanh.avif" 
       alt="ReLU vs Tanh Comparison"
       width="90%">
  <br>
  <em>Training dynamics: ReLU (green) vs Tanh (red) over 20 epochs</em>
</div>

### Observations from the comparison:

| Metric | Tanh (Original) | ReLU (Improved) | Improvement |
|--------|----------------|----------------|-------------|
| Best Test Accuracy | 98.99% | 99.11% | **+0.12%** |
| Final Test Accuracy | 98.86% | 99.03% | **+0.17%** |
| Minimum Test Loss | 0.0327 | 0.0286 | **-12.5%** |
| Epochs to 99% | 16 | 10 | **6 epochs faster** |
| Overfitting Gap | 0.94% | 0.73% | **-22%** |

### Why ReLU matters

This improvement bridges the 12-year gap between LeNet-5 (1998) and the deep learning revolution (2010-2012):
- **1998**: LeNet-5 with tanh - works for shallow networks
- **2010**: ReLU proposed - enables training of deep networks
- **2012**: AlexNet uses ReLU, wins ImageNet with 8 layers

## How it works

The model is a simple CNN with two convolutional layers, two max-pooling layers, and two fully connected layers. The [train.py](./src/train/train.py) code includes detailed comments explaining:

- Why convolutions slide across images (position invariance)
- What pooling does (reduces size, adds robustness)
- Where weights come from (random initialization, then adjusted via backpropagation)
- Where labels are used (only during training, to calculate loss)

The code uses Polars for efficient Parquet data loading and supports mixed precision (FP16) for faster training on Apple MPS or CUDA devices.

The [model.py](./src/train/model.py) code is also thorougly commented. Along with train.py, they contain all the involved training logic.

For inference, see [predict.py](./src/infer/predict.py).

## Requirements

- Python 3.14 or later
- uv package manager

All dependencies are managed through `uv sync` – no manual installation required.