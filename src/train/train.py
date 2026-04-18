#!/usr/bin/env python3
"""
Train a Convolutional Neural Network on the MNIST handwritten digit dataset.

This code demonstrates the complete ML pipeline:
1. Load data from Parquet files (using Polars for efficiency)
2. Create a CNN model
3. Train the model using backpropagation
4. Evaluate on test data
5. Save checkpoints and the best model

Every concept is explained in the comments below.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim

# Import the model architecture defined in model.py
from model import LeNet5
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ParquetMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for MNIST data stored in Parquet format.

    Parquet is a columnar storage format that is efficient for reading large datasets.
    Polars is used to read the Parquet file because it's faster than Pandas and
    has zero-copy integration with NumPy.

    The Parquet file contains:
    - One row per image
    - First column: label (integer 0-9)
    - Next 784 columns: pixel_0 through pixel_783 (values 0-255)
    """

    def __init__(self, parquet_path: str):
        """
        Args:
            parquet_path: Path to the Parquet file (e.g., "data/mnist_train.parquet")
        """
        # Read the entire Parquet file into a Polars DataFrame
        # Polars is lazy by default but read_parquet executes immediately
        self.df = pl.read_parquet(parquet_path)

        # Precompute pixel column names for faster access
        # 28x28 = 784 pixels, named pixel_0 to pixel_783
        self.pixel_cols = [f"pixel_{i}" for i in range(784)]

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return a single sample (image, label) as PyTorch tensors.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            image: torch.Tensor of shape (1, 28, 28) normalized to approximately [-1, 1]
            label: torch.Tensor containing a single integer (0-9)
        """
        # Get the row as a tuple of values
        # Row format: (label, pixel_0, pixel_1, ..., pixel_783)
        row = self.df.row(idx)

        # First value is the label (which digit this image represents)
        label = row[0]

        # Remaining 784 values are the pixel intensities (0 = black, 255 = white)
        pixels = np.array(row[1:], dtype=np.float32)

        # Reshape from flat array (784,) to 2D image (1, 28, 28)
        # The 1 is the channel dimension (grayscale = 1 channel)
        image = pixels.reshape(1, 28, 28)

        # ----- NORMALIZATION -----
        # Step 1: Scale pixel values from [0, 255] to [0, 1]
        image = image / 255.0

        # Step 2: Standardize to approximately [-1, 1] using dataset statistics
        # These values (0.1307, 0.3081) are the mean and standard deviation of MNIST
        # Standardization formula: (x - mean) / std
        # This helps the model train faster and more stably
        image = (image - 0.1307) / 0.3081

        # Convert NumPy array to PyTorch tensor
        # Labels are converted to long integers (required for CrossEntropyLoss)
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def find_project_root(marker="pyproject.toml"):
    """
    Find the project root directory by walking up until a marker file is found.

    This avoids hardcoding paths or using OS-specific environment variables.
    It works regardless of where the script is called from.

    Args:
        marker: Filename that exists only in the project root (e.g., pyproject.toml)

    Returns:
        Path object pointing to the project root directory

    Raises:
        RuntimeError: If the marker file is not found in any parent directory
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find {marker}")


def confusion_matrix(labels, predictions, num_classes=10):
    """
    Compute a confusion matrix manually (no scikit-learn dependency).

    A confusion matrix shows how many times each true class was predicted as each possible class.
    Rows = true labels, Columns = predicted labels.

    Args:
        labels: List of true labels (ground truth)
        predictions: List of predicted labels from the model
        num_classes: Number of classes (10 for MNIST digits 0-9)

    Returns:
        A 2D numpy array where cm[i][j] = count of class i predicted as class j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, predictions, strict=False):
        cm[true, pred] += 1
    return cm


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one complete pass through the training dataset.

    1. Forward pass: Images pass through the network to produce predictions
    2. Loss calculation: Compare predictions to true labels using cross-entropy
    3. Backward pass (backpropagation): Compute gradients of loss with respect to each weight
    4. Weight update: Optimizer adjusts weights slightly to reduce future loss

    Args:
        model: The neural network
        loader: DataLoader that yields batches of (images, labels)
        optimizer: Algorithm that updates weights (Adam in this case)
        criterion: Loss function (CrossEntropyLoss)
        device: 'cpu', 'mps', or 'cuda'

    Returns:
        avg_loss: Average loss over all batches in this epoch
        accuracy: Percentage of correct predictions
    """
    model.train()  # Set to training mode (enables dropout, batch norm, etc.)

    running_loss = 0.0  # Accumulated loss across all batches
    correct = 0  # Count of correct predictions
    total = 0  # Count of total predictions

    # tqdm shows a progress bar with dynamic updates
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        # Move data to the compute device (GPU or CPU)
        # This is necessary because tensors start on CPU
        images = images.to(device)
        labels = labels.to(device)

        # ----- STEP 1: Zero the gradients -----
        # Gradients accumulate by default in PyTorch.
        # We must zero them before each backward pass.
        optimizer.zero_grad()

        # ----- STEP 2: Forward pass -----
        # Images flow through the network, producing raw logits (not probabilities)
        outputs = model(images)

        # ----- STEP 3: Calculate loss -----
        # CrossEntropyLoss combines LogSoftmax + Negative Log Likelihood
        # It measures "how surprised" the model is by the correct answer
        # Lower loss = better predictions
        loss = criterion(outputs, labels)

        # ----- STEP 4: Backward pass (compute gradients) -----
        # This computes dLoss/dWeight for every weight in the network
        # Using the chain rule of calculus, it propagates error backward
        loss.backward()

        # ----- STEP 5: Update weights -----
        # The optimizer takes a step in the direction that reduces loss
        # For Adam: combines momentum and adaptive learning rates
        optimizer.step()

        # ----- TRACKING: Update statistics -----
        running_loss += loss.item() * images.size(0)

        # Get the predicted class (highest logit value)
        # torch.max returns (values, indices). We only need the indices
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate averages for this epoch
    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.

    Similar to train_epoch but WITHOUT:
    - Gradient computation (no backward pass)
    - Weight updates
    - Dropout (automatically disabled in eval mode)

    This is faster and uses less memory because we don't store gradients.

    Args:
        model: The neural network
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: 'cpu', 'mps', or 'cuda'

    Returns:
        avg_loss: Average loss over the evaluation set
        accuracy: Percentage of correct predictions
        all_labels: List of all true labels (for confusion matrix)
        all_preds: List of all predicted labels (for confusion matrix)
    """
    model.eval()  # Sets to evaluation mode (disables dropout, uses running stats for batch norm)

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # torch.no_grad() disables gradient computation
    # This saves memory and speeds up computation
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only (no backward pass)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predictions for later analysis (confusion matrix)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_preds


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save a training checkpoint.

    Checkpoints allow resuming training later and keep the best model.

    Args:
        model: The neural network
        optimizer: The optimizer (saving its state allows resuming exactly)
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'best_acc': 98.5})
        path: Where to save the checkpoint file
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),  # All the learned weights
            "optimizer_state_dict": optimizer.state_dict(),  # Optimizer momentum
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file
        model: The neural network (will load weights into it)
        optimizer: Optional optimizer (will load its state)

    Returns:
        epoch: The epoch number from the checkpoint
        metrics: The metrics dictionary from the checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["metrics"]


def main():
    """
    Main training script.

    Command-line arguments control:
    - Number of epochs
    - Batch size
    - Learning rate
    - Device (CPU, MPS, CUDA)
    - Whether to limit samples (for quick testing)
    - Whether to resume from a checkpoint
    """
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (passes through the entire dataset)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of images processed before updating weights",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate: how much to adjust weights each step",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to train on: cpu, mps (Apple Silicon), or cuda (Nvidia)",
    )

    # to set dataset size
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training samples (useful for quick tests)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Limit test samples (useful for quick tests)",
    )

    # training from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )

    args = parser.parse_args()

    # ----- SETUP: Find project directories -----
    # This works from anywhere as we search for pyproject.toml
    project_root = find_project_root()
    data_dir = (
        project_root / "data"
    )  # Contains mnist_train.parquet and mnist_test.parquet
    model_dir = project_root / "models"  # Where to save checkpoints

    # ----- SETUP: Choose compute device -----
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS (Metal Performance Shaders) - Apple Silicon GPU")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA - Nvidia GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✅ Using CPU (no GPU acceleration)")

    # Create the models directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)

    # ----- DATA LOADING -----
    print("\n📥 Loading datasets...")

    # Create dataset objects (they read Parquet files)
    train_dataset = ParquetMNISTDataset(str(data_dir / "mnist_train.parquet"))
    test_dataset = ParquetMNISTDataset(str(data_dir / "mnist_test.parquet"))

    # Limit dataset size if requested (for quick testing)
    if args.max_train_samples:
        train_dataset.df = train_dataset.df.head(args.max_train_samples)
    if args.max_test_samples:
        test_dataset.df = test_dataset.df.head(args.max_test_samples)

    # Create DataLoaders which handle batching and shuffling
    # shuffle=True for training (randomizes order each epoch)
    # shuffle=False for testing (order doesn't matter)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"✅ Train: {len(train_dataset):,} samples")
    print(f"✅ Test: {len(test_dataset):,} samples")

    # ----- MODEL CREATION -----
    model = LeNet5().to(device)  # Move model to the compute device
    print(f"✅ Model: {model.get_num_params():,} trainable parameters")

    # ----- LOSS FUNCTION AND OPTIMIZER -----
    # CrossEntropyLoss = LogSoftmax + Negative Log Likelihood
    # Standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam (Adaptive Moment Estimation) optimizer
    # Combines benefits of AdaGrad (sparse gradients) and RMSProp (non-stationary)
    # Usually works well without extensive tuning
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----- RESUME FROM CHECKPOINT (if requested) -----
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"🔄 Resuming from {resume_path}")
            start_epoch, metrics = load_checkpoint(str(resume_path), model, optimizer)
            best_acc = metrics.get("best_acc", 0.0)
            print(
                f"   Resumed from epoch {start_epoch}, best acc so far: {best_acc:.2f}%"
            )
        else:
            print(f"❌ Checkpoint not found: {resume_path}")

    # ----- TRAINING LOOP -----
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print("=" * 50)

    # History dictionary to track metrics over time (for later plotting)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        print(f"\n📊 Epoch {epoch}/{start_epoch + args.epochs}")
        print("-" * 30)

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate on test set
        test_loss, test_acc, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device
        )

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Print results for this epoch
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"🎯 Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        # ----- SAVE BEST MODEL -----
        # Keep the model with the highest test accuracy
        if test_acc > best_acc:
            best_acc = test_acc

            # Save checkpoint with all training state
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"best_acc": best_acc},
                model_dir / "best_model.pt",
            )
            print(f"💾 New best model! (accuracy: {best_acc:.2f}%)")

            # Also save confusion matrix for this best model
            cm = confusion_matrix(test_labels, test_preds)
            np.save(model_dir / "confusion_matrix.npy", cm)

        # Always save the latest checkpoint (for resuming after interruption)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            {"best_acc": best_acc},
            model_dir / "checkpoint_latest.pt",
        )

    # ----- SAVE TRAINING HISTORY -----
    # This JSON file can be used to plot learning curves
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ----- FINAL SUMMARY -----
    print("\n" + "=" * 50)
    print("✅ Training complete!")
    print(f"🏆 Best test accuracy: {best_acc:.2f}%")
    print(f"📁 Model saved to: {model_dir / 'best_model.pt'}")
    print(f"📁 Confusion matrix saved to: {model_dir / 'confusion_matrix.npy'}")
    print(f"📁 Training history saved to: {model_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
