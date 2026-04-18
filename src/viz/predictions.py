#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from model import LeNet5

from train import ParquetMNISTDataset

try:
    from PIL import Image
    from pillow_avif import AvifImagePlugin  # noqa: F401

    AVIF_AVAILABLE = True
except ImportError:
    AVIF_AVAILABLE = False


def find_project_root(marker="pyproject.toml"):
    """Find project root by looking for marker file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find {marker}")


def plot_confusion_matrix(cm, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(ax.images[0], ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=range(10),
        yticklabels=range(10),
        title="Confusion Matrix - MNIST Test Set",
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    save_path = output_dir / "confusion_matrix"
    if AVIF_AVAILABLE:
        temp_png = save_path.with_suffix(".png")
        fig.savefig(temp_png, dpi=150, bbox_inches="tight")
        im = Image.open(temp_png)
        im.save(save_path.with_suffix(".avif"), "avif", quality=90)
        temp_png.unlink()
        print(f"✅ Saved to {save_path}.avif")
    else:
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"✅ Saved to {save_path}.png")
    plt.close(fig)


def plot_prediction_examples(
    num_examples: int, data_dir: Path, model_dir: Path, output_dir: Path
):
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "best_model.pt"

    if not model_path.exists():
        print(f"❌ No trained model found at {model_path}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = LeNet5()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    train_path = data_dir / "mnist_train.parquet"
    if not train_path.exists():
        print(f"❌ Data not found at {train_path}")
        return

    test_dataset = ParquetMNISTDataset(str(data_dir / "mnist_test.parquet"))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    correct_idx = np.where(all_preds == all_labels)[0]
    incorrect_idx = np.where(all_preds != all_labels)[0]

    n_correct = num_examples // 2
    n_incorrect = num_examples - n_correct

    sampled_correct = np.random.choice(
        correct_idx, min(n_correct, len(correct_idx)), replace=False
    )
    sampled_incorrect = np.random.choice(
        incorrect_idx, min(n_incorrect, len(incorrect_idx)), replace=False
    )

    sampled_idx = list(sampled_correct) + list(sampled_incorrect)
    np.random.shuffle(sampled_idx)

    grid_size = int(np.ceil(np.sqrt(num_examples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for i, idx in enumerate(sampled_idx[:num_examples]):
        ax = axes[i // grid_size, i % grid_size] if grid_size > 1 else axes
        img = all_images[idx][0]
        img = (img * 0.3081) + 0.1307
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap="gray")
        is_correct = all_preds[idx] == all_labels[idx]
        color = "green" if is_correct else "red"
        ax.set_title(
            f"True: {all_labels[idx]} | Pred: {all_preds[idx]}", color=color, fontsize=9
        )
        ax.axis("off")

    for i in range(len(sampled_idx), grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size] if grid_size > 1 else axes
        ax.axis("off")

    plt.suptitle("Prediction Examples (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "prediction_examples"
    if AVIF_AVAILABLE:
        temp_png = save_path.with_suffix(".png")
        fig.savefig(temp_png, dpi=150, bbox_inches="tight")
        im = Image.open(temp_png)
        im.save(save_path.with_suffix(".avif"), "avif", quality=90)
        temp_png.unlink()
        print(f"✅ Saved to {save_path}.avif")
    else:
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"✅ Saved to {save_path}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="confusion",
        choices=["confusion", "examples", "all"],
    )
    parser.add_argument("--num", type=int, default=16)
    args = parser.parse_args()

    project_root = find_project_root()
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    output_dir = project_root / "visualisations"
    model_dir = project_root / "models"

    if args.type in ["confusion", "all"]:
        cm_path = model_dir / "confusion_matrix.npy"
        if cm_path.exists():
            cm = np.load(cm_path)
            plot_confusion_matrix(cm, output_dir)
        else:
            print(
                f"❌ Confusion matrix not found at {cm_path}. Run 'just train' first."
            )

    if args.type in ["examples", "all"]:
        plot_prediction_examples(args.num, data_dir, model_dir, output_dir)

    print("\n✨ Visualization complete!")


if __name__ == "__main__":
    main()
