#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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


def save_as_avif(fig, path: Path, dpi=150):
    temp_png = path.with_suffix(".png")
    fig.savefig(temp_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if AVIF_AVAILABLE:
        im = Image.open(temp_png)
        im.save(path.with_suffix(".avif"), "avif", quality=90)
        temp_png.unlink()
    else:
        print(f"   Saved as PNG: {temp_png}")


def visualize_samples(num_samples: int, data_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = data_dir / "mnist_train.parquet"
    if not train_path.exists():
        print(f"❌ Data not found at {train_path}")
        return

    train_df = pl.read_parquet(train_path)
    samples = train_df.sample(n=min(num_samples, len(train_df)))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for idx, row in enumerate(samples.iter_rows()):
        if idx >= grid_size * grid_size:
            break
        row_idx = idx // grid_size
        col_idx = idx % grid_size
        pixels = np.array(row[1:]).reshape(28, 28)
        ax = axes[row_idx, col_idx] if grid_size > 1 else axes
        ax.imshow(pixels, cmap="gray")
        ax.set_title(f"Label: {row[0]}", fontsize=8)
        ax.axis("off")

    for idx in range(len(samples), grid_size * grid_size):
        row_idx = idx // grid_size
        col_idx = idx % grid_size
        if grid_size > 1:
            axes[row_idx, col_idx].axis("off")

    plt.suptitle(f"MNIST Sample Images ({num_samples} random examples)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "dataset_samples"
    save_as_avif(fig, save_path)


def visualize_class_distribution(data_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = data_dir / "mnist_train.parquet"
    test_path = data_dir / "mnist_test.parquet"

    if not train_path.exists():
        print(f"❌ Data not found at {train_path}")
        return

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    train_counts = train_df.group_by("label").agg(pl.len()).sort("label")
    test_counts = test_df.group_by("label").agg(pl.len()).sort("label")
    train_vals = [row[1] for row in train_counts.iter_rows()]
    test_vals = [row[1] for row in test_counts.iter_rows()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(range(10), train_vals, color="steelblue")
    ax1.set_xlabel("Digit")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Training Set (n={len(train_df):,})")
    ax1.set_xticks(range(10))

    for bar, count in zip(bars1, train_vals, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = ax2.bar(range(10), test_vals, color="coral")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Test Set (n={len(test_df):,})")
    ax2.set_xticks(range(10))

    for bar, count in zip(bars2, test_vals, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle("MNIST Class Distribution (Balanced Dataset)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "class_distribution"
    save_as_avif(fig, save_path)


def visualize_pixel_heatmap(data_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    train_path = data_dir / "mnist_train.parquet"

    if not train_path.exists():
        print(f"❌ Data not found at {train_path}")
        return

    train_df = pl.read_parquet(train_path)
    pixel_cols = [f"pixel_{i}" for i in range(784)]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for digit in range(10):
        row = digit // 5
        col = digit % 5
        digit_df = train_df.filter(pl.col("label") == digit)
        avg_pixels = digit_df.select(pixel_cols).mean().to_numpy().reshape(28, 28)
        axes[row, col].imshow(avg_pixels, cmap="hot", interpolation="nearest")
        axes[row, col].set_title(f"Digit {digit}", fontsize=12)
        axes[row, col].axis("off")

    plt.suptitle("Average Pixel Intensity per Digit ('Typical' Image)", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "pixel_heatmap"
    save_as_avif(fig, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="samples",
        choices=["samples", "distribution", "heatmap", "all"],
    )
    parser.add_argument("--num_samples", type=int, default=25)
    args = parser.parse_args()

    project_root = find_project_root()
    data_dir = project_root / "data"
    output_dir = project_root / "visualisations"

    if args.type == "samples" or args.type == "all":
        visualize_samples(args.num_samples, data_dir, output_dir)
    if args.type == "distribution" or args.type == "all":
        visualize_class_distribution(data_dir, output_dir)
    if args.type == "heatmap" or args.type == "all":
        visualize_pixel_heatmap(data_dir, output_dir)

    print("\n✨ Visualization complete!")


if __name__ == "__main__":
    main()
