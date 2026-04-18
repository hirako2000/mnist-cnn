#!/usr/bin/env python3

import argparse
from pathlib import Path

import polars as pl
from datasets import load_dataset
from tqdm import tqdm


def find_project_root(marker="pyproject.toml"):
    """Find project root by looking for marker file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find {marker}")


def fetch_mnist(save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)

    print("📥 Loading MNIST from Hugging Face...")
    dataset = load_dataset("ylecun/mnist", trust_remote_code=True)

    for split in ["train", "test"]:
        print(f"💾 Processing {split} split...")

        data = []
        for item in tqdm(dataset[split], desc=f"Converting {split}"):
            pixels = list(item["image"].getdata())
            row = {"label": item["label"]}
            for i, val in enumerate(pixels):
                row[f"pixel_{i}"] = val
            data.append(row)

        df = pl.DataFrame(data)
        parquet_path = save_dir / f"mnist_{split}.parquet"
        df.write_parquet(parquet_path)
        print(f"✅ Saved {len(df)} samples to {parquet_path}")

    return dataset


def show_info(data_dir: Path):
    train_path = data_dir / "mnist_train.parquet"
    test_path = data_dir / "mnist_test.parquet"

    if not train_path.exists():
        print("❌ Data not found. Run 'just data fetch' first.")
        return

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    print("=" * 50)
    print("MNIST DATASET INFORMATION")
    print("=" * 50)
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Total: {len(train_df) + len(test_df):,}")
    print()

    train_dist = train_df.group_by("label").agg(pl.len()).sort("label")
    test_dist = test_df.group_by("label").agg(pl.len()).sort("label")

    print("Class distribution (train):")
    for row in train_dist.iter_rows():
        print(f"  Digit {row[0]}: {row[1]:,} samples ({row[1]/len(train_df)*100:.1f}%)")

    print("\nClass distribution (test):")
    for row in test_dist.iter_rows():
        print(f"  Digit {row[0]}: {row[1]:,} samples ({row[1]/len(test_df)*100:.1f}%)")

    print("\n" + "=" * 50)

    pixel_cols = [f"pixel_{i}" for i in range(784)]
    pixel_data = train_df.select(pixel_cols).to_numpy()
    print("Image size: 28x28 = 784 pixels")
    print(f"Pixel value range: {pixel_data.min()} - {pixel_data.max()}")
    print(f"Mean pixel value: {pixel_data.mean():.2f}")
    print(f"Std pixel value: {pixel_data.std():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    project_root = find_project_root()
    data_dir = project_root / "data"

    if args.info:
        show_info(data_dir)
    else:
        fetch_mnist(data_dir)
        show_info(data_dir)


if __name__ == "__main__":
    main()
