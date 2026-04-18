#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from train.model import LeNet5


def find_project_root(marker="pyproject.toml"):
    """Find project root by looking for marker file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find {marker}")


def resolve_path(path_str: str, project_root: Path) -> Path:
    """Resolve a path. If relative, try current dir, then project root."""
    path = Path(path_str)
    if path.exists():
        return path.resolve()
    # Try relative to project root
    project_path = project_root / path_str
    if project_path.exists():
        return project_path.resolve()
    return path  # Return original for error message


def load_model(model_path: str, device):
    model = LeNet5()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, force_invert: bool = None):
    """
    Preprocess a user-submitted image for MNIST model prediction.
    Args:
        image_path: Path to the image file
        force_invert: If True, force inversion. If False, don't invert.
                     If None, auto-detect based on image characteristics.
    Returns:
        Tensor ready for model input (1, 1, 28, 28)
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    # Auto-detect if we should invert
    if force_invert is None:
        # Calculate image statistics
        mean_val = img_array.mean()
        # Check if image already looks like MNIST format (white on black)
        # MNIST characteristics:
        # - Low mean (mostly black background, ~33)
        # - High contrast (bright digits, std dev typically > 60)
        looks_like_mnist = mean_val < 80 and img_array.std() > 50
        if not looks_like_mnist:
            # Assume user uploaded dark-on-light (pen on paper)
            print(f"   Auto-inverting: detected light background (mean={mean_val:.1f})")
            img_array = 255 - img_array
        else:
            print(
                f"   Keeping as-is: already looks like MNIST format (mean={mean_val:.1f})"
            )
    elif force_invert:
        print("   Force-inverting image")
        img_array = 255 - img_array
    # Resize to 28x28 (MNIST standard)
    img = Image.fromarray(img_array)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    # Convert to array and normalize
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    # Add batch and channel dimensions: (28,28) -> (1,1,28,28)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence, probs.cpu().numpy()[0]


def predict_random_samples(num_samples: int, data_dir: Path, model, device):
    """Test model on random samples from test set."""
    test_path = data_dir / "mnist_test.parquet"
    if not test_path.exists():
        print(f"❌ Test data not found at {test_path}")
        return

    df = pl.read_parquet(test_path)
    samples = df.sample(n=min(num_samples, len(df)))

    correct = 0
    for idx, row in enumerate(samples.iter_rows()):
        label = row[0]
        pixels = np.array(row[1:], dtype=np.float32)
        image = pixels.reshape(1, 28, 28)
        image = image / 255.0
        image = (image - 0.1307) / 0.3081
        tensor = torch.from_numpy(image).unsqueeze(0)

        pred, confidence, _ = predict(model, tensor, device)
        is_correct = pred == label
        correct += is_correct

        print(
            f"Sample {idx+1}: True={label}, Pred={pred}, Confidence={confidence:.2%}, {'✓' if is_correct else '✗'}"
        )

    print(f"\nAccuracy on {num_samples} random samples: {correct/num_samples:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--dir", type=str)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    args = parser.parse_args()

    project_root = find_project_root()
    model_path = project_root / args.model
    data_dir = project_root / "data"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run 'just train' first.")
        return

    model = load_model(str(model_path), device)
    print(f"✅ Model loaded from {model_path}")

    if args.interactive:
        print("✍️ Interactive mode coming soon...")
        return

    if args.image:
        image_path = resolve_path(args.image, project_root)
        if not image_path.exists():
            print(f"❌ Image not found: {args.image}")
            return
        tensor = preprocess_image(image_path)
        pred, conf, all_probs = predict(model, tensor, device)
        print(f"\n🔮 Prediction: {pred}")
        print(f"📊 Confidence: {conf:.2%}")
        print("\n📈 Probabilities:")
        for i, p in enumerate(all_probs):
            bar = "█" * int(p * 50)
            print(f"   {i}: {p:.2%} {bar}")
        return

    if args.dir:
        dir_path = resolve_path(args.dir, project_root)
        if not dir_path.exists():
            print(f"❌ Directory not found: {args.dir}")
            return
        for img_path in dir_path.glob("*.*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                tensor = preprocess_image(img_path)
                pred, conf, _ = predict(model, tensor, device)
                print(f"{img_path.name}: {pred} ({conf:.2%})")
        return

    if args.random:
        print(f"🎲 Testing on {args.num} random samples...")
        predict_random_samples(args.num, data_dir, model, device)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
