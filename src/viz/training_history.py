#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    """Save figure as AVIF (with PNG fallback)."""
    temp_png = path.with_suffix(".png")
    fig.savefig(temp_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if AVIF_AVAILABLE:
        im = Image.open(temp_png)
        im.save(path.with_suffix(".avif"), "avif", quality=90)
        temp_png.unlink()
    else:
        print(f"   Saved as PNG: {temp_png}")
        print("   (Install pillow-avif-plugin for AVIF format)")


def load_history(history_path: Path):
    """Load training history JSON file."""
    if not history_path.exists():
        return None
    with open(history_path) as f:
        return json.load(f)


def plot_single_history(
    history, title_prefix: str, output_dir: Path, filename_suffix: str
):
    """
    Plot training curves for a single model.

    Creates a 2x2 grid showing:
    - Loss curves (train + test)
    - Accuracy curves (train + test)
    - Learning dynamics (loss gap)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Loss curves
    axes[0, 0].plot(
        epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2
    )
    axes[0, 0].plot(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"{title_prefix} - Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[0, 1].plot(epochs, history["test_acc"], "r-", label="Test Acc", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title(f"{title_prefix} - Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([90, 100])

    # Plot 3: Overfitting gap (train - test accuracy)
    overfit_gap = [
        t - v for t, v in zip(history["train_acc"], history["test_acc"], strict=False)
    ]
    axes[1, 0].bar(epochs, overfit_gap, color="orange", alpha=0.7)
    axes[1, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Overfitting Gap (%)")
    axes[1, 0].set_title(f"{title_prefix} - Train-Test Accuracy Gap (Overfitting)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Loss gap
    loss_gap = [
        t - v for t, v in zip(history["train_loss"], history["test_loss"], strict=False)
    ]
    axes[1, 1].plot(epochs, loss_gap, "purple", marker="o", markersize=4, linewidth=2)
    axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss Gap (Train - Test)")
    axes[1, 1].set_title(f"{title_prefix} - Generalization Gap")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{title_prefix} - Training Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / f"training_analysis_{filename_suffix}"
    save_as_avif(fig, save_path)


def plot_comparison_relu_tanh(history_relu, history_tanh, output_dir: Path):
    """
    Compare ReLU vs Tanh training dynamics.

    Creates:
    - Overlapping loss curves
    - Overlapping accuracy curves
    - Convergence speed comparison
    - Final metrics bar chart
    """
    epochs_relu = range(1, len(history_relu["train_loss"]) + 1)
    epochs_tanh = range(1, len(history_tanh["train_loss"]) + 1)

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Test Accuracy Comparison (primary metric)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        epochs_relu,
        history_relu["test_acc"],
        "g-",
        label="ReLU",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs_tanh,
        history_tanh["test_acc"],
        "r-",
        label="Tanh",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy: ReLU vs Tanh")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([97, 100])

    # Highlight best accuracy
    best_relu = max(history_relu["test_acc"])
    best_tanh = max(history_tanh["test_acc"])
    best_relu_epoch = history_relu["test_acc"].index(best_relu) + 1
    best_tanh_epoch = history_tanh["test_acc"].index(best_tanh) + 1
    ax1.plot(
        best_relu_epoch,
        best_relu,
        "g*",
        markersize=15,
        label=f"ReLU Best: {best_relu:.2f}%",
    )
    ax1.plot(
        best_tanh_epoch,
        best_tanh,
        "r*",
        markersize=15,
        label=f"Tanh Best: {best_tanh:.2f}%",
    )

    # Plot 2: Test Loss Comparison
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(
        epochs_relu,
        history_relu["test_loss"],
        "g-",
        label="ReLU",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs_tanh,
        history_tanh["test_loss"],
        "r-",
        label="Tanh",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Test Loss: ReLU vs Tanh (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence Speed (epochs to reach accuracy thresholds)
    ax3 = plt.subplot(2, 2, 3)
    thresholds = [98.0, 98.5, 99.0]
    relu_epochs = []
    tanh_epochs = []

    for threshold in thresholds:
        # Find first epoch where test_acc >= threshold
        relu_epoch = next(
            (
                i + 1
                for i, acc in enumerate(history_relu["test_acc"])
                if acc >= threshold
            ),
            len(history_relu["test_acc"]),
        )
        tanh_epoch = next(
            (
                i + 1
                for i, acc in enumerate(history_tanh["test_acc"])
                if acc >= threshold
            ),
            len(history_tanh["test_acc"]),
        )
        relu_epochs.append(relu_epoch)
        tanh_epochs.append(tanh_epoch)

    x = np.arange(len(thresholds))
    width = 0.35
    bars1 = ax3.bar(
        x - width / 2, relu_epochs, width, label="ReLU", color="green", alpha=0.7
    )
    bars2 = ax3.bar(
        x + width / 2, tanh_epochs, width, label="Tanh", color="red", alpha=0.7
    )
    ax3.set_xlabel("Accuracy Threshold (%)")
    ax3.set_ylabel("Epochs to Reach")
    ax3.set_title("Convergence Speed: Epochs to Target Accuracy")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t}%" for t in thresholds])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

    # Plot 4: Final Metrics Bar Chart
    ax4 = plt.subplot(2, 2, 4)
    metrics = [
        "Best Test\nAcc (%)",
        "Final Test\nAcc (%)",
        "Min Test\nLoss",
        "Overfitting\nGap (%)",
    ]
    relu_values = [
        best_relu,
        history_relu["test_acc"][-1],
        min(history_relu["test_loss"]),
        history_relu["train_acc"][-1] - history_relu["test_acc"][-1],
    ]
    tanh_values = [
        best_tanh,
        history_tanh["test_acc"][-1],
        min(history_tanh["test_loss"]),
        history_tanh["train_acc"][-1] - history_tanh["test_acc"][-1],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width / 2, relu_values, width, label="ReLU", color="green", alpha=0.7)
    ax4.bar(x + width / 2, tanh_values, width, label="Tanh", color="red", alpha=0.7)
    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Value")
    ax4.set_title("Final Performance Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    
    # Set Y-axis limits to 95-100 for better visibility of accuracy differences
    ax4.set_ylim(95, 100)

    # Add improvement annotations
    improvement = best_relu - best_tanh
    plt.suptitle(
        f"ReLU vs Tanh: ReLU improves best accuracy by {improvement:.2f}%",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    save_path = output_dir / "training_comparison_relu_vs_tanh"
    save_as_avif(fig, save_path)

    # Print statistics to console
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: ReLU vs Tanh")
    print("=" * 60)
    print(
        f"Best Test Accuracy:     ReLU: {best_relu:.2f}% | Tanh: {best_tanh:.2f}% | Δ: +{improvement:.2f}%"
    )
    print(
        f"Final Test Accuracy:    ReLU: {history_relu['test_acc'][-1]:.2f}% | Tanh: {history_tanh['test_acc'][-1]:.2f}%"
    )
    print(
        f"Best Test Loss:         ReLU: {min(history_relu['test_loss']):.4f} | Tanh: {min(history_tanh['test_loss']):.4f}"
    )
    print(
        f"Overfitting Gap:        ReLU: {relu_values[3]:.2f}% | Tanh: {tanh_values[3]:.2f}%"
    )

    # Convergence analysis
    print("\nConvergence Speed (epochs to reach):")
    for i, thresh in enumerate(thresholds):
        print(
            f"  {thresh}%: ReLU={relu_epochs[i]} | Tanh={tanh_epochs[i]} | Δ={tanh_epochs[i]-relu_epochs[i]} epochs"
        )
    print("=" * 60)


def plot_comparison_xavier_relu(history_xavier, history_relu, history_tanh, output_dir: Path):
    """
    Compare Xavier vs ReLU vs Tanh training dynamics.

    Creates:
    - Overlapping loss curves (all three)
    - Overlapping accuracy curves (all three)
    - Convergence speed comparison
    - Final metrics bar chart
    """
    epochs_xavier = range(1, len(history_xavier["train_loss"]) + 1)
    epochs_relu = range(1, len(history_relu["train_loss"]) + 1)
    epochs_tanh = range(1, len(history_tanh["train_loss"]) + 1)

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Test Accuracy Comparison (primary metric)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        epochs_xavier,
        history_xavier["test_acc"],
        "b-",
        label="Xavier + ReLU",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs_relu,
        history_relu["test_acc"],
        "g-",
        label="ReLU (Default Init)",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )
    ax1.plot(
        epochs_tanh,
        history_tanh["test_acc"],
        "r-",
        label="Tanh (Original)",
        linewidth=2.5,
        marker="^",
        markersize=4,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy: Xavier+ReLU vs ReLU vs Tanh")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([97, 100])

    # Highlight best accuracy
    best_xavier = max(history_xavier["test_acc"])
    best_relu = max(history_relu["test_acc"])
    best_tanh = max(history_tanh["test_acc"])
    best_xavier_epoch = history_xavier["test_acc"].index(best_xavier) + 1
    best_relu_epoch = history_relu["test_acc"].index(best_relu) + 1
    best_tanh_epoch = history_tanh["test_acc"].index(best_tanh) + 1
    ax1.plot(
        best_xavier_epoch,
        best_xavier,
        "b*",
        markersize=15,
        label=f"Xavier Best: {best_xavier:.2f}%",
    )
    ax1.plot(
        best_relu_epoch,
        best_relu,
        "g*",
        markersize=15,
        label=f"ReLU Best: {best_relu:.2f}%",
    )
    ax1.plot(
        best_tanh_epoch,
        best_tanh,
        "r*",
        markersize=15,
        label=f"Tanh Best: {best_tanh:.2f}%",
    )

    # Plot 2: Test Loss Comparison
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(
        epochs_xavier,
        history_xavier["test_loss"],
        "b-",
        label="Xavier + ReLU",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs_relu,
        history_relu["test_loss"],
        "g-",
        label="ReLU (Default Init)",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )
    ax2.plot(
        epochs_tanh,
        history_tanh["test_loss"],
        "r-",
        label="Tanh (Original)",
        linewidth=2.5,
        marker="^",
        markersize=4,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Test Loss: Xavier+ReLU vs ReLU vs Tanh (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence Speed (epochs to reach accuracy thresholds)
    ax3 = plt.subplot(2, 2, 3)
    thresholds = [98.0, 98.5, 99.0]
    xavier_epochs = []
    relu_epochs = []
    tanh_epochs = []

    for threshold in thresholds:
        # Find first epoch where test_acc >= threshold
        xavier_epoch = next(
            (
                i + 1
                for i, acc in enumerate(history_xavier["test_acc"])
                if acc >= threshold
            ),
            len(history_xavier["test_acc"]),
        )
        relu_epoch = next(
            (
                i + 1
                for i, acc in enumerate(history_relu["test_acc"])
                if acc >= threshold
            ),
            len(history_relu["test_acc"]),
        )
        tanh_epoch = next(
            (
                i + 1
                for i, acc in enumerate(history_tanh["test_acc"])
                if acc >= threshold
            ),
            len(history_tanh["test_acc"]),
        )
        xavier_epochs.append(xavier_epoch)
        relu_epochs.append(relu_epoch)
        tanh_epochs.append(tanh_epoch)

    x = np.arange(len(thresholds))
    width = 0.25
    bars1 = ax3.bar(
        x - width, xavier_epochs, width, label="Xavier+ReLU", color="blue", alpha=0.7
    )
    bars2 = ax3.bar(
        x, relu_epochs, width, label="ReLU Only", color="green", alpha=0.7
    )
    bars3 = ax3.bar(
        x + width, tanh_epochs, width, label="Tanh", color="red", alpha=0.7
    )
    ax3.set_xlabel("Accuracy Threshold (%)")
    ax3.set_ylabel("Epochs to Reach")
    ax3.set_title("Convergence Speed: Epochs to Target Accuracy")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t}%" for t in thresholds])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

    # Plot 4: Final Metrics Bar Chart
    ax4 = plt.subplot(2, 2, 4)
    metrics = [
        "Best Test\nAcc (%)",
        "Final Test\nAcc (%)",
        "Min Test\nLoss",
        "Overfitting\nGap (%)",
    ]
    xavier_values = [
        best_xavier,
        history_xavier["test_acc"][-1],
        min(history_xavier["test_loss"]),
        history_xavier["train_acc"][-1] - history_xavier["test_acc"][-1],
    ]
    relu_values = [
        best_relu,
        history_relu["test_acc"][-1],
        min(history_relu["test_loss"]),
        history_relu["train_acc"][-1] - history_relu["test_acc"][-1],
    ]
    tanh_values = [
        best_tanh,
        history_tanh["test_acc"][-1],
        min(history_tanh["test_loss"]),
        history_tanh["train_acc"][-1] - history_tanh["test_acc"][-1],
    ]

    x = np.arange(len(metrics))
    width = 0.25
    bars1 = ax4.bar(
        x - width, xavier_values, width, label="Xavier+ReLU", color="blue", alpha=0.7
    )
    bars2 = ax4.bar(
        x, relu_values, width, label="ReLU Only", color="green", alpha=0.7
    )
    bars3 = ax4.bar(
        x + width, tanh_values, width, label="Tanh", color="red", alpha=0.7
    )
    ax4.set_xlabel("Metric")
    ax4.set_ylabel("Value")
    ax4.set_title("Final Performance Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    
    # Set Y-axis limits to 95-100 for better visibility of accuracy differences
    ax4.set_ylim(95, 100)

    # Add improvement annotations
    improvement_xavier_over_relu = best_xavier - best_relu
    improvement_relu_over_tanh = best_relu - best_tanh
    improvement_xavier_over_tanh = best_xavier - best_tanh
    plt.suptitle(
        f"Xavier+ReLU vs ReLU vs Tanh\n"
        f"Xavier improves over ReLU: +{improvement_xavier_over_relu:.2f}% | "
        f"ReLU improves over Tanh: +{improvement_relu_over_tanh:.2f}%",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save_path = output_dir / "training_comparison_xavier_vs_relu_vs_tanh"
    save_as_avif(fig, save_path)

    # Print statistics to console
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: Xavier+ReLU vs ReLU vs Tanh")
    print("=" * 60)
    print(
        f"Best Test Accuracy:     Xavier+ReLU: {best_xavier:.2f}% | ReLU: {best_relu:.2f}% | Tanh: {best_tanh:.2f}%"
    )
    print(
        f"Final Test Accuracy:    Xavier+ReLU: {history_xavier['test_acc'][-1]:.2f}% | ReLU: {history_relu['test_acc'][-1]:.2f}% | Tanh: {history_tanh['test_acc'][-1]:.2f}%"
    )
    print(
        f"Best Test Loss:         Xavier+ReLU: {min(history_xavier['test_loss']):.4f} | ReLU: {min(history_relu['test_loss']):.4f} | Tanh: {min(history_tanh['test_loss']):.4f}"
    )
    print(
        f"Overfitting Gap:        Xavier+ReLU: {xavier_values[3]:.2f}% | ReLU: {relu_values[3]:.2f}% | Tanh: {tanh_values[3]:.2f}%"
    )

    # Convergence analysis
    print("\nConvergence Speed (epochs to reach):")
    for i, thresh in enumerate(thresholds):
        print(
            f"  {thresh}%: Xavier+ReLU={xavier_epochs[i]} | ReLU={relu_epochs[i]} | Tanh={tanh_epochs[i]}"
        )
    print("\nImprovements:")
    print(f"  Xavier+ReLU vs ReLU: +{improvement_xavier_over_relu:.2f}%")
    print(f"  ReLU vs Tanh: +{improvement_relu_over_tanh:.2f}%")
    print(f"  Xavier+ReLU vs Tanh: +{improvement_xavier_over_tanh:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training history")
    parser.add_argument(
        "--type",
        type=str,
        default="single",
        choices=["single", "compare"],
        help="What to visualize",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["tanh", "relu", "xavier"],
        help="Which model to visualize (for single mode)",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    model_dir = project_root / "models"
    output_dir = project_root / "visualisations"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check for history files with standard naming
    tanh_path = model_dir / "training_tanh_history.json"
    relu_path = model_dir / "training_relu_history.json"
    xavier_path = model_dir / "training_xavier_history.json"

    # Also check legacy naming
    if not tanh_path.exists():
        legacy_path = model_dir / "training_history.json"
        if legacy_path.exists():
            # If only one exists, assume it's tanh (original)
            tanh_path = legacy_path
            print("📝 Using legacy training_history.json as Tanh baseline")

    has_tanh = tanh_path.exists()
    has_relu = relu_path.exists()
    has_xavier = xavier_path.exists()

    # Handle single model visualization with --model flag
    if args.type == "single":
        if args.model == "tanh" and has_tanh:
            print("\n📈 Generating Tanh training analysis...")
            history = load_history(tanh_path)
            plot_single_history(history, "Tanh (Original LeNet-5)", output_dir, "tanh")
        elif args.model == "relu" and has_relu:
            print("\n📈 Generating ReLU training analysis...")
            history = load_history(relu_path)
            plot_single_history(history, "ReLU", output_dir, "relu")
        elif args.model == "xavier" and has_xavier:
            print("\n📈 Generating Xavier training analysis...")
            history = load_history(xavier_path)
            plot_single_history(history, "Xavier Init", output_dir, "xavier")
        elif args.model is None:
            if has_xavier:
                print("\n📈 Generating Xavier training analysis...")
                history = load_history(xavier_path)
                plot_single_history(history, "Xavier Init", output_dir, "xavier")
            elif has_relu:
                print("\n📈 Generating ReLU training analysis...")
                history = load_history(relu_path)
                plot_single_history(history, "ReLU", output_dir, "relu")
            elif has_tanh:
                print("\n📈 Generating Tanh training analysis...")
                history = load_history(tanh_path)
                plot_single_history(
                    history, "Tanh (Original LeNet-5)", output_dir, "tanh"
                )
            else:
                print("❌ No training history files found")
                return
        else:
            print(f"❌ {args.model} history file not found")
            return

    # Handle comparison
    elif args.type == "compare":
        # Check which comparisons are possible
        if has_xavier and has_relu and has_tanh:
            print("\n📊 Generating three-way comparison: Xavier+ReLU vs ReLU vs Tanh...")
            history_tanh = load_history(tanh_path)
            history_relu = load_history(relu_path)
            history_xavier = load_history(xavier_path)
            plot_comparison_xavier_relu(history_xavier, history_relu, history_tanh, output_dir)
        elif has_relu and has_tanh:
            print("\n📊 Generating ReLU vs Tanh comparison...")
            history_tanh = load_history(tanh_path)
            history_relu = load_history(relu_path)
            plot_comparison_relu_tanh(history_relu, history_tanh, output_dir)
        else:
            print("❌ Need at least two history files for comparison")
            print(f"   Tanh exists: {has_tanh}")
            print(f"   ReLU exists: {has_relu}")
            print(f"   Xavier exists: {has_xavier}")
            print("\n   Tip: Save your runs as:")
            print("     models/training_tanh_history.json")
            print("     models/training_relu_history.json")
            print("     models/training_xavier_history.json")
            return


if __name__ == "__main__":
    main()
