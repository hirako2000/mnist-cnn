set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n🎓 Training Module\n"
    @printf "=================\n\n"
    @printf "  just train quick         - Quick test (1 epoch, 1000 samples)\n"
    @printf "  just train full          - Full training (10 epochs)\n"
    @printf "  just train mps [N]       - Train on MPS (default 5 epochs)\n"
    @printf "  just train cuda [N]      - Train on CUDA (default 5 epochs)\n"
    @printf "  just train resume        - Resume from checkpoint\n\n"
    @printf "Examples:\n"
    @printf "  just train\n"
    @printf "  just train mps 20\n"
    @printf "  just train mps\n"

# Default training (5 epochs, CPU)
train:
    @echo "🚀 Starting training (5 epochs, CPU)..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --epochs 5 --device cpu

# Quick test
quick:
    @echo "🧪 Running quick test..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --epochs 1 --max_train_samples 1000 --max_test_samples 200 --device cpu

# Full training (10 epochs, CPU)
full:
    @echo "🚀 Running full training (10 epochs)..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --epochs 10 --device cpu

# Train on MPS (default 5 epochs, or pass custom epoch count)
mps epochs="5":
    @echo "🚀 Training on MPS ({{epochs}} epochs)..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --epochs {{epochs}} --device mps

# Train on CUDA
cuda epochs="5":
    @echo "🚀 Training on CUDA ({{epochs}} epochs)..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --epochs {{epochs}} --device cuda

# Resume from checkpoint
resume:
    @echo "🔄 Resuming training from checkpoint..."
    @PYTHONPATH=../src uv run python ../src/train/train.py --resume models/checkpoint_latest.pt

# Show model info
info:
    @echo "📊 Model information:"
    @cd ../src/train && uv run python -c "from model import LeNet5; m = LeNet5(); print(f'Parameters: {m.get_num_params():,}')"