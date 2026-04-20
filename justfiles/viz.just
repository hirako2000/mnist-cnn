set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n📈 Visualization Module\n"
    @printf "======================\n\n"
    @printf "Dataset Visualizations:\n"
    @printf "  just viz samples [N]    - Random sample images (default 25)\n"
    @printf "  just viz distribution   - Class distribution chart\n"
    @printf "  just viz heatmap        - Average pixel heatmap\n\n"
    @printf "Model Prediction Visualizations (requires trained model):\n"
    @printf "  just viz confusion      - Confusion matrix\n"
    @printf "  just viz examples [N]   - Prediction examples (default 16)\n\n"
    @printf "Training History Visualizations (requires training_history.json):\n"
    @printf "  just viz history        - Training curves (loss/accuracy)\n"
    @printf "  just viz history-tanh   - Tanh model analysis\n"
    @printf "  just viz history-relu   - ReLU model analysis\n"
    @printf "  just viz history-xavier   - Xavier model analysis\n"
    @printf "  just viz history-compare        -  From pure LeNET comparison\n\n"
    @printf "Batch Generation:\n"
    @printf "  just viz all            - Generate all dataset + prediction visualizations\n"
    @printf "  just viz all-history    - Generate all training history visualizations\n"
    @printf "  just viz everything     - Generate ALL visualizations\n\n"
    @printf "Examples:\n"
    @printf "  just viz samples\n"
    @printf "  just viz samples 50\n"
    @printf "  just viz history-compare\n"
    @printf "  just viz all-history\n"

# ============================================
# Dataset visualizations (no model needed)
# ============================================
samples N="25":
    @echo "🎨 Generating {{N}} sample images..."
    @uv run python ../src/viz/dataset.py --type samples --num_samples {{N}}

distribution:
    @echo "📊 Generating class distribution chart..."
    @uv run python ../src/viz/dataset.py --type distribution

heatmap:
    @echo "🔥 Generating pixel heatmap..."
    @uv run python ../src/viz/dataset.py --type heatmap

# ============================================
# Model prediction visualizations
# ============================================
confusion:
    @echo "🔄 Generating confusion matrix..."
    @uv run python ../src/viz/predictions.py --type confusion

examples N="16":
    @echo "🎯 Generating {{N}} prediction examples..."
    @uv run python ../src/viz/predictions.py --type examples --num {{N}}

# ============================================
# Training history visualizations (NEW)
# ============================================
history:
    @echo "📈 Generating training history analysis..."
    @uv run python ../src/viz/training_history.py --type single

history-tanh:
    @echo "📈 Generating Tanh (original) training analysis..."
    @uv run python ../src/viz/training_history.py --type single --model tanh

history-relu:
    @echo "📈 Generating ReLU (improved) training analysis..."
    @uv run python ../src/viz/training_history.py --type single --model relu

history-xavier:
    @echo "📈 Generating ReLU (improved) training analysis..."
    @uv run python ../src/viz/training_history.py --type single --model xavier

history-compare:
    @echo "📊 Generating comparison..."
    @uv run python ../src/viz/training_history.py --type compare

# ============================================
# Batch generation
# ============================================
all:
    @echo "📊 Generating all dataset and prediction visualizations..."
    @echo ""
    @just --justfile viz.just samples
    @just --justfile viz.just distribution
    @just --justfile viz.just heatmap
    @echo ""
    @just --justfile viz.just confusion
    @just --justfile viz.just examples

all-history:
    @echo "📈 Generating all training history visualizations..."
    @echo ""
    @just --justfile viz.just history-tanh
    @just --justfile viz.just history-relu
    @just --justfile viz.just compare

everything:
    @echo "🌟 Generating ALL visualizations..."
    @just --justfile viz.just all
    @echo ""
    @just --justfile viz.just all-history