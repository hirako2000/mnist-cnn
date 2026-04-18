set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n📈 Visualization Module\n"
    @printf "======================\n\n"
    @printf "  just viz samples [N]    - Random sample images (default 25)\n"
    @printf "  just viz distribution   - Class distribution chart\n"
    @printf "  just viz heatmap        - Average pixel heatmap\n"
    @printf "  just viz confusion      - Confusion matrix (requires trained model)\n"
    @printf "  just viz examples [N]   - Prediction examples (default 16)\n"
    @printf "  just viz all            - Generate all visualizations\n\n"
    @printf "Examples:\n"
    @printf "  just viz samples\n"
    @printf "  just viz samples 50\n"
    @printf "  just viz all\n"

# Dataset visualizations (no model needed)
samples N="25":
    @echo "🎨 Generating {{N}} sample images..."
    @uv run python ../src/viz/dataset.py --type samples --num_samples {{N}}

distribution:
    @echo "📊 Generating class distribution chart..."
    @uv run python ../src/viz/dataset.py --type distribution

heatmap:
    @echo "🔥 Generating pixel heatmap..."
    @uv run python ../src/viz/dataset.py --type heatmap

# Model prediction visualizations (Python scripts handle warnings)
confusion:
    @echo "🔄 Generating confusion matrix..."
    @uv run python ../src/viz/predictions.py --type confusion

examples N="16":
    @echo "🎯 Generating {{N}} prediction examples..."
    @uv run python ../src/viz/predictions.py --type examples --num {{N}}

# All visualizations - just run them, Python scripts handle missing model gracefully
all:
    @echo "📊 Generating all visualizations..."
    @echo ""
    @just --justfile viz.just samples
    @just --justfile viz.just distribution
    @just --justfile viz.just heatmap
    @echo ""
    @just --justfile viz.just confusion
    @just --justfile viz.just examples