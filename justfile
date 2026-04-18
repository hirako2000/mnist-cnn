# ====================================================================
# Main justfile - Dispatches to module-specific justfiles
# ====================================================================
set dotenv-load := true
set shell := ["bash", "-uc"]

[default]
default:
    @just --justfile justfiles/main.just

# Data module
data *args:
    @just --justfile justfiles/data.just {{args}}

# Visualization module
viz *args:
    @just --justfile justfiles/viz.just {{args}}

# Training module
train *args:
    @just --justfile justfiles/train.just {{args}}

# Inference module
infer *args:
    @just --justfile justfiles/infer.just {{args}}

export-onnx:
    uv run python src/export_onnx.py

# Linting module
lint *args:
    @just --justfile justfiles/lint.just {{args}}

clean:
    @echo "🧹 Cleaning up..."
    @rm -rf data/*.parquet models/*.pt output/*.avif output/*.png
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "✅ Clean complete"

all:
    @echo "🔧 Setting up project..."
    @just data fetch
    @just train quick
    @just viz all
    @echo "✅ Setup complete!"