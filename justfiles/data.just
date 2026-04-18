set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n📦 Data Module\n"
    @printf "==============\n\n"
    @printf "  just data fetch          - Download MNIST and save as parquet\n"
    @printf "  just data info           - Show dataset statistics\n"
    @printf "  just data check          - Verify data exists\n"
    @printf "  just data preview        - Show first 5 rows of parquet\n\n"
    @printf "Examples:\n"
    @printf "  just data fetch\n"
    @printf "  just data info\n"

fetch:
    @echo "📥 Fetching MNIST dataset..."
    @uv run python ../src/data/fetch.py

info:
    @uv run python ../src/data/fetch.py --info

check:
    @uv run python ../src/data/fetch.py --info 2>&1 | grep -q "Data not found" && echo "❌ Data not found. Run 'just data fetch' first." || echo "✅ Data found"