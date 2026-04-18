# ====================================================================
# Linting Module - Code quality and formatting
# ====================================================================
set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n✨ Linting Module\n"
    @printf "================\n\n"
    @printf "  just lint fix       - Auto-fix linting issues\n"
    @printf "  just lint format    - Format with black\n"
    @printf "  just lint check     - Check formatting without changing\n"
    @printf "  just lint all       - Run all checks\n\n"
    @printf "Examples:\n"
    @printf "  just lint fix\n"
    @printf "  just lint format\n"

# Run linter
[doc("Run ruff linter")]
lint:
    @echo "🔍 Running ruff linter..."
    @uv run ruff check ../src/

[doc("Auto-fix linting issues")]
fix:
    @echo "🔧 Auto-fixing linting issues..."
    @uv run ruff check --fix ../src/

[doc("Format code with black")]
format:
    @echo "📝 Formatting code..."
    @uv run black ../src/

[doc("Check formatting without changing")]
check:
    @echo "🔍 Checking formatting..."
    @uv run black --check ../src/

[doc("Run all linting and formatting checks")]
all:
    @echo "🔍 Running all checks..."
    @uv run ruff check ../src/
    @uv run black --check ../src/
    @echo "✅ All checks passed!"