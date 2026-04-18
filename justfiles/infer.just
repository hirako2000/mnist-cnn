set shell := ["bash", "-uc"]
set dotenv-load := true

[default]
default:
    @printf "\n🔮 Inference Module\n"
    @printf "==================\n\n"
    @printf "  just infer image FILE   - Predict on single image\n"
    @printf "  just infer random N     - Test on N random samples\n"
    @printf "  just infer batch DIR    - Predict all images in directory\n"
    @printf "  just infer interactive  - Interactive drawing mode\n\n"
    @printf "Examples:\n"
    @printf "  just infer image my_digit.png\n"
    @printf "  just infer random 20\n"
    @printf "  just infer batch test_images/\n"

image file:
    @echo "🔮 Predicting on {{file}}..."
    @PYTHONPATH=../src uv run python ../src/infer/predict.py --image "{{file}}"

random N="10":
    @echo "🎲 Testing on {{N}} random samples..."
    @PYTHONPATH=../src uv run python ../src/infer/predict.py --random --num {{N}}

batch dir:
    @echo "📁 Running batch inference on {{dir}}..."
    @PYTHONPATH=../src uv run python ../src/infer/predict.py --dir "{{dir}}"

interactive:
    @echo "✍️ Starting interactive mode..."
    @PYTHONPATH=../src uv run python ../src/infer/predict.py --interactive