# ====================================================================
# Main overview - Shows available commands
# ====================================================================
set dotenv-load := true

[default]
default:
    @printf "\n\033[1m🧠 MNIST CNN Project\033[0m\n"
    @printf "=======================================\n\n"
    @printf "\033[1m📦 DATA\033[0m\n"
    @printf "  just data fetch          - Download MNIST and save as parquet\n"
    @printf "  just data info           - Show dataset statistics\n"
    @printf "  just data check          - Verify data exists\n\n"
    
    @printf "\033[1m📈 VIZ\033[0m\n"
    @printf "  just viz samples         - Random sample images → AVIF\n"
    @printf "  just viz distribution    - Class distribution chart → AVIF\n"
    @printf "  just viz heatmap         - Average pixel heatmap → AVIF\n"
    @printf "  just viz confusion       - Confusion matrix → AVIF (requires trained model)\n"
    @printf "  just viz examples        - Prediction examples → AVIF (requires trained model)\n"
    @printf "  just viz all             - Generate all visualizations\n\n"
    
    @printf "\033[1m🎓 TRAIN\033[0m\n"
    @printf "  just train quick         - Quick test (1 epoch, 1000 samples)\n"
    @printf "  just train full          - Full training (10 epochs)\n"
    @printf "  just train mps           - Train on Apple GPU (MPS)\n"
    @printf "  just train cuda          - Train on Nvidia GPU\n"
    @printf "  just train resume        - Resume from checkpoint\n\n"
    
    @printf "\033[1m🔮 INFER\033[0m\n"
    @printf "  just infer image FILE    - Predict on single image\n"
    @printf "  just infer random        - Test on random samples\n"
    @printf "  just infer batch DIR     - Predict all images in directory\n\n"
    
    @printf "\033[1m✨ LINT\033[0m\n"
    @printf "  just lint fix            - Auto-fix linting issues\n"
    @printf "  just lint format         - Format code with black\n\n"
    
    @printf "\033[1m🧹 UTILITIES\033[0m\n"
    @printf "  just clean               - Remove generated files\n"
    @printf "  just setup               - Full project setup\n\n"
    
    @printf "=======================================\n"
    @printf "Run 'just <module>' to see module-specific help\n"
    @printf "Example: just data\n"