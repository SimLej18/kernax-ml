#!/bin/bash
# Development environment setup script for Kernax using Conda

set -e  # Exit on error

echo "==================================="
echo "Kernax Development Setup (Conda)"
echo "==================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda version: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^Kernax "; then
    echo "Conda environment 'Kernax' already exists."
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n Kernax -y
    else
        echo "Using existing environment."
        SKIP_CREATE=true
    fi
fi

# Create conda environment if needed
if [ "$SKIP_CREATE" != "true" ]; then
    echo ""
    echo "Creating conda environment 'Kernax' with Python 3.14..."
    conda create -n Kernax python=3.14 -y
fi

# Activate environment
echo ""
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate Kernax

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in editable mode
echo ""
echo "Installing Kernax in editable mode..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import kernax; print(f'Kernax version: {kernax.__version__}')" || {
    echo "Error: Failed to import kernax"
    exit 1
}

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate Kernax"
echo ""
echo "Useful commands:"
echo "  make test          - Run tests"
echo "  make test-cov      - Run tests with coverage"
echo "  make lint          - Check code quality"
echo "  make format        - Format code"
echo ""
echo "See SETUP_CHECKLIST.md for next steps."
