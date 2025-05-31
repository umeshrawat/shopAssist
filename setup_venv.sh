#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Python if not already installed
    if ! command -v python3 &> /dev/null; then
        echo "Installing Python..."
        brew install python
    fi
else
    # Install system dependencies required for Linux
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv
fi

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Print success message
echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate" 