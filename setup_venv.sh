#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install system dependencies required for EC2
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv

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