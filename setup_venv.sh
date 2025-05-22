#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base requirements
pip install -r requirements.txt

# Check if running on AWS EC2
if [ -f "/sys/hypervisor/uuid" ] && grep -q "ec2" /sys/hypervisor/uuid; then
    echo "AWS EC2 environment detected. Installing AWS-specific requirements..."
    pip install -r requirements-aws.txt
fi

# Print success message
echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate" 