#!/bin/bash

# MAGMA - Setup Script

echo "======================================"
echo "MAGMA: Multi-Graph based Agentic Memory Architecture"
echo "Setup Script"
echo "======================================"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p cache
mkdir -p results

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env and add your OPENAI_API_KEY"
fi

echo "======================================"
echo "Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Add your OpenAI API key to .env"
echo "3. Run example: python main.py --mode test --input examples/locomo_sample.json"
echo "======================================"