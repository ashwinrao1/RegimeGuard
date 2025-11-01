#!/bin/bash

# Setup script for Robust Portfolio Optimization System

echo "🚀 Setting up Robust Portfolio Optimization System"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is required but not installed"
    echo "Please install pip and try again"
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✅ Using $PIP_CMD for package installation"

# Install required packages
echo ""
echo "📦 Installing required packages..."
echo "This may take a few minutes..."

$PIP_CMD install numpy pandas scipy scikit-learn matplotlib seaborn
$PIP_CMD install yfinance fredapi cvxpy pulp hmmlearn
$PIP_CMD install jupyter notebook ipykernel
$PIP_CMD install pytest pytest-cov

echo ""
echo "✅ Package installation completed!"

# Create necessary directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/cache
mkdir -p data/processed
mkdir -p logs
mkdir -p output
mkdir -p plots
mkdir -p exports

echo "✅ Directory structure created!"

# Set up environment variables (optional)
echo ""
echo "🔧 Environment Setup (Optional)"
echo "==============================="
echo "For full functionality, you may want to set up:"
echo ""
echo "1. FRED API Key (for macroeconomic data):"
echo "   export FRED_API_KEY='your_api_key_here'"
echo "   Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
echo ""
echo "2. Add to your ~/.bashrc or ~/.zshrc for persistence:"
echo "   echo 'export FRED_API_KEY=\"your_api_key_here\"' >> ~/.bashrc"

# Test the installation
echo ""
echo "🧪 Testing installation..."
if python3 -c "import numpy, pandas, scipy, sklearn, matplotlib, cvxpy; print('✅ Core packages imported successfully')" 2>/dev/null; then
    echo "✅ Installation test passed!"
else
    echo "❌ Installation test failed"
    echo "Some packages may not have installed correctly"
    exit 1
fi

echo ""
echo "🎉 SETUP COMPLETED SUCCESSFULLY!"
echo "================================"
echo ""
echo "🚀 Quick Start Options:"
echo ""
echo "1. Run the quick demo:"
echo "   python3 run_example.py"
echo ""
echo "2. Run the complete example:"
echo "   python3 examples/complete_example.py"
echo ""
echo "3. Start with Jupyter notebooks:"
echo "   jupyter notebook notebooks/01_data_preparation.ipynb"
echo ""
echo "4. Explore the system:"
echo "   - Check out README.md for detailed documentation"
echo "   - Modify config.yaml to customize parameters"
echo "   - Look at examples/ directory for more examples"
echo ""
echo "💡 Pro Tips:"
echo "   • Set up FRED API key for real macroeconomic data"
echo "   • The system works with sample data even without API keys"
echo "   • Check the output/ directory after running examples"
echo ""
echo "Happy optimizing! 🎯"