#!/bin/bash
set -e

echo "================================"
echo " Cyber Risk Detection System"
echo " Yerevan State University 2025"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Create directories
echo "Creating directories..."
mkdir -p data/raw models outputs/plots outputs/reports config
echo "✅ Directories ready"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"

# Verify core imports
echo ""
echo "Verifying installation..."
python3 -c "
import pandas, numpy, sklearn, xgboost, lightgbm
import streamlit, reportlab, joblib, plotly
print('✅ All core packages OK')
print(f'  pandas {pandas.__version__}')
print(f'  sklearn {sklearn.__version__}')
print(f'  xgboost {xgboost.__version__}')
print(f'  lightgbm {lightgbm.__version__}')
print(f'  streamlit {streamlit.__version__}')
"

echo ""
echo "Starting Streamlit app..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run src/ui/app.py
