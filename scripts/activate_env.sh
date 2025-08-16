#!/bin/bash
# ModelMuxer Virtual Environment Activation Script

echo "🚀 Activating ModelMuxer Virtual Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python: $(which python)"
echo "📦 Pip: $(which pip)"
echo ""
echo "🧪 To run tests: python test_enhanced_modelmuxer.py"
echo "🚀 To start server: uvicorn app.main_enhanced:app --reload"
echo "🔧 To deactivate: deactivate"
echo ""
