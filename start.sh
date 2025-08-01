#!/bin/bash

# ModelMuxer Startup Script
# This script helps you get started with the LLM Router API

set -e

echo "🚀 ModelMuxer LLM Router Startup"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Please edit .env file with your API keys before continuing."
        echo "   Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, or MISTRAL_API_KEY"
        echo ""
        read -p "Press Enter after you've configured your API keys..."
    else
        echo "❌ .env.example file not found!"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if at least one API key is configured
echo "🔑 Checking API key configuration..."
source .env

API_KEYS_CONFIGURED=0
if [ ! -z "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "sk-your-openai-key-here" ]; then
    echo "✅ OpenAI API key configured"
    API_KEYS_CONFIGURED=1
fi

if [ ! -z "$ANTHROPIC_API_KEY" ] && [ "$ANTHROPIC_API_KEY" != "sk-ant-your-anthropic-key-here" ]; then
    echo "✅ Anthropic API key configured"
    API_KEYS_CONFIGURED=1
fi

if [ ! -z "$MISTRAL_API_KEY" ] && [ "$MISTRAL_API_KEY" != "your-mistral-key-here" ]; then
    echo "✅ Mistral API key configured"
    API_KEYS_CONFIGURED=1
fi

if [ $API_KEYS_CONFIGURED -eq 0 ]; then
    echo "❌ No API keys configured! Please edit .env file with valid API keys."
    exit 1
fi

# Start the server
echo ""
echo "🎯 Starting ModelMuxer LLM Router..."
echo "   Server will be available at: http://localhost:8000"
echo "   Health check: http://localhost:8000/health"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
