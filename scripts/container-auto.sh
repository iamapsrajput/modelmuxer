#!/bin/bash
# ModelMuxer Auto-Container Detection Script
# Automatically detects and uses the best available containerization system

set -e

echo "🔍 modelmuxer Auto-Container Detection"
echo "======================================"

# Function to detect the best containerization system
detect_container_system() {
    local system=""
    local reason=""

    # Check for Apple Container (highest priority on macOS 15+)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local macos_version=$(sw_vers -productVersion | cut -d. -f1)
        if [[ $macos_version -ge 15 ]] && command -v container &> /dev/null; then
            system="apple"
            reason="Apple Container available on macOS 15+"
        elif command -v podman &> /dev/null; then
            system="podman"
            reason="Podman available (recommended for macOS)"
        elif command -v docker &> /dev/null; then
            system="docker"
            reason="Docker available (fallback)"
        fi
    else
        # Non-macOS systems
        if command -v podman &> /dev/null; then
            system="podman"
            reason="Podman available (recommended for Linux)"
        elif command -v docker &> /dev/null; then
            system="docker"
            reason="Docker available"
        fi
    fi

    if [[ -z "$system" ]]; then
        echo "❌ No containerization system found!"
        echo ""
        echo "Please install one of the following:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  1. Apple Container (macOS 15+ Beta)"
            echo "  2. Podman: brew install podman"
            echo "  3. Docker Desktop"
        else
            echo "  1. Podman: https://podman.io/getting-started/installation"
            echo "  2. Docker: https://docs.docker.com/get-docker/"
        fi
        exit 1
    fi

    echo "✅ Detected: $system ($reason)"
    echo "$system"
}

# Function to run the appropriate script
run_container_script() {
    local system=$1
    local command=$2

    case "$system" in
        "apple")
            echo "🍎 Using Apple Container..."
            ./scripts/apple-container-commands.sh "$command"
            ;;
        "podman")
            echo "🐳 Using Podman..."
            ./scripts/podman-commands.sh "$command"
            ;;
        "docker")
            echo "🐋 Using Docker..."
            if [[ -f "scripts/docker-commands.sh" ]]; then
                ./scripts/docker-commands.sh "$command"
            else
                echo "Running Docker commands directly..."
                case "$command" in
                    "run")
                        docker build -t modelmuxer:latest .
                        docker run -d --name modelmuxer --env-file .env -p 8000:8000 modelmuxer:latest
                        ;;
                    "stop")
                        docker stop modelmuxer 2>/dev/null || true
                        docker rm modelmuxer 2>/dev/null || true
                        ;;
                    "logs")
                        docker logs -f modelmuxer
                        ;;
                    *)
                        echo "Command '$command' not implemented for Docker fallback"
                        echo "Use docker commands directly or create scripts/docker-commands.sh"
                        ;;
                esac
            fi
            ;;
        *)
            echo "❌ Unknown container system: $system"
            exit 1
            ;;
    esac
}

# Function to show system information
show_info() {
    echo "🖥️  System Information:"
    echo "OS: $OSTYPE"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS Version: $(sw_vers -productVersion)"
        echo "Architecture: $(uname -m)"
    else
        echo "Kernel: $(uname -r)"
        echo "Architecture: $(uname -m)"
    fi
    echo ""

    echo "📦 Available Container Systems:"

    # Check Apple Container
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v container &> /dev/null; then
            echo "✅ Apple Container: $(container --version 2>/dev/null || echo 'Available')"
        else
            echo "❌ Apple Container: Not available"
        fi
    fi

    # Check Podman
    if command -v podman &> /dev/null; then
        echo "✅ Podman: $(podman --version)"
    else
        echo "❌ Podman: Not installed"
    fi

    # Check Docker
    if command -v docker &> /dev/null; then
        echo "✅ Docker: $(docker --version)"
    else
        echo "❌ Docker: Not installed"
    fi
}

# Main script logic
command="${1:-help}"

case "$command" in
    "info")
        show_info
        ;;
    "detect")
        detect_container_system
        ;;
    "help")
        echo "Usage: $0 {run|stop|logs|test|build|compose|info|detect|help}"
        echo ""
        echo "This script automatically detects the best available containerization system:"
        echo "  1. Apple Container (macOS 15+ Beta)"
        echo "  2. Podman (recommended alternative)"
        echo "  3. Docker (fallback)"
        echo ""
        echo "Commands:"
        echo "  run      - Build and run modelmuxer container"
        echo "  stop     - Stop and remove containers"
        echo "  logs     - Show container logs"
        echo "  test     - Test the running application"
        echo "  build    - Build container image only"
        echo "  compose  - Run with compose (if available)"
        echo "  info     - Show system and container information"
        echo "  detect   - Detect and show the best container system"
        echo "  help     - Show this help message"
        ;;
    *)
        system=$(detect_container_system)
        run_container_script "$system" "$command"
        ;;
esac
