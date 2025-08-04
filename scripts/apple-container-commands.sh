#!/bin/bash
# ModelMuxer Apple Container Commands
# Native containerization for macOS 15+ Beta

set -e

echo "🍎 modelmuxer Apple Container Commands"
echo "======================================"

# Function to check if Apple Container is available
check_apple_container() {
    if ! command -v container &> /dev/null; then
        echo "❌ Apple Container is not available. Requirements:"
        echo "   - macOS 15 Beta or later"
        echo "   - Apple Silicon Mac (M1/M2/M3)"
        echo "   - Xcode Command Line Tools"
        echo ""
        echo "💡 Alternative: Use Podman or Docker"
        echo "   ./scripts/podman-commands.sh"
        exit 1
    fi
    echo "✅ Apple Container is available"
}

# Function to check macOS version
check_macos_version() {
    local version=$(sw_vers -productVersion)
    local major=$(echo $version | cut -d. -f1)

    if [[ $major -lt 15 ]]; then
        echo "⚠️  Warning: macOS $version detected. Apple Container requires macOS 15+"
        echo "   Consider using Podman: ./scripts/podman-commands.sh"
    else
        echo "✅ macOS $version is compatible"
    fi
}

# Function to build the image
build_image() {
    echo "🔨 Building modelmuxer image with Apple Container..."
    container build -t modelmuxer:latest -f Dockerfile.apple .
    echo "✅ Image built successfully"
}

# Function to run with compose
run_with_compose() {
    echo "🐳 Running with Apple Container Compose..."

    # Create volumes
    mkdir -p data logs

    if [[ -f "container-compose.yaml" ]]; then
        container compose -f container-compose.yaml up -d
        echo "✅ Container started with compose"
    else
        echo "❌ container-compose.yaml not found"
        exit 1
    fi
}

# Function to run container directly
run_container() {
    echo "🚀 Running modelmuxer container directly..."

    # Create volumes
    mkdir -p data logs

    # Run container with environment variables from .env
    container run -d \
        --name modelmuxer \
        --env-file .env \
        -e TESTING=false \
        -e DATABASE_URL=sqlite:///data/modelmuxer.db \
        -p 8000:8000 \
        -v ./data:/app/data \
        -v ./logs:/app/logs \
        --replace \
        modelmuxer:latest

    echo "✅ Container started successfully"
    echo "🌐 Application available at: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
}

# Function to stop and clean up
cleanup() {
    echo "🧹 Cleaning up containers..."
    container stop modelmuxer 2>/dev/null || true
    container rm modelmuxer 2>/dev/null || true
    echo "✅ Cleanup complete"
}

# Function to show logs
show_logs() {
    echo "📋 Showing container logs..."
    container logs -f modelmuxer
}

# Function to test the application
test_app() {
    echo "🧪 Testing modelmuxer application..."

    # Wait for container to start
    sleep 5

    # Test health endpoint
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        return 1
    fi

    # Test providers endpoint (with auth)
    if curl -s -H "Authorization: Bearer sk-test-key-1" http://localhost:8000/providers | grep -q "providers"; then
        echo "✅ Providers endpoint working"
    else
        echo "❌ Providers endpoint failed"
        return 1
    fi

    echo "🎉 All tests passed!"
}

# Function to show container info
info() {
    echo "📊 Apple Container Information:"
    echo "Container version: $(container --version 2>/dev/null || echo 'Unknown')"
    echo "System info:"
    echo "  macOS: $(sw_vers -productVersion)"
    echo "  Architecture: $(uname -m)"
    echo ""
    echo "Running containers:"
    container ps 2>/dev/null || echo "No containers running"
}

# Main script logic
case "${1:-help}" in
    "check")
        check_apple_container
        check_macos_version
        ;;
    "build")
        check_apple_container
        build_image
        ;;
    "run")
        check_apple_container
        build_image
        run_container
        ;;
    "compose")
        check_apple_container
        run_with_compose
        ;;
    "logs")
        show_logs
        ;;
    "test")
        test_app
        ;;
    "stop")
        cleanup
        ;;
    "restart")
        cleanup
        check_apple_container
        build_image
        run_container
        ;;
    "info")
        info
        ;;
    "help"|*)
        echo "Usage: $0 {check|build|run|compose|logs|test|stop|restart|info|help}"
        echo ""
        echo "Commands:"
        echo "  check    - Check Apple Container availability and macOS version"
        echo "  build    - Build the modelmuxer image"
        echo "  run      - Build and run container directly"
        echo "  compose  - Run with Apple Container Compose"
        echo "  logs     - Show container logs"
        echo "  test     - Test the running application"
        echo "  stop     - Stop and remove containers"
        echo "  restart  - Stop, rebuild, and restart"
        echo "  info     - Show system and container information"
        echo "  help     - Show this help message"
        echo ""
        echo "🍎 Apple Container requires macOS 15+ Beta"
        echo "💡 For older macOS versions, use: ./scripts/podman-commands.sh"
        ;;
esac
