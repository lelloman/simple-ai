#!/bin/bash
set -e

# Deploy simple-ai-runner to a remote host via systemd user service
# Usage: ./deploy-runner.sh <host> [--build]

if [ -z "$1" ]; then
    echo "Usage: $0 <host> [--build]"
    echo "  host: SSH host (e.g., lelloman@192.168.1.103)"
    echo "  --build: Build the binary before deploying"
    exit 1
fi

HOST="$1"
BUILD=false

for arg in "$@"; do
    case $arg in
        --build)
            BUILD=true
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/target/release/simple-ai-runner"
SERVICE_FILE="$SCRIPT_DIR/simple-ai-runner.service"

# Build if requested or binary doesn't exist
if [ "$BUILD" = true ] || [ ! -f "$BINARY" ]; then
    echo "=== Building simple-ai-runner ==="
    cargo build --release -p simple-ai-runner --manifest-path "$PROJECT_DIR/Cargo.toml"
fi

echo "=== Deploying to $HOST ==="

# Stop the user service if running
echo "Stopping simple-ai-runner service..."
ssh "$HOST" "systemctl --user stop simple-ai-runner 2>/dev/null || true"

# Kill any rogue processes using port 8080
echo "Checking for processes on port 8080..."
ssh "$HOST" "fuser -k 8080/tcp 2>/dev/null || true"
sleep 1

# Upload the new binary
echo "Uploading binary..."
scp "$BINARY" "$HOST:~/simple-ai-runner-new"

# Replace the old binary
echo "Replacing binary..."
ssh "$HOST" "mv ~/simple-ai-runner-new ~/simple-ai-runner && chmod +x ~/simple-ai-runner"

# Ensure user systemd directory exists and install service
echo "Installing systemd user service..."
ssh "$HOST" "mkdir -p ~/.config/systemd/user"
scp "$SERVICE_FILE" "$HOST:~/.config/systemd/user/simple-ai-runner.service"
ssh "$HOST" "systemctl --user daemon-reload && systemctl --user enable simple-ai-runner"

# Start the service
echo "Starting simple-ai-runner service..."
ssh "$HOST" "systemctl --user start simple-ai-runner"

# Verify it's running
sleep 2
if ssh "$HOST" "systemctl --user is-active --quiet simple-ai-runner"; then
    echo "=== Deployment successful ==="
    ssh "$HOST" "~/simple-ai-runner --version" 2>/dev/null || true
    ssh "$HOST" "systemctl --user status simple-ai-runner --no-pager | head -10"
else
    echo "=== ERROR: simple-ai-runner failed to start ==="
    ssh "$HOST" "journalctl --user -u simple-ai-runner -n 20 --no-pager"
    exit 1
fi
