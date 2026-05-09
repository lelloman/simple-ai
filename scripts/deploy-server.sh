#!/bin/bash
set -euo pipefail

# Build, push, and optionally restart the simple-ai backend server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REGISTRY="${REGISTRY:-registry.homelab:5000}"
IMAGE="${IMAGE:-$REGISTRY/simple-ai-backend:latest}"
REMOTE="${REMOTE:-homelab}"
REMOTE_COMPOSE_DIR="${REMOTE_COMPOSE_DIR:-/home/lelloman/homelab/simple-ai}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-simple-ai}"

DRY_RUN=false
RESTART=true

usage() {
    cat <<EOF
Usage: $0 [options]

Build and deploy the simple-ai backend Docker image.

Options:
  --image <image>             Image tag to build and push (default: $IMAGE)
  --remote <host>             SSH host for docker compose restart (default: $REMOTE)
  --compose-dir <path>        Remote compose directory (default: $REMOTE_COMPOSE_DIR)
  --service <name>            Compose service name (default: $COMPOSE_SERVICE)
  --no-restart                Build and push only
  --dry-run                   Print commands without executing them
  -h, --help                  Show this help message

Environment overrides:
  REGISTRY, IMAGE, REMOTE, REMOTE_COMPOSE_DIR, COMPOSE_SERVICE
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --remote)
            REMOTE="$2"
            shift 2
            ;;
        --compose-dir)
            REMOTE_COMPOSE_DIR="$2"
            shift 2
            ;;
        --service)
            COMPOSE_SERVICE="$2"
            shift 2
            ;;
        --no-restart)
            RESTART=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == true ]]; then
        printf '[DRY-RUN]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "Building backend image: $IMAGE"
run docker build -f "$PROJECT_DIR/backend/Dockerfile" -t "$IMAGE" "$PROJECT_DIR"

echo "Pushing backend image: $IMAGE"
run docker push "$IMAGE"

if [[ "$RESTART" == true ]]; then
    echo "Restarting $COMPOSE_SERVICE on $REMOTE"
    run ssh "$REMOTE" "cd '$REMOTE_COMPOSE_DIR' && docker compose pull '$COMPOSE_SERVICE' && docker compose up -d '$COMPOSE_SERVICE'"
else
    echo "Skipping remote restart (--no-restart)"
fi
