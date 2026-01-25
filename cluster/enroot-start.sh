#!/bin/bash
# ============================================================================
# Enroot Start Script for SetFit Finetuning
# ============================================================================
# This script imports a Docker image, creates a persistent container, and
# starts it with read-write access for development.
#
# Persistence: Using `enroot create` + `enroot start --rw` ensures that
# changes to the container filesystem persist between restarts.
#
# Usage:
#   ./enroot-start.sh [OPTIONS]
#
# Options:
#   -r, --registry REGISTRY   Docker registry (default: ghcr.io)
#   -i, --image IMAGE         Docker image name
#   -t, --tag TAG             Image tag (default: cuda-12.4)
#   -p, --port PORT           SSH port (default: 10022)
#   --root                    Enable root access in container
#   -h, --help                Show help
#
# Example:
#   ./enroot-start.sh -p 10023
#   ./enroot-start.sh -i myuser/setfit-finetuning -t latest
# ============================================================================

set -e

# Default values
DEFAULT_REGISTRY="ghcr.io"
DEFAULT_IMAGE="your-github-user/setfit-finetuning"  # <-- UPDATE THIS
DEFAULT_TAG="cuda-12.4"
DEFAULT_SSH_PORT="10022"
ROOT_ACCESS=false

# Display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Start a persistent enroot container for SetFit Finetuning."
    echo
    echo "Options:"
    echo "  -r, --registry REGISTRY   Docker registry (default: $DEFAULT_REGISTRY)"
    echo "  -i, --image IMAGE         Docker image (default: $DEFAULT_IMAGE)"
    echo "  -t, --tag TAG             Image tag (default: $DEFAULT_TAG)"
    echo "  -p, --port PORT           SSH port (default: $DEFAULT_SSH_PORT)"
    echo "      --root                Enable root access in container"
    echo "  -h, --help                Show this help"
    echo
    echo "Examples:"
    echo "  $0                        # Start with defaults"
    echo "  $0 -p 10023               # Use SSH port 10023"
    echo "  $0 -i user/img -t v1.0    # Custom image and tag"
    echo
    echo "Persistence:"
    echo "  The container is created with 'enroot create' and started with '--rw'."
    echo "  This means changes to the container filesystem persist between restarts."
    echo "  Your home directory is mounted, so repo changes are always preserved."
    exit 0
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--port)
            SSH_PORT="$2"
            shift 2
            ;;
        --root)
            ROOT_ACCESS=true
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

# Get script directory (for finding enroot-config.sh)
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Apply defaults
REGISTRY=${REGISTRY:-$DEFAULT_REGISTRY}
IMAGE=${IMAGE:-$DEFAULT_IMAGE}
TAG=${TAG:-$DEFAULT_TAG}
SSH_PORT=${SSH_PORT:-$DEFAULT_SSH_PORT}

# Construct full image path
FULL_IMAGE="$REGISTRY/$IMAGE"

# Container and squashfs file names
SQSH_FILE="$(basename $IMAGE | tr '/' '_')+$TAG.sqsh"
CONTAINER_NAME="$(basename $IMAGE | tr '/' '_')+$TAG.container"

# Set enroot data path
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.local/share/enroot/}"

# Export SSH port for config script
export SSH_PORT

echo "============================================"
echo "SetFit Finetuning - Enroot Container Start"
echo "============================================"
echo "Registry:   $REGISTRY"
echo "Image:      $IMAGE"
echo "Tag:        $TAG"
echo "SSH Port:   $SSH_PORT"
echo "Container:  $CONTAINER_NAME"
echo "============================================"

# Step 1: Import Docker image to squashfs (if not already done)
if [ ! -f "$SQSH_FILE" ]; then
    echo ""
    echo "[1/3] Importing Docker image: $FULL_IMAGE:$TAG"
    enroot import -o "$SQSH_FILE" "docker://$FULL_IMAGE:$TAG"
else
    echo ""
    echo "[1/3] Squashfs file already exists: $SQSH_FILE"
fi

# Step 2: Create container (if not already done)
# Check if container exists by listing containers
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "[2/3] Creating container: $CONTAINER_NAME"
    enroot create --name "$CONTAINER_NAME" "$SQSH_FILE"
else
    echo ""
    echo "[2/3] Container already exists: $CONTAINER_NAME"
fi

# Step 3: Start container with read-write access for persistence
echo ""
echo "[3/3] Starting container with --rw (persistent mode)..."
echo ""

# Build enroot start command
ENROOT_CMD="enroot start --rw"

# Add config script
ENROOT_CMD="$ENROOT_CMD -c \"$SCRIPT_DIR/enroot-config.sh\""

# Mount home directory
ENROOT_CMD="$ENROOT_CMD -m \"$HOME:$HOME\""

# Add root flag if requested
if [[ "$ROOT_ACCESS" == true ]]; then
    ENROOT_CMD="$ENROOT_CMD --root"
    echo "Warning: Running with --root. SSH server may not work in root mode."
fi

# Add container name
ENROOT_CMD="$ENROOT_CMD $CONTAINER_NAME"

# Execute
eval $ENROOT_CMD
