#!/bin/bash
# ============================================================================
# Cluster Setup Script for SetFit Finetuning
# ============================================================================
# This script sets up the cluster environment for running setfit_finetuning
# in an enroot container with SSH access.
#
# What it does:
# 1. Configures enroot credentials for GitHub Container Registry (ghcr.io)
# 2. Configures Git credentials for cloning repositories
# 3. Adds your SSH public key for remote access to the container
# 4. Creates a symlink for easy container startup
#
# Usage:
#   ./setup.sh <GITHUB_USERNAME> <GITHUB_TOKEN> "<PUBLIC_SSH_KEY>"
#
# Arguments:
#   GITHUB_USERNAME  Your GitHub username
#   GITHUB_TOKEN     GitHub Personal Access Token (with read:packages scope)
#   PUBLIC_SSH_KEY   Your SSH public key (in quotes)
#
# Example:
#   ./setup.sh myuser ghp_xxxxxxxxxxxx "ssh-ed25519 AAAA... user@host"
# ============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -ne 3 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo ""
    echo "Usage: $0 <GITHUB_USERNAME> <GITHUB_TOKEN> \"<PUBLIC_SSH_KEY>\""
    echo ""
    echo "Arguments:"
    echo "  GITHUB_USERNAME  Your GitHub username"
    echo "  GITHUB_TOKEN     GitHub Personal Access Token (read:packages scope)"
    echo "  PUBLIC_SSH_KEY   Your SSH public key (in quotes)"
    echo ""
    echo "Example:"
    echo "  $0 myuser ghp_xxxxxxxxxxxx \"ssh-ed25519 AAAA... user@host\""
    exit 1
fi

# Read arguments
GITHUB_USERNAME=$1
GITHUB_TOKEN=$2
PUBLIC_SSH_KEY=$3

# Define paths
ENROOT_CONFIG_DIR="$HOME/.config/enroot"
ENROOT_CREDENTIALS_FILE="$ENROOT_CONFIG_DIR/.credentials"
GIT_CREDENTIALS_FILE="$HOME/.git-credentials"
SSH_DIR="$HOME/.ssh"
AUTHORIZED_KEYS_FILE="$SSH_DIR/authorized_keys"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SetFit Finetuning - Cluster Setup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# ============================================
# Step 1: Configure Enroot Credentials
# ============================================
echo -e "${YELLOW}[Step 1/4] Configuring Enroot credentials...${NC}"

if [ ! -d "$ENROOT_CONFIG_DIR" ]; then
    mkdir -p "$ENROOT_CONFIG_DIR"
    echo -e "  Created: $ENROOT_CONFIG_DIR"
fi

if [ ! -f "$ENROOT_CREDENTIALS_FILE" ]; then
    echo "machine ghcr.io login $GITHUB_USERNAME password $GITHUB_TOKEN" > "$ENROOT_CREDENTIALS_FILE"
    chmod 600 "$ENROOT_CREDENTIALS_FILE"
    echo -e "  ${GREEN}Created enroot credentials file${NC}"
else
    echo -e "  Enroot credentials already exist (skipped)"
fi

# ============================================
# Step 2: Configure Git Credentials
# ============================================
echo ""
echo -e "${YELLOW}[Step 2/4] Configuring Git credentials...${NC}"

if [ ! -f "$GIT_CREDENTIALS_FILE" ]; then
    git config --global credential.helper store
    echo "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com" > "$GIT_CREDENTIALS_FILE"
    chmod 600 "$GIT_CREDENTIALS_FILE"
    echo -e "  ${GREEN}Created Git credentials file${NC}"
else
    echo -e "  Git credentials already exist (skipped)"
fi

# ============================================
# Step 3: Add SSH Public Key
# ============================================
echo ""
echo -e "${YELLOW}[Step 3/4] Adding SSH public key...${NC}"

if [ ! -d "$SSH_DIR" ]; then
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    echo -e "  Created: $SSH_DIR"
fi

if [ ! -f "$AUTHORIZED_KEYS_FILE" ]; then
    touch "$AUTHORIZED_KEYS_FILE"
    chmod 600 "$AUTHORIZED_KEYS_FILE"
fi

# Add key if not already present
if ! grep -qF "$PUBLIC_SSH_KEY" "$AUTHORIZED_KEYS_FILE" 2>/dev/null; then
    echo "$PUBLIC_SSH_KEY" >> "$AUTHORIZED_KEYS_FILE"
    echo -e "  ${GREEN}Added SSH public key to authorized_keys${NC}"
else
    echo -e "  SSH key already in authorized_keys (skipped)"
fi

# ============================================
# Step 4: Create Convenience Symlink
# ============================================
echo ""
echo -e "${YELLOW}[Step 4/4] Creating convenience symlink...${NC}"

SYMLINK_PATH="$HOME/enroot-start-setfit.sh"
TARGET_PATH="$SCRIPT_DIR/enroot-start.sh"

if [ -L "$SYMLINK_PATH" ]; then
    rm "$SYMLINK_PATH"
fi

ln -s "$TARGET_PATH" "$SYMLINK_PATH"
chmod +x "$SYMLINK_PATH"
echo -e "  ${GREEN}Created: $SYMLINK_PATH -> $TARGET_PATH${NC}"

# ============================================
# Done
# ============================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Next steps:"
echo ""
echo -e "  1. ${YELLOW}Start the container:${NC}"
echo -e "     ~/enroot-start-setfit.sh -p 10022"
echo ""
echo -e "  2. ${YELLOW}SSH into the container (from local machine):${NC}"
echo -e "     ssh -p 10022 \$USER@<cluster-node>"
echo ""
echo -e "  3. ${YELLOW}Run training inside the container:${NC}"
echo -e "     cd ~/setfit_finetuning"
echo -e "     uv sync"
echo -e "     uv run python 2modernbert_ver3_abstuerzschutz.py"
echo ""
echo -e "${BLUE}============================================${NC}"
