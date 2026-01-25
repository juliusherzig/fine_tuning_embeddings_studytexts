#!/bin/bash
# ============================================================================
# Enroot Configuration Script for SetFit Finetuning
# ============================================================================
# This script is used by enroot to configure the container environment.
# It is passed to `enroot start -c <config>` and defines mounts, hooks, and rc.
#
# Environment Variables:
# - SSH_PORT: SSH port to use (default: 10022)
# - WORKSPACE_DIR: Workspace directory inside container (default: /workspace)
#
# Usage:
#   SSH_PORT=10023 enroot start --rw -c enroot-config.sh <container>
# ============================================================================

# Mount configuration
# Returns space-separated pairs: "host_path container_path"
mounts() {
    # Mount user's home directory (contains the repo)
    echo "${HOME} ${HOME}"
}

# Pre-start hooks
# Sets environment variables and performs setup before container starts
hooks() {
    # Export environment variables into the container
    echo "SSH_PORT=${SSH_PORT:-10022}" >> ${ENROOT_ENVIRON}
    echo "WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}" >> ${ENROOT_ENVIRON}
    echo "HOME_DIR=${HOME}" >> ${ENROOT_ENVIRON}
}

# Container initialization (rc)
# Runs when the container starts - sets up environment and starts services
rc() {
    # Display container info
    echo -e "\n"
    echo -e "\033[1;34m[==========================================]\033[0m"
    echo -e "\033[1;32m   SetFit Finetuning Container Started     \033[0m"
    echo -e "\033[1;34m[==========================================]\033[0m"
    echo -e "\n"
    echo -e "\033[1;33m  SSH Port:    \033[1;31m${SSH_PORT}\033[0m"
    echo -e "\033[1;33m  Workspace:   \033[1;31m${WORKSPACE_DIR}\033[0m"
    echo -e "\033[1;33m  Home:        \033[1;31m${HOME_DIR}\033[0m"
    echo -e "\n"

    # Create workspace symlink if it doesn't exist
    if [ ! -e "${WORKSPACE_DIR}" ]; then
        ln -s "${HOME_DIR}" "${WORKSPACE_DIR}"
    fi

    # Set HOME to workspace for convenience
    export HOME="${WORKSPACE_DIR}"

    # Define working directory (where the repo lives)
    WD="${HOME_DIR}/setfit_finetuning"

    # Start SSH server for remote access
    /usr/sbin/sshd -p ${SSH_PORT}
    echo -e "\033[1;32m  SSH server started on port ${SSH_PORT}\033[0m"
    echo -e "\033[1;33m  Connect with: ssh -p ${SSH_PORT} <user>@<cluster-node>\033[0m"
    echo -e "\n"

    # Navigate to working directory and start interactive shell
    if [ -d "${WD}" ]; then
        cd "${WD}"
        echo -e "\033[1;32m  Changed to: ${WD}\033[0m"
    else
        echo -e "\033[1;33m  Warning: ${WD} not found, staying in ${HOME}\033[0m"
        cd "${HOME}"
    fi

    exec bash
}
