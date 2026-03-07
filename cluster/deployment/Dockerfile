# SetFit Finetuning - CUDA Docker Image
# Build: docker build -t setfit-finetuning -f cluster/deployment/Dockerfile .
# Push:  docker tag setfit-finetuning ghcr.io/<user>/setfit-finetuning:cuda-12.8
#        docker push ghcr.io/<user>/setfit-finetuning:cuda-12.8

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Build argument for root password (set during build)
ARG ROOT_PASSWORD=changeme

# ==========================================
# System Dependencies
# ==========================================
RUN apt update && apt install -y \
    wget \
    git \
    vim \
    nano \
    curl \
    pciutils \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# UV Package Manager Installation
# ==========================================
# Install uv (extremely fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Verify uv installation
RUN uv --version

# ==========================================
# SSH Server Configuration
# ==========================================
# Configure SSH for remote access to the container
RUN mkdir -p /var/run/sshd && \
    echo 'Port 10022' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config && \
    echo "root:${ROOT_PASSWORD}" | chpasswd

# ==========================================
# SSH Host Keys (from repo for consistent fingerprint)
# ==========================================
COPY ./cluster/deployment/ssh/ed25519_key /etc/ssh/ssh_host_ed25519_key
COPY ./cluster/deployment/ssh/ed25519_key.pub /etc/ssh/ssh_host_ed25519_key.pub
COPY ./cluster/deployment/ssh/rsa_key /etc/ssh/ssh_host_rsa_key
COPY ./cluster/deployment/ssh/rsa_key.pub /etc/ssh/ssh_host_rsa_key.pub

# Set permissions and configure host keys
RUN chmod 600 /etc/ssh/ssh_host_* && \
    chown root:root /etc/ssh/ssh_host_* && \
    echo 'HostKey /etc/ssh/ssh_host_ed25519_key' >> /etc/ssh/sshd_config && \
    echo 'HostKey /etc/ssh/ssh_host_rsa_key' >> /etc/ssh/sshd_config

# ==========================================
# SSH Banner
# ==========================================
COPY ./cluster/deployment/ssh/banner /etc/ssh/sshd_banner
RUN chmod 644 /etc/ssh/sshd_banner && \
    chown root:root /etc/ssh/sshd_banner && \
    echo 'Banner /etc/ssh/sshd_banner' >> /etc/ssh/sshd_config

# ==========================================
# Working Directory Setup
# ==========================================
WORKDIR /workspace

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
