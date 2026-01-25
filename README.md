# SetFit Fine-Tuning

Fine-tuning SetFit models with ModernBERT for text classification.

## Setup

Install dependencies (uses CUDA 12.4 and Python 3.12 by default):

```bash
uv sync
```

For other CUDA versions, edit `pyproject.toml` and change the pytorch index URL:
- CUDA 12.4: `https://download.pytorch.org/whl/cu124`
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CPU only: `https://download.pytorch.org/whl/cpu`

### Verify GPU Setup

After installation, verify your setup:

```bash
uv run python test_gpu.py
```

This will show:
- Python version compatibility
- CUDA availability and GPU info
- Triton installation status (required for torch.compile on GPU)
- Model loading test

## Running

```bash
uv run python 2modernbert_ver3_abstuerzschutz.py
```

## Data

Place JSONL data files in the `data/` directory.

## Cluster Deployment (Enroot)

Run this project on a GPU cluster using NVIDIA enroot containers with SSH access.

### Build Docker Image

**Option 1: GitHub Actions (Automated)**

The Docker image is automatically built and pushed to `ghcr.io` when changes to `cluster/deployment/` are pushed to main. You can also trigger a build manually from the Actions tab.

To set up:
1. Add `ROOT_PASSWORD` secret in your repo settings (Settings > Secrets > Actions)
2. The image will be published to `ghcr.io/<your-user>/setfit-finetuning:cuda-12.4`

**Option 2: Manual Build**

```bash
# Build
docker build -t setfit-finetuning -f cluster/deployment/Dockerfile .

# Tag and push to GitHub Container Registry
docker tag setfit-finetuning ghcr.io/<your-user>/setfit-finetuning:cuda-12.4
docker push ghcr.io/<your-user>/setfit-finetuning:cuda-12.4
```

### Cluster Setup (First Time)

```bash
# Run setup script with your GitHub credentials and SSH key
./cluster/setup.sh <GITHUB_USER> <GITHUB_TOKEN> "<SSH_PUBLIC_KEY>"
```

### Start Container

```bash
# Start persistent container (changes to container filesystem persist)
./cluster/enroot-start.sh -p 10022

# Or use the symlink created by setup.sh
~/enroot-start-setfit.sh -p 10022
```

The `--rw` flag ensures container modifications persist between restarts. Your home directory is mounted, so repo changes always persist.

### SSH Access

```bash
# From your local machine
ssh -p 10022 $USER@<cluster-node>

# Inside container: run training
cd ~/setfit_finetuning
uv sync
uv run python 2modernbert_ver3_abstuerzschutz.py
```
