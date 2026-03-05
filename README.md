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
uv run python setfit_modernbert_finetuning.py
```

## Generating fine-tuned embeddings

This script loads the fine-tuned SetFit/ModernBERT models from the Hugging Face Hub, 
encodes study texts into embeddings, and reduces their dimensionality using PCA.  
The reduced embeddings are exported as `.parquet` files so they can be used for prediction with external models along with other features.

```bash
uv run python embedding-extraction_dimensionality-reduction_export.py
```


## Data

Place JSONL data files in the `data/` directory.

## HuggingFace Hub

Share trained models via HuggingFace Hub.

### Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your HuggingFace tokens to `.env`:
   - `HF_TOKEN_WRITE`: Your token with write access (for pushing)
   - `HF_TOKEN_READ`: Token for downloading (can be a different user's token)

   Create tokens at: https://huggingface.co/settings/tokens

### Push Model

Upload a trained model to HuggingFace Hub:

```bash
uv run hf_model.py push
```

### Load Model

Download a model from HuggingFace Hub:

```bash
uv run hf_model.py load
```

The repository is configured as a constant in `hf_model.py`.

## Cluster Deployment (Enroot)

Run this project on a GPU cluster using NVIDIA enroot containers with SSH access.

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
