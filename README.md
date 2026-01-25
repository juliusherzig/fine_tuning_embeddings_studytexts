# SetFit Fine-Tuning

Fine-tuning SetFit models with ModernBERT for text classification.

## Setup

Install dependencies (uses CUDA 12.4 by default for GPU support):

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

This will show whether CUDA is available and which GPU is detected.

## Running

```bash
uv run python 2modernbert_ver3_abstuerzschutz.py
```

## Data

Place JSONL data files in the `data/` directory.
