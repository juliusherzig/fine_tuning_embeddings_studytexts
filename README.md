# SetFit Fine-Tuning

* **Fine-tuning modernBERT** (nomic-ai/modernbert-embed-base) for text classification using SetFit. The model is fine-tuned to distinguish between **replicable and non-replicable studies** based on text parts of the original article referring to the study that had undergone a replication attempt. One **separate model is trained for each of four equal-sized segments of the study text**. The segmentation has to be performed in advance (see Data). The segmented approach serves, first, to prevent exceeding ModernBERT’s token limit and, second, to allow for a more sensitive detection of segment-specific differences between the two text groups.
After fine-tuning, all text segments are embedded by the respective fine-tuned model. Than a Primary Component Analysis (PCA) is applied for dimension reduction and the compressed embeddings are exported to be used among other features in a prediction model for replicability.

## Data

Place JSONL data files in the `data/` directory.
Note: The study texts should be separeted in four equal-sized segments in advance

### Requirements:
* **Naming:** Files must be named `studytextPart{i}.jsonl` (where `i` is 1 to 4).
* **Content:** Each JSONL file should contain the following fields:
    * `text`: The specific study text segment.
    * `label`: Replicability label (0 for non-replicable, 1 for replicable).
    * `id`: Unique identifier for the study.
    * `split` (optional): "train" or "test" to ensure consistent data splitting across all four segments during fine-tuning. 

## Setup

Install dependencies (uses CUDA 12.4 and Python 3.12 by default):

```bash
uv sync
```

## Run Fine-Tuning

This script trains a fine-tuned embedding model based on ModernBERT using SetFit for each of the four study text segments.

```bash
uv run python setfit_modernbert_finetuning.py            # Train all 4 parts sequentially
uv run python setfit_modernbert_finetuning.py --part 1    # Train only Part 1
```

Use `--part` to run multiple parts in parallel (e.g. in separate terminal sessions or SLURM jobs).

## Generate fine-tuned embeddings

This script loads the SetFit-fine-tuned ModernBERT models (from local folder or Hugging Face Hub), encodes study texts into embeddings, and reduces their dimensionality using PCA.  The reduced embeddings are exported as `.parquet` files to be used for prediction with external models along with other features.
**This script can be used to embed new texts with the fine-tuned models.**

```bash
uv run python embedding-extraction_dimensionality-reduction_export.py            # All 4 parts
uv run python embedding-extraction_dimensionality-reduction_export.py --part 2   # Only Part 2
```

## Fine-tuned Models on HuggingFace Hub

The models that were fine-tuned in script setfit_modernbert_finetuning.py and used in the embedding scripts can also be accessed independently via Hugging Face Hub for external inference: embedding generation and SetFit-integrated classification head.

### Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your HuggingFace tokens to `.env`:
   - `HF_TOKEN_WRITE`: Your token with write access (for pushing)
   - `HF_TOKEN_READ`: Token for downloading (can be a different user's token)

   Create tokens at: https://huggingface.co/settings/tokens

### Load Model

Download the fine-tuned model from HuggingFace Hub. 

```bash
uv run hf_model_j.py load #load the models for all text segments (1-4) 
uv run hf_model_j.py load --part 2  # download only the model for Part 2
```

### Push Model

Upload (another) model that was trained in setfit_modernbert_finetuning.py to HuggingFace Hub:

```bash
uv run hf_model_j.py push #upload the models for all text segments (1-4)
uv run hf_model_j.py push --part 1  # upload only the model for Part 1
```

*Note: The scripts provided here are configured for local execution. For the actual training of the embedding-models, the code was optimized for GPU-accelerated cluster infrastructure, generously provided and supported by Ivo Zilkenat.*
