"""
1. Purpose: This script generates the embeddings for the study texts segments (1-4), reduces the dimensionality from 768 to 32 using PCA, and exports the embeddings as .parquet files for further use in R.

Use:
uv run python embedding-extraction_dimensionality-reduction_export.py --part 1 # only Part 1 will be processed
uv run python embedding-extraction_dimensionality-reduction_export.py # without specifying --part, all 4 parts will be processed
""" 


#load libraries
import argparse
from dotenv import load_dotenv
from pathlib import Path
import sklearn.decomposition
import pandas as pd
import os
from setfit import SetFitModel


## Storage location of the fine-tuned model: locally and on the Hugging Face Hub as a constant
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell" #local folder name of the fine-tuned model
DEFAULT_REPO_BASE = "juliusherzig/setfit-modernbert-studien" #folder name of the fine-tuned model on Hugging Face Hub


def process_part(i, output_dir):
    """Embedding Generation for each Text Segment(1-4)."""
    print(f"--- Start Embedding Generation for Segment {i} ---")

    # --------------------------
    # 1. Preparation: Load fine-tuned Sentence-Transformer and load texts
    # --------------------------

    # 1.1  Check if the fine-tuned model already exists in local storage and, otherwise download it from the 🤗 Hub
    # Check: Does the local folder for the fine-tuned model exist? If not, download from Hugging Face Hub.
    local_folder = f"{DEFAULT_LOCAL_PATH}_{i}" #local path of the fine-tuned model for segment part i
    if Path(local_folder).exists():
        model_source = local_folder
        print("loading local model from: ", local_folder)
    else:
        repo_id = f"{DEFAULT_REPO_BASE}{i}" 
        model_source = repo_id
        print("loading model from Hugging Face Hub: ", repo_id)

    model = SetFitModel.from_pretrained(model_source, trust_remote_code=True)

    # 1.2 Preparation of Texts: Load JSON text files and creates a Panda dataframe
    df = pd.read_json(f"data/studytextPart{i}.jsonl", lines=True) 

    # 1.3 Rename columns 
    df = df.rename(columns={ 
        "text": "text",
        "replicationSuccessSigDir": "label",
        "numberOriginal": "numberOriginal",
        "setfitSplit": "split"
    })

    # --------------------------
    # 2. Create embeddings with the fine-tuned model for all texts
    # --------------------------
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=4)  # shape: number of texts and embedding dimension 
    print(f"Original Embedding-Shape: {embeddings.shape}")

    # --------------------------
    # 3. PCA-based Dimension Reduction
    # --------------------------
    pca = sklearn.decomposition.PCA(n_components=32, random_state=42) #n_components can be maximum n_samples
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"Reduced Embedding-Shape: {embeddings_reduced.shape}")

    # --------------------------
    # 4️. Export
    # --------------------------
    # 4.1 Tranform Embedding into Dataframe
    emb_df = pd.DataFrame(embeddings_reduced)
    # 4.2 Name columns (dim_0, dim_1, ...)
    emb_df.columns = [f"dim_{e}" for e in range(emb_df.shape[1])]
    # 4.3 ID for merge with all other variables in R
    emb_df['numberOriginal'] = df['numberOriginal'].values
    # 4.4 Storage for Export
    file_path = os.path.join(output_dir, f"embeddingsPart{i}.parquet")
    #4.5 Export as .parquet-Datei
    emb_df.to_parquet(file_path)
    print(f"Embeddings für Part {i} wurden erfolgreich exportiert nach: {file_path}")


#Define the command that can be typed in the terminal to run this script
def main():
    load_dotenv() #loads the .env file to access the tokens for HuggingFace Hub where the model is downloaded from if it is not available locally

    parser = argparse.ArgumentParser() #create an empty argument parser to handle command line arguments
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4]) # option to specify the part number (1-4) to be processed. If no part is specified, all parts will be processed.
    args = parser.parse_args()

    #output directory for the embeddings
    output_dir = "output_embeddings"
    # creates output_dir if it does not exist yet
    os.makedirs(output_dir, exist_ok=True)

    if args.part:
        process_part(args.part, output_dir)
    else:
        for i in range(1, 5):
            process_part(i, output_dir)


main()