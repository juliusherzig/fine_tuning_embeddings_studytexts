"""
Enables Uploading and Downloading the trained SetFit-Models from and to Huggingface Hub.

To limit text length, the study texts had been separeted into 4 equal-sized parts. For each part, a single model was trained. 

Use:
    uv run hf_model_j.py push              # Upload all 4 local models to HuggingFace Hub
    uv run hf_model_j.py load              # Download Model from Hub
    uv run hf_model_j.py push --part 1     # Upload only Part 1
    uv run hf_model_j.py load --part 3     # Download only Part 3"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from setfit import SetFitModel

# Constants - adapt here
DEFAULT_REPO_BASE = "juliusherzig/setfit-modernbert-studien"
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell"

# Specifies whether the model for text part 1, 2, 3 or 4 or all should be pulled/uploaded
def process_part(command, part_number):
    """Processes a single part (1-4) for push or load."""
    repo_id = f"{DEFAULT_REPO_BASE}{part_number}"
    local_path = f"{DEFAULT_LOCAL_PATH}_{part_number}"
    
    if command == "push":
        if not Path(local_path).exists():
            print(f"⚠️  Part {part_number} skipped: Folder {local_path} not found.")
            return
        
        print(f"Uploading model for Part {part_number} to {repo_id}.")
        push_model(repo_id, local_path)


    elif command == "load":    
        print(f"Downloading model for Part {part_number} from {repo_id}.")
        download_model(repo_id, local_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

#Function for Uplaoding models to HuggingFace Hub and Downloading models from HuggingFace Hub
def push_model(repo_id: str, local_path: str) -> None:
    """Pushes a local model to HuggingFace.
    
    Args:
        repo_id: HuggingFace Repository ID (e.g., "user/model-name")
        local_path: Path to the local model directory
    """
    # Load token that authenticates the user for pushing to HuggingFace Hub
    token = os.getenv("HF_TOKEN_WRITE") #Token is stored in a separate .env file, so that it is not hardcoded in the code
    if not token:
        print("Error: HF_TOKEN_WRITE not found in .env")
        print("Create a .env file with your HuggingFace Token (write access)")
        sys.exit(1)
    
    login(token=token)
    
    # Check if the local model exists 
    if not Path(local_path).exists():
        print(f"Error: Local model not found: {local_path}")
        print("Train the model first with the training script before pushing to HuggingFace Hub.")
        sys.exit(1)
    
    print(f"Loading model from {local_path}...")
    model = SetFitModel.from_pretrained(local_path)
    
    print(f"Pushing model to {repo_id}...")
    model.push_to_hub(repo_id)
    
    print(f"Successfully uploaded: https://huggingface.co/{repo_id}")


def download_model (repo_id: str, local_path: str) -> None:
    """Downloads a model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace Repository ID (e.g., "user/model-name")
        local_path: Path to save the model
    """
    # Load token and authenticate (for private Repos)
    token = os.getenv("HF_TOKEN_READ") #Token is stored in a separate .env file, so that it is not hardcoded in the code
    if token:
        login(token=token)
        print("Authenticated with HF_TOKEN_READ")
    else:
        print("No HF_TOKEN_READ found - trying without authentication")
        print("(Only works for public models)")
    
    print(f"Loading model from {repo_id}...")
    model = SetFitModel.from_pretrained(repo_id, trust_remote_code=True)
    
    print(f"Saving model to {local_path}...")
    model.save_pretrained(local_path)
    
    print(f"Successfully downloaded: {local_path}")


def main():
    """Main function with argument parsing."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub Model Management for SetFit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

    
Use:
  uv run hf_model_j.py push           # Upload all 4 local models to HuggingFace Hub
  uv run hf_model_j.py load           # Download all 4 models from HuggingFace Hub
  uv run hf_model_j.py push --part 1  # Upload only the model for Part 1
  uv run hf_model_j.py load --part 3  # Download only the model for Part 3
"""
    )
    
    parser.add_argument("command", choices=["push", "load"], help="Action: Upload or download model")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], help="Specific part (1-4). If not specified, all parts will be processed.")

    args = parser.parse_args()

    if args.part:
        process_part(args.command, args.part)
    else:
        print("Processing all models (1-4)...")
        for i in range(1, 5):
            process_part(args.command, i)


if __name__ == "__main__":
    main()
