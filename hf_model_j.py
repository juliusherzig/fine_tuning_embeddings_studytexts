"""
Enables Uploading and Downloading the fine-tuned SetFit-Models from and to Huggingface Hub.

To limit text length, the study texts had been separeted into 4 equal-sized parts. For each part, a single model was trained. 

Use:
    uv run hf_model_j.py push              # Upload all 4 local models to HuggingFace Hub
    uv run hf_model_j.py load              # Download Model from Hub
    uv run hf_model_j.py push --part 1     # Upload only Part 1
    uv run hf_model_j.py load --part 3     # Download only Part 3"""


import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import login
from setfit import SetFitModel

# Constants - adapt here
DEFAULT_REPO_BASE = "juliusherzig/setfit-modernbert-studien"
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell"

# Specifies whether the model for text part 1, 2, 3 or 4 or all should be pulled/uploaded
def process_part(command, part_number):
    """Processes a single part (1-4) for push or load."""
    repo_id = f"{DEFAULT_REPO_BASE}{part_number}" #combines the base repo with the part number to create the full repo ID in huggingface   
    local_path = f"{DEFAULT_LOCAL_PATH}_{part_number}" #comines the base local path with the part number to create the full locak path where the model is stored
    
    if command == "push": # if the command is push, the push-function by huggingface_hub is called to upload the model to the hub      
        print(f"Uploading model for Part {part_number} to {repo_id}.")
        push_model(repo_id, local_path)


    elif command == "load":    
        print(f"Downloading model for Part {part_number} from {repo_id}.")
        download_model(repo_id, local_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

#Function for Uplaoding models to HuggingFace Hub and Downloading models from HuggingFace Hub
def push_model(repo_id, local_path):

    # Load token that authenticates the user for pushing to HuggingFace Hub
    token = os.getenv("HF_TOKEN_WRITE") #Token is stored in a separate .env file, so that it is not hardcoded in the code 
    login(token=token) # log in to HuggingFace Hub using the token

    model = SetFitModel.from_pretrained(local_path) #load the model, stored locally in the local_path, to be pushed to HuggingFace Hub
    
    model.push_to_hub(repo_id) # use the huggingface_hub function to push this model to the specified repo_id on HuggingFace Hub
    
    print(f"Successfully uploaded: https://huggingface.co/{repo_id}")


def download_model (repo_id, local_path):

    # Load token and authenticate (for private Repos)
    token = os.getenv("HF_TOKEN_READ") #Token is stored in a separate .env file, so that it is not hardcoded in the code
    login(token=token) # log in to HuggingFace Hub using the token
    
    model = SetFitModel.from_pretrained(repo_id, trust_remote_code=True)  #load the model, stored on HuggingFace Hub in the repo_id, to be downloaded to the local_path
    
    model.save_pretrained(local_path) #storing the model in the predefined local path
    
    print(f"Successfully downloaded: {local_path}")


#Define the command that can be typed in the terminal to run this script
def main():
    """Main function with argument parsing."""
    load_dotenv() #loads the .env file to access the tokens for HuggingFace Hub
    
    parser = argparse.ArgumentParser() #creates an empty argument parser to handle command line arguments
    parser.add_argument("command", choices=["push", "load"]) #defines the command for the script, either push or load
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4]) # defines the command for the script, either push or load, and the part number (1-4) to be processed. If no part is specified, all parts will be processed.

    args = parser.parse_args() #parses the command line arguments and stores them 

    if args.part:
        process_part(args.command, args.part) #if a part is specified, process only that part
    else: #otherwise... 
        for i in range(1, 5):
            process_part(args.command, i) #run the script sequentially for all parts (1-4)


main()
