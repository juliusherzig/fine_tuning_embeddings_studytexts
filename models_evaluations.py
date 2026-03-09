''''
Anwendung:
uv run python models_evaluations.py --part 1 # nur Part 1 wird verarbeitet
uv run python models_evaluations.py # ohne Angabe von --part werden alle 4 Parts verarbeitet
''' 

#load libraries
import argparse
from dotenv import load_dotenv
from pathlib import Path
from setfit import Trainer
import pandas as pd
from datasets import Dataset
from utils.gpu_utils import get_device_config, get_training_args, load_model, logger #von Ivo geschrieben, um GPU/CPU zu erkennen und trainings_args wie z.B. batch_size anzupassen

## Speicherort von fine-tuned Model lokal und in Hugging Face Hub

DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell" #lokale Bezeichnung der fine-tuned Modelle
DEFAULT_REPO_BASE = "ivozilkenat/setfit-modernbert-studien" #Bezeichnung der fine-tuned Modelle auf Hugging Face Hub


def process_part(config, i):
    """Embedding-Extraktion für einen einzelnen Part (1-4)."""
    logger.info(f"--- STARTE EMBEDDING-EXTRAKTION FÜR PART {i} ---")

    # --------------------------
    # 1. Vorbereitung: Fine-tuned Sentence-Transformer laden und Texte laden
    # --------------------------

    # 1.1  Download from the 🤗 Hub
    # Check: Existiert der lokale Ordner?
    local_folder = f"{DEFAULT_LOCAL_PATH}_{i}" #lokaler Name des fine-tuned Modells für Part i
    if Path(local_folder).exists():
        model_source = local_folder
        logger.info(f"Nutze lokales Modell aus Ordner: {local_folder}")
    else:
        repo_id = f"{DEFAULT_REPO_BASE}{i}" #Name des fine-tuned Modells auf Hugging Face Hub
        model_source = repo_id
        logger.info(f"Lokal nicht gefunden. Lade von Hugging Face: {repo_id}")

    model = load_model(config, model_name=model_source)

    # 1.2 Vorbereitung: JSONL einlesen und Spalten umbenennen
    df = pd.read_json(f"data/studytextPart{i}.jsonl", lines=True)  # erstellt ein Pandas DataFrame

    # 1.3 Spalten umbenennen
    df = df.rename(columns={  # Spalten umbenennen
        "text": "text",
        "replicationSuccessSigDir": "label",
        "numberOriginal": "numberOriginal",
        "setfitSplit": "split"
    })

    # -------------------------- Modelevaluation
    ## Modelle nur an dem Testset evaluieren, an dem sie nicht trainiert wurden
    train_df = df[df['split'] == "train"]
    test_df = df[df['split'] == "test"]

    # in Hugging Face Datasets umwandeln
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,  # Kein Training, daher kein Trainingsdatensatz
        eval_dataset=test_dataset,  # Evaluationsdatensatz ist der test_dataset
        metric="accuracy",  # Evaulation an der Accuracy (Trefferquote) messen
    )

    eval_results = trainer.evaluate()
    logger.info(f"\nFinale Evaluationsergebnisse: {eval_results}")
    logger.info(f"Ergebnis Part {i}: {eval_results}")



def main():
    """Haupt-Funktion für Embedding-Extraktion."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluation der fine-tuned Modelle auf dem Testset (hold-out)")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Spezifischer Part (1-4). Ohne Angabe werden alle verarbeitet.")
    args = parser.parse_args()

    # 0.1 erkennt ob GPU oder CPU vorhanden ist und passt batch_size entsprechend an
    config = get_device_config()


    if args.part:
        process_part(config, args.part)
    else:
        for i in range(1, 5):
            process_part(config, i)


if __name__ == "__main__":
    main()
