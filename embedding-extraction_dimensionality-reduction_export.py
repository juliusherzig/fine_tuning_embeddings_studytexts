''''
Anwendung:
uv run python embedding-extraction_dimensionality-reduction_export.py --part 1 # nur Part 1 wird verarbeitet
uv run python embedding-extraction_dimensionality-reduction_export.py # ohne Angabe von --part werden alle 4 Parts verarbeitet
''' 

#load libraries
import argparse
import argparse
from dotenv import load_dotenv
from pathlib import Path
import sklearn
import pandas as pd
import os
from utils.gpu_utils import get_device_config, load_model, logger #von Ivo geschrieben, um GPU/CPU zu erkennen und trainings_args wie z.B. batch_size anzupassen


## Speicherort von fine-tuned Model lokal und in Hugging Face Hub als Konstante
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell" #lokale Bezeichnung der fine-tuned Modelle
DEFAULT_REPO_BASE = "ivozilkenat/setfit-modernbert-studien" #Bezeichnung der fine-tuned Modelle auf Hugging Face Hub

## 0.3 Speicherort von fine-tuned Model lokal und in Hugging Face Hub

DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell" #lokale Bezeichnung der fine-tuned Modelle
DEFAULT_REPO_BASE = "ivozilkenat/setfit-modernbert-studien" #Bezeichnung der fine-tuned Modelle auf Hugging Face Hub


def process_part(config, i, output_dir):
    """Embedding-Extraktion für einen einzelnen Part (1-4)."""
def process_part(config, i, output_dir):
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
        logger.info(f"Nutze lokales Modell aus Ordner: {local_folder}")
    else:
        repo_id = f"{DEFAULT_REPO_BASE}{i}" #Name des fine-tuned Modells auf Hugging Face Hub
        model_source = repo_id
        logger.info(f"Lokal nicht gefunden. Lade von Hugging Face: {repo_id}")

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

    # --------------------------
    # 2. Fine-Tuned Embeddings für alle Texte erzeugen
    # --------------------------
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=config.batch_size)  # shape: (Anzahl_Texte, Embedding-Dimension)
    logger.info(f"Original Embedding-Shape: {embeddings.shape}")
    logger.info(f"Original Embedding-Shape: {embeddings.shape}")

    # --------------------------
    # 3. Mit PCA auf weniger Dimensionen reduzieren
    # --------------------------
    pca = sklearn.decomposition.PCA(n_components=32, random_state=42) #n_components can be maximum n_samples
    embeddings_reduced = pca.fit_transform(embeddings)
    logger.info(f"Reduced Embedding-Shape: {embeddings_reduced.shape}")
    logger.info(f"Reduced Embedding-Shape: {embeddings_reduced.shape}")

    # --------------------------
    # 4️. Exportieren
    # --------------------------
    # 4.1 Embeddings in einen DataFrame umwandeln
    emb_df = pd.DataFrame(embeddings_reduced)

    # 4.2 Spalten benennen (z.B. dim_0, dim_1, ...)
    emb_df.columns = [f"dim_{e}" for e in range(emb_df.shape[1])]

    # 4.3 Die ID für merge mit restlichen Variablen in R
    emb_df['numberOriginal'] = df['numberOriginal'].values

    # 4.4 Speicherort für Export
    file_path = os.path.join(output_dir, f"embeddingsPart{i}.parquet")

    #4.5 Export als .parquet-Datei
    emb_df.to_parquet(file_path)
    logger.info(f"Embeddings für Part {i} wurden erfolgreich exportiert nach: {file_path}")


def main():
    """Haupt-Funktion für Embedding-Extraktion."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Embedding-Extraktion und PCA-Reduktion")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Spezifischer Part (1-4). Ohne Angabe werden alle verarbeitet.")
    args = parser.parse_args()

    # 0.1 erkennt ob GPU oder CPU vorhanden ist und passt batch_size entsprechend an
    config = get_device_config()

    # 0.2 erkennt output-Ordner für die zu exportierenden Embeddings und erstellt ihn ggf. neu
    output_dir = "output_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    if args.part:
        process_part(config, args.part, output_dir)
    else:
        for i in range(1, 5):
            process_part(config, i, output_dir)


if __name__ == "__main__":
    main()



def main():
    """Haupt-Funktion für Embedding-Extraktion."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Embedding-Extraktion und PCA-Reduktion")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Spezifischer Part (1-4). Ohne Angabe werden alle verarbeitet.")
    args = parser.parse_args()

    # 0.1 erkennt ob GPU oder CPU vorhanden ist und passt batch_size entsprechend an
    config = get_device_config()

    # 0.2 erkennt output-Ordner für die zu exportierenden Embeddings und erstellt ihn ggf. neu
    output_dir = "output_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    if args.part:
        process_part(config, args.part, output_dir)
    else:
        for i in range(1, 5):
            process_part(config, i, output_dir)


if __name__ == "__main__":
    main()
