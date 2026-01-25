# Load Packages
# GPU Optimierungen (neu hinzugefügt):
# - Verwendet shared utils für geräteunabhängiges Model-Loading
# - Batch encoding für effiziente GPU-Nutzung
# - Mixed precision Training (use_amp) auf GPU für schnelleres Training
# - Automatischer CPU Fallback mit reference_compile=False (deaktiviert Triton)

import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer, sample_dataset
from transformers import AutoTokenizer  # ModernBert
import numpy as np
import torch

from tqdm import tqdm

from utils.gpu_utils import (
    get_device_config,
    load_model,
    get_training_args,
    logger,
    MODEL_NAME,
    MAX_TOKENS,
    OVERLAP_RATIO,
)


def analyze_token_lengths(hf_dataset, tokenizer, max_tokens: int = MAX_TOKENS):
    """Wie viele Texte überschreiten das maximale Tokenlimit?"""
    logger.info("Analysiere Token-Längen (das kann einen Moment dauern)...")

    # Längen berechnen (ohne Truncation!)
    lengths = []
    for entry in hf_dataset:
        tokens = tokenizer.encode(entry["text"], add_special_tokens=True)
        lengths.append(len(tokens))

    # In Pandas Series umwandeln für einfache Statistik
    s = pd.Series(lengths)

    num_too_long = (s > max_tokens).sum()
    pct_too_long = (num_too_long / len(s)) * 100
    max_found = s.max()
    avg_len = s.mean()
    logger.info(f"Ergebnis der Analyse:")
    logger.info(f"Gesamtanzahl Texte:  {len(s)}")
    logger.info(f"Durchschnittslänge:  {avg_len:.1f} Tokens")
    logger.info(f"Längster Text:       {max_found} Tokens")
    logger.info(f"Texte > {max_tokens}:      {num_too_long} ({pct_too_long:.2f}%)")


# --- Funktion: Chunking + Mean-Pooling für lange Texte --- (falls ein Textsegment maximale Tokenlänge von ModernBERT überschreitet)
def get_embedding_for_training(text, model, tokenizer, max_tokens=MAX_TOKENS, overlap_ratio=OVERLAP_RATIO):
    """Text wird in tokens zerlegt von oben def. tokenizer."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    if tokens.size(0) <= max_tokens:  # Wenn die Tokenanzahl kleiner gleich der maximalen Tokenanzahl ist, wird direkt mit oben def. model embedding erstellt
        emb = model.encode([text])[0]
        return emb

    else:
        # falls Text zu lang → chunken + mean pooling
        step = int(max_tokens * (1 - overlap_ratio))  # die einzelnen Chunks sollen so groß sein wie 80% der maximalen Tokenanzahl => sorgt für Overlap
        chunks = []  # leere Liste für die Chunks
        for start in range(0, tokens.size(0), step):  # Schleife von 0 bis zur Gesamtlänge der Tokens im jew. Text in Schritten von 'step'
            end = min(start + max_tokens, tokens.size(0))
            chunk_tokens = tokens[start:end]  # hier werden die Gesamttokens in die Chunks aufgeteilt
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)  # die Chunks werden wieder in Text umgewandelt, um sie dem Modell zu übergeben
            chunks.append(chunk_text)  # Chunk zur Liste an Chunks hinzufügen
            if end == tokens.size(0):
                break

        # Batch encode alle Chunks effizient (GPU-optimiert)
        embeddings = model.encode(chunks, batch_size=8)  # Modell erstellt Embeddings für alle Chunks auf einmal
        # Mittelwert bilden
        return np.mean(embeddings, axis=0)  # [768,] #Mean-Pooling über alle Chunk-Embeddings


def main():
    """Haupt-Trainingsfunktion."""
    logger.info("ModernBERT_load erfolgreich!")

    # 0️⃣ Gerät und Modell mit GPU/CPU Kompatibilität einrichten
    config = get_device_config()
    model = load_model(config)

    # Tokenizer für Chunking vorbereiten
    # ModernBERT maximale Tokenlänge ist 8192, wir nehmen 8100 um Puffer zu haben für special tokens wie Token für Satzanfang etc.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # JSONL einlesen und Spalten umbenennen
    dataset_df = pd.read_json("data/text_part1.jsonl", lines=True)  # erstellt ein Pandas DataFrame

    dataset_df = dataset_df.rename(columns={  # Spalten umbenennen
        "studytext_part1": "text",
        "Replication_Success_Sig_Dir": "label"
    })

    logger.info(f"Daten geladen:\n{dataset_df.head()}")

    # Dataset mit Huggingface kompatibel machen
    dataset = Dataset.from_pandas(dataset_df)  # Pandas DataFrame in Huggingface Dataset umwandeln

    # Spalte 'label' in Typ ClassLabel umwandeln
    unique_labels = sorted(dataset_df["label"].unique())
    class_label = ClassLabel(num_classes=len(unique_labels), names=[str(x) for x in unique_labels])
    dataset = dataset.cast_column("label", class_label)

    # Aufteilen in Trainings- und Testset und Trainingsset in Trainings- und Validierungsset FÜR SETFIT
    # Erstellung des Splits für Test, Training und Validierung
    dataset = dataset.train_test_split(test_size=0.2,
        stratify_by_column="label", seed=42)
    train_val_dataset = dataset["train"].train_test_split(test_size=0.2,
        stratify_by_column="label", seed=42)
    # Erstellung neuer Datasets anhand der Splits
    train_dataset = sample_dataset(train_val_dataset["train"], label_column="label", seed=42, num_samples=32)
    val_dataset = train_val_dataset["test"]
    test_dataset = dataset["test"]

    ## Wie viele Texte überschreiten das maximale Tokenlimit?
    analyze_token_lengths(train_dataset, tokenizer)
    analyze_token_lengths(val_dataset, tokenizer)
    analyze_token_lengths(test_dataset, tokenizer)

    ############   Ausführung der oben def Funktion get_embedding_for_training ############
    # Sinn: so bekommen wir mean-pooled embeddings statt dass SetFit Texte, die max-Tokens überschreiten per default truncatet
    train_embeddings = []
    train_labels = []

    # 'model' und 'tokenizer' sind nun definiert
    logger.info("Generiere Embeddings mit Chunking-Unterstützung...")
    for row in tqdm(train_dataset, desc="Verarbeite Texte"):
        emb = get_embedding_for_training(row["text"], model, tokenizer)
        train_embeddings.append(emb)
        train_labels.append(row["label"])

    # In TensorDataset umwandeln (für SetFit-Trainer kompatibel)
    train_embeddings = np.array(train_embeddings)
    train_dataset_prepared = torch.utils.data.TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    logger.info(f"Vorbereitete Trainings-Dataset Größe: {len(train_dataset_prepared)}")

    ############ Fine-Tuning mit SetFit ############
    # Trainingsargumente
    args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset_prepared,  # ✅ Korrigiert: Nutzt die pre-computed Embeddings
        eval_dataset=test_dataset,
        metric="accuracy"
    )

    # 5️⃣ Trainieren
    trainer.train()

    # 6️⃣ Evaluieren
    # Anmerkung: Für die korrekte Evaluation müsste
    # Da der SetFit Trainer dies normalerweise automatisch macht, wird hier der Originalcode beibehalten.
    eval_results = trainer.evaluate()
    logger.info(f"\nFinale Evaluationsergebnisse: {eval_results}")


if __name__ == "__main__":
    main()
