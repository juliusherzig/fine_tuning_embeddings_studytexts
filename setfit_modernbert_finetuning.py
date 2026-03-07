# GPU Optimierungen (neu hinzugefügt):
# - Verwendet shared utils für geräteunabhängiges Model-Loading
# - Mixed precision Training (use_amp) auf GPU für schnelleres Training
# - Automatischer CPU Fallback mit reference_compile=False (deaktiviert Triton)

# Package-Imports müssen vor main-Schleife stehen
import argparse
import os
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer
from dotenv import load_dotenv
from huggingface_hub import login

from utils.gpu_utils import get_device_config, load_model, get_training_args, logger #von Ivo geschrieben, um GPU/CPU zu erkennen und trainings_args wie z.B. batch_size anzupassen

load_dotenv()
HF_REPO_PREFIX = "ivozilkenat/setfit-modernbert-studien-modell"


def train_part(config, i):
    """Trainiert ein einzelnes Part-Modell (1-4)."""
    logger.info(f"--- STARTE TRAINING FÜR PART {i} ---")

    model = load_model(config)
    logger.info("Model-load erfolgreich!")

    # JSONL einlesen und Spalten umbenennen
    df = pd.read_json(f"data/studytextPart{i}.jsonl", lines=True)  # erstellt ein Pandas DataFrame

    df = df.rename(columns={  # Spalten umbenennen
        "text": "text",
        "replicationSuccessSigDir": "label",
        "numberOriginal": "numberOriginal",
        "setfitSplit": "split"
    })

    logger.info(f"Daten geladen:\n{df.head()}")

    ################################ Datasplit ################################
    # Die Variable Split nutzen, die wurde zuvor in R erstellt, damit die Aufteilung in allen 4 Textsegmenten gleich ist.

    train_df = df[df['split'] == "train"]
    test_df = df[df['split'] == "test"]

    logger.info(f"Verteilung der Klassen im Trainingsset:\n{train_df['label'].value_counts()}")
    logger.info(f"Verteilung der Klassen im Testset:\n{test_df['label'].value_counts()}")

    # in Hugging Face Datasets umwandeln
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Spalte 'label' in Typ ClassLabel umwandeln
    unique_labels = sorted(df["label"].unique())
    class_label = ClassLabel(num_classes=len(unique_labels), names=[str(x) for x in unique_labels])
    train_dataset = train_dataset.cast_column("label", class_label)
    test_dataset = test_dataset.cast_column("label", class_label)

    del df, train_df, test_df  # Speicherplatz freigeben

    # Dieser Befehl ist in in Turorials nur dafür da, aus großen künstlich kleine Datensätze zu machen, um die Leistung von SetFit an kleinen Datensätzen zu demonstrieren.
    # Das brauche ich hier nicht.
    # train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=31) #erstellt ein neues Dataset mit 31 Samples pro Klasse

    ############ Fine-Tuning mit SetFit ############
    # Trainingsargumente
    # batch_size wird durch device config bestimmt (16 für GPU, 8 für CPU)
    # use_amp=True für mixed precision auf GPU
    args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Evaluationsdatensatz ist der test_dataset
        metric="accuracy",  # Evaulation an der Accuracy (Trefferquote) messen
    )

    # Trainieren
    trainer.train()

    # 6.  Evaluieren
    eval_results = trainer.evaluate()
    logger.info(f"\nFinale Evaluationsergebnisse: {eval_results}")
    logger.info(f"Ergebnis Part {i}: {eval_results}")

    ## Modell speichern
    model.save_pretrained(f"mein_modernbert_studien_modell_{i}")


def main():
    """Haupt-Trainingsfunktion."""
    parser = argparse.ArgumentParser(description="SetFit ModernBERT Finetuning")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Spezifischer Part (1-4). Ohne Angabe werden alle trainiert.")
    args = parser.parse_args()

    # HuggingFace login for model push
    hf_token = os.getenv("HF_TOKEN_WRITE")
    if hf_token:
        login(token=hf_token)
        logger.info("HuggingFace login erfolgreich")
    else:
        logger.warning("HF_TOKEN_WRITE nicht gesetzt - Modelle werden nur lokal gespeichert")

    # 1. Gerät und Modell mit GPU/CPU Kompatibilität einrichten
    config = get_device_config()

    if args.part:
        train_part(config, args.part)
    else:
        for i in range(1, 5):
            train_part(config, i)


if __name__ == "__main__":
    main()