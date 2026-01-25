# Load Packages
# GPU Optimierungen (neu hinzugefügt):
# - Verwendet shared utils für geräteunabhängiges Model-Loading
# - Mixed precision Training (use_amp) auf GPU für schnelleres Training
# - Automatischer CPU Fallback mit reference_compile=False (deaktiviert Triton)

import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer
from transformers import AutoTokenizer  # ModernBert

from utils.gpu_utils import (
    get_device_config,
    load_model,
    get_training_args,
    logger,
    MODEL_NAME,
    MAX_TOKENS,
)


def analyze_token_lengths(hf_dataset, tokenizer, max_tokens: int = MAX_TOKENS):
    """Analysiert Token-Längen im Dataset."""
    # Längen berechnen (ohne Truncation!)
    lengths = []
    for entry in hf_dataset:
        tokens = tokenizer.encode(entry["text"], add_special_tokens=True)
        lengths.append(len(tokens))

    # In Pandas Series umwandeln zur einfacheren Analyse
    s = pd.Series(lengths)

    num_too_long = (s > max_tokens).sum()
    pct_too_long = (num_too_long / len(s)) * 100
    max_found = s.max()
    avg_len = s.mean()
    logger.info(f"Gesamtanzahl Texte:  {len(s)}")  # durch das f in print(f"....") können Variablen in {} eingebunden werden
    logger.info(f"Durchschnittslänge:  {avg_len:.1f} Tokens")
    logger.info(f"Längster Text:       {max_found} Tokens")
    logger.info(f"Texte > {max_tokens}:      {num_too_long} ({pct_too_long:.2f}%)")

    if num_too_long == 0:
        logger.info("Keines der Textsegmente überschreitet das Tokenlimit von ModernBERT. Kein Chunking notwendig.")

    if num_too_long > 0:
        logger.info("Mind. n=1 Textsegment überschreitet das Tokenlimit von ModernBERT. Chunking notwendig.")


def main():
    """Haupt-Trainingsfunktion."""
    logger.info("library-load erfolgreich!")

    # Gerät und Modell mit GPU/CPU Kompatibilität einrichten
    config = get_device_config()
    model = load_model(config)
    logger.info("Model-load erfolgreich!")

    # JSONL einlesen und Spalten umbenennen
    df = pd.read_json("data/studytext_part1.jsonl", lines=True)  # erstellt ein Pandas DataFrame

    df = df.rename(columns={  # Spalten umbenennen
        "text": "text",
        "Replication_Success_Sig_Dir": "label",
        "Number_Original": "Number_Original",
        "setfit_split": "split"
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

    # Dieser Befehl ist in in Turorials nur dafür da, aus großen künstlich kleine Datensätze zu machen, um die Leistung von SetFit an kleinen Datensätzen zu demonstrieren.
    # Das brauche ich hier nicht.
    # train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=31) #erstellt ein neues Dataset mit 31 Samples pro Klasse

    ######### Wie viele Texte überschreiten das maximale Tokenlimit? ########
    # ModernBERTs Tokenlimit ist 8192. Wir nehmen 8100 für Puffer für special tokens, z.B. Token für Satzanfang etc.
    # Tokenizer wählen, um Tokens zu zählen
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    analyze_token_lengths(train_dataset, tokenizer)
    analyze_token_lengths(test_dataset, tokenizer)

    # import sys
    # sys.exit("Stoppbefehl")

    ############ Fine-Tuning mit SetFit ############
    # Trainingsargumente
    args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="accuracy"
    )

    # Trainieren
    trainer.train()

    # 6.  Evaluieren
    eval_results = trainer.evaluate()
    logger.info(f"\nFinale Evaluationsergebnisse: {eval_results}")

    ## Speichern der Embeddings für spätere Verwendung fehlt noch


if __name__ == "__main__":
    main()
