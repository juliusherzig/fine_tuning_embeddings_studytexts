########### ÄNDERUNGEN IN DIESEM SKRIIPT ############
# Das Skript 2modernbert_ver2_ohnechunking.py ist beim Training abgestürzt wegen RAM-Überlastung
#Daher
    # kleinere batch_size (von 8 auf 2) => weniger Textpaare gleichzeitig im RAM
    # nicht genutzte DataFrames löschen, um RAM zu sparen
    # keine Modellevaluation zwischendurch im Training (eval_strategy="no"), da das auch viel Rechenzeit und RAM kostet

# GPU Optimierungen (neu hinzugefügt):
# - Verwendet shared utils für geräteunabhängiges Model-Loading
# - Mixed precision Training (use_amp) auf GPU für schnelleres Training
# - Automatischer CPU Fallback mit reference_compile=False (deaktiviert Triton)

# Package-Imports müssen vor main-Schleife stehen
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer

from utils.gpu_utils import get_device_config, load_model, get_training_args, logger


def main():
    """Haupt-Trainingsfunktion."""
    # 1. Gerät und Modell mit GPU/CPU Kompatibilität einrichten
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

    ## Modell speichern
    model.save_pretrained("mein_modernbert_studien_modell")


if __name__ == "__main__":  # schützt Computer davor, das Skript ungewollt auf mehreren Kernen auszuführen, was zu Abstürzen führen kann.
    main()
