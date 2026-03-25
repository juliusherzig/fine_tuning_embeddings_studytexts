# GPU Optimierungen:
# - Verwendet shared utils für geräteunabhängiges Model-Loading
# - Mixed precision Training (use_amp) auf GPU für schnelleres Training
# - Automatischer CPU Fallback mit reference_compile=False (deaktiviert Triton)

# Package-Imports müssen vor main-Schleife stehen
import argparse
import os
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer, TrainingArguments
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import EarlyStoppingCallback 

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
    val_df = df[df['split'] == "val"] #neu: Validierungsset aus Trainingsset splitten für stepwise Evaluation und Early Stopping

    logger.info(f"Verteilung der Klassen im Trainingsset:\n{train_df['label'].value_counts()}")
    logger.info(f"Verteilung der Klassen im Testset:\n{test_df['label'].value_counts()}")
    logger.info(f"Verteilung der Klassen im Validierungsset:\n{val_df['label'].value_counts()}")

    # in Hugging Face Datasets umwandeln
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = Dataset.from_pandas(train_df) 
    val_dataset = Dataset.from_pandas(val_df)   

    # Klasse von label in ClassLabel umwandeln
    class_label = ClassLabel(num_classes=2, names=["0", "1"])
    train_dataset = train_dataset.cast_column("label", class_label)
    val_dataset = val_dataset.cast_column("label", class_label)
    test_dataset = test_dataset.cast_column("label", class_label)
    
    del df, train_df, test_df  # Speicherplatz freigeben

    ############ Fine-Tuning mit SetFit ############
    # Trainingsargumente
    # batch_size wird durch device config bestimmt (16 für GPU, 8 für CPU)
    # use_amp=True für mixed precision auf GPU
    args = TrainingArguments(
        num_epochs = 1,
        batch_size = config.batch_size, #geräteangepasste (GPU/ CPU)
        use_amp = config.use_amp, #geräteangepasste (GPU/ CPU)
        sampling_strategy = "undersampling",
        eval_strategy = "steps", #neu: war bislang "no" => kein Validierungsset. 
        #eval_strategy "steps" => Evaluation während des Trainings. Detektion davon, dass nur noch Trainingsloss besser wird, aber Vorhersage an Validierungsset nicht mehr, was Overfitting anzeigt. Dann wird Early Stopping getriggert, um Training zu stoppen.
        save_strategy = "steps", #neu: nötig, um vorheriges Modell laden zu können, wenn Early Stopping getriggert wird, weil weiteres Training keine Verbesserung mehr bringt
        logging_strategy= "steps",
        save_steps = 50,        
        eval_steps = 50,
        logging_steps = 50,
        load_best_model_at_end = True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Validierungsdatensatz! (Split vom Trainingsset)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], #neu: Early Stopping beugt Overfitting vor, indem es das Training stoppt, wenn sich die Leistung auf dem Validierungsset nicht mehr verbessert. Hier wird eine Geduld von 5 angegeben, d.h. das Training wird gestoppt, wenn sich die Leistung sich am Validierungsset nicht mehr verbessert.
        metric="accuracy",  # Evaulation an der Accuracy (Trefferquote) messen
    )

    # Trainieren
    trainer.train()

    # 6.  Evaluieren
    trainer.eval_dataset = test_dataset #Testdatensatz!
    test_results = trainer.evaluate()
    logger.info(f"\nFinale Evaluationsergebnisse: {test_results}")
    logger.info(f"Ergebnis Part {i}: {test_results}")

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