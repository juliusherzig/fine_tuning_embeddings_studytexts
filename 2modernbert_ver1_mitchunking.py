# Load Packages
import pandas as pd
from datasets import Dataset, load_dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from transformers import AutoTokenizer #ModernBert
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
import random


print("ModernBERT_load erfolgreich!")

# 0️⃣ Tokenizer für Chunking vorbereiten
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
MAX_TOKENS = 8100 #ModernBERT maximale Tokenlänge ist 8192, wir nehmen 8100 um Puffer zu haben für special tokens wie Token für Satzanfang etc.
OVERLAP_RATIO = 0.2

# 2️⃣ SetFit-Modell laden 
model = SetFitModel.from_pretrained(
    "nomic-ai/modernbert-embed-base", trust_remote_code=True,
)


# JSONL einlesen und Spalten umbenennen
dataset_df = pd.read_json("data/text_part1.jsonl", lines=True) #erstellt ein Pandas DataFrame

dataset_df = dataset_df.rename(columns={ #Spalten umbenennen
    "studytext_part1": "text",
    "Replication_Success_Sig_Dir": "label"
})

print(dataset_df.head())


# Dataset mit Huggingface kompatibel machen 
dataset = Dataset.from_pandas(dataset_df) #Pandas DataFrame in Huggingface Dataset umwandeln

# Spalte 'label' in Typ ClassLabel umwandeln
unique_labels = sorted(dataset_df["label"].unique())
class_label = ClassLabel(num_classes=len(unique_labels), names=[str(x) for x in unique_labels])
dataset = dataset.cast_column("label", class_label)

# Aufteilen in Trainings- und Testset und Trainingsset in Trainings- und Validierungsset FÜR SETFIT
    #Erstellung des Splits für Test, Training und Validierung
dataset = dataset.train_test_split(test_size=0.2, 
    stratify_by_column="label", seed=42)
train_val_dataset = dataset["train"].train_test_split(test_size=0.2, 
    stratify_by_column="label", seed=42)
    # Erstellung neuer Datasets anhand der Splits
train_dataset = sample_dataset(train_val_dataset["train"], label_column="label", seed=42, num_samples=32)
val_dataset = train_val_dataset["test"]
test_dataset = dataset["test"]


## Wie viele Texte überschreiten das maximale Tokenlimit?
def analyze_token_lengths(hf_dataset):
    print("Analysiere Token-Längen (das kann einen Moment dauern)...")
    
    # Längen berechnen (ohne Truncation!)
    lengths = []
    for entry in hf_dataset:
        tokens = tokenizer.encode(entry["text"], add_special_tokens=True)
        lengths.append(len(tokens))
    
    # In Pandas Series umwandeln für einfache Statistik
    s = pd.Series(lengths)
    
    num_too_long = (s > MAX_TOKENS).sum()
    pct_too_long = (num_too_long / len(s)) * 100
    max_found = s.max()
    avg_len = s.mean()
    print(f"Ergebnis der Analyse:")
    print(f"Gesamtanzahl Texte:  {len(s)}")
    print(f"Durchschnittslänge:  {avg_len:.1f} Tokens")
    print(f"Längster Text:       {max_found} Tokens")
    print(f"Texte > {MAX_TOKENS}:      {num_too_long} ({pct_too_long:.2f}%)")
    
analyze_token_lengths(train_dataset)
analyze_token_lengths(val_dataset)
analyze_token_lengths(test_dataset)

# --- Funktion: Chunking + Mean-Pooling für lange Texte --- (falls ein Textsegment maximale Tokenlänge von ModernBERT überschreitet)
def get_embedding_for_training(text, model, tokenizer, max_tokens=MAX_TOKENS, overlap_ratio=OVERLAP_RATIO): #Text wird in tokens zerlegt von oben def. tokenizer 
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0] 
    if tokens.size(0) <= max_tokens: #Wenn die Tokenanzahl kleiner gleich der maximalen Tokenanzahl ist, wird direkt mit oben def. model embedding erstellt
        emb = model.encode([text])[0]
        return emb
        
    else:
        # falls Text zu lang → chunken + mean pooling
        step = int(max_tokens * (1 - overlap_ratio)) #die einzelnen Chunks sollen so groß sein wie 80% der maximalen Tokenanzahl => sorgt für Overlap
        chunks = [] #leere Liste für die Chunks
        for start in range(0, tokens.size(0), step): #Schleife von 0 bis zur Gesamtlänge der Tokens im jew. Text in Schritten von 'step'
            end = min(start + max_tokens, tokens.size(0)) 
            chunk_tokens = tokens[start:end] #hier werden die Gesamttokens in die Chunks aufgeteilt
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True) #die Chunks werden wieder in Text umgewandelt, um sie dem Modell zu übergeben
            chunks.append(chunk_text) #Chunk zur Liste an Chunks hinzufügen
            if end == tokens.size(0):
                break
        
        # Generiert eine Liste von (1, 768) Tensoren
        embeddings = [model.encode([c]) for c in chunks] #Modell erstellt Embeddings für jeden Chunk
        # Stapeln und Mittelwert bilden
        return np.mean(np.vstack(embeddings), axis=0) # [768,] #Mean-Pooling über alle Chunk-Embeddings



############   Ausführung der oben def Funktion get_embedding_for_training ############
#Sinn: so bekommen wir mean-pooled embeddings statt dass SetFit Texte, die max-Tokens überschreiten per default truncatet
train_embeddings = []
train_labels = []

# 'model' und 'tokenizer' sind nun definiert
for row in train_dataset: 
    emb = get_embedding_for_training(row["text"], model, tokenizer)
    train_embeddings.append(emb)
    train_labels.append(row["label"])



# In TensorDataset umwandeln (für SetFit-Trainer kompatibel)
train_embeddings = np.array(train_embeddings)
train_dataset_prepared = torch.utils.data.TensorDataset(
    torch.tensor(train_embeddings, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long)
)
print(f"Prepared training dataset size: {len(train_dataset_prepared)}")

############ Fine-Tuning mit SetFit ############
# Trainingsargumente
args = TrainingArguments(
    num_epochs=1,
    batch_size=8,
    sampling_strategy="undersampling"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset_prepared, # ✅ Korrigiert: Nutzt die pre-computed Embeddings
    eval_dataset=test_dataset,
    metric="accuracy"
)

# 5️⃣ Trainieren
trainer.train()

# 6️⃣ Evaluieren
# Anmerkung: Für die korrekte Evaluation müsste 
# Da der SetFit Trainer dies normalerweise automatisch macht, wird hier der Originalcode beibehalten.
eval_results = trainer.evaluate()
print("\nFinal evaluation results:", eval_results)