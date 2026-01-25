# Load Packages
from py_compile import main
import pandas as pd
from datasets import Dataset, load_dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from transformers import AutoTokenizer #ModernBert

print("library-load erfolgreich!")

# SetFit-Modell laden 
model = SetFitModel.from_pretrained(
    "nomic-ai/modernbert-embed-base", trust_remote_code=True,
)

print("Model-load erfolgreich!")

# JSONL einlesen und Spalten umbenennen
df = pd.read_json("data/studytext_part1.jsonl", lines=True) #erstellt ein Pandas DataFrame

df = df.rename(columns={ #Spalten umbenennen
    "text": "text",
    "Replication_Success_Sig_Dir": "label",
    "Number_Original": "Number_Original",
    "setfit_split": "split"
})

print(df.head())

################################ Datasplit ################################
#Die Variable Split nutzen,die urde zuvor in R erstellt, damit die Aufteilung in allen 4 Textsegmenten gleich ist.

train_df = df[df['split'] == "train"] 
test_df= df[df['split'] == "test"]

print(f"Verteilung der Klassen im Trainingsset:\n{train_df['label'].value_counts()}")
print(f"Verteilung der Klassen im Testset:\n{test_df['label'].value_counts()}")

# in Hugging Face Datasets umwandeln
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Spalte 'label' in Typ ClassLabel umwandeln
unique_labels = sorted(df["label"].unique())
class_label = ClassLabel(num_classes=len(unique_labels), names=[str(x) for x in unique_labels])
train_dataset = train_dataset.cast_column("label", class_label)
test_dataset  = test_dataset.cast_column("label", class_label)

#Dieser Befehl ist in in Turorials nur dafür da, aus großen künstlich kleine Datensätze zu machen, um die Leistung von SetFit an kleinen Datensätzen zu demonstrieren. 
# Das brauche ich hier nicht.   
# train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=31) #erstellt ein neues Dataset mit 31 Samples pro Klasse



######### Wie viele Texte überschreiten das maximale Tokenlimit? ########
MAX_TOKENS = 8100 #ModernBERTs Tokenlimit ist 8192. Wir nehmen 8100 für Puffer für special tokens, z.B. Token für Satzanfang etc.
# Tokenizer wählen, um Tokens zu zählen
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")

def analyze_token_lengths(hf_dataset):
       
    # Längen berechnen (ohne Truncation!)
    lengths = []
    for entry in hf_dataset:
        tokens = tokenizer.encode(entry["text"], add_special_tokens=True)
        lengths.append(len(tokens))

    # In Pandas Series umwandeln zur einfacheren Analyse
    s = pd.Series(lengths)
    
    num_too_long = (s > MAX_TOKENS).sum()
    pct_too_long = (num_too_long / len(s)) * 100
    max_found = s.max()
    avg_len = s.mean()
    print(f"Gesamtanzahl Texte:  {len(s)}") #durch das f in print(f"....") können Variablen in {} eingebunden werden
    print(f"Durchschnittslänge:  {avg_len:.1f} Tokens")
    print(f"Längster Text:       {max_found} Tokens")
    print(f"Texte > {MAX_TOKENS}:      {num_too_long} ({pct_too_long:.2f}%)")

    if num_too_long == 0:
        print("Keines der Textsegmente überschreitet das Tokenlimit von ModernBERT. Kein Chunking notwendig.")
    
    if num_too_long > 0:
        print("Mind. n=1 Textsegment überschreitet das Tokenlimit von ModernBERT. Chunking notwendig.")
    
analyze_token_lengths(train_dataset)
analyze_token_lengths(test_dataset)

#import sys
#sys.exit("Stoppbefehl")

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
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    metric="accuracy"
)

# Trainieren
trainer.train()

# 6.  Evaluieren
# 
eval_results = trainer.evaluate()
print("\nFinal evaluation results:", eval_results)

## Speichern der Embeddings für spätere Verwendung fehlt noch
