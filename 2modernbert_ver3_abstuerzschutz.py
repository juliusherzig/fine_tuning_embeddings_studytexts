########### ÄNDERUNGEN IN DIESEM SKRIIPT ############
# Das Skript 2modernbert_ver2_ohnechunking.py ist beim Training abgestürzt wegen RAM-Überlastung
#Daher
    # kleinere batch_size (von 8 auf 2) => weniger Textpaare gleichzeitig im RAM
    # nicht genutzte DataFrames löschen, um RAM zu sparen
    # keine Modellevaluation zwischendurch im Training (eval_strategy="no"), da das auch viel Rechenzeit und RAM kostet



# Package-Imports müssen vor main-Schleife stehen
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments
import torch

# Auto-detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("library-load erfolgreich!")


#def main():
# SetFit-Modell laden 
model = SetFitModel.from_pretrained(
    "nomic-ai/modernbert-embed-base", 
    trust_remote_code=True,
    device=device,
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

del df, train_df, test_df  #Speicherplatz freigeben

#Dieser Befehl ist in in Turorials nur dafür da, aus großen künstlich kleine Datensätze zu machen, um die Leistung von SetFit an kleinen Datensätzen zu demonstrieren. 
# Das brauche ich hier nicht.   
# train_dataset = sample_dataset(train_dataset, label_column="label", num_samples=31) #erstellt ein neues Dataset mit 31 Samples pro Klasse

############ Fine-Tuning mit SetFit ############
# Trainingsargumente
args = TrainingArguments(
    num_epochs=1,
    batch_size=16,  #Bestimmt die Anzahl der Textpaare, die gleichzeitig im RAM gehalten werden. Verringerung entlastet RAM.
    #max_steps=10,  #begrenzt Trainingsdurchläufe (sonst eben Gesamtzahl an Paaren entsprechend der sampling_strategy/ Anzahl Paar pro Durchlauf). Pro Durchlauf wird Anzahl an Textpaaren = batch_size durchlaufen und daran die Modellgewichte angepasst.

    sampling_strategy="undersampling", #Sorgt für gleiche Anzahl an pos. und neg. Paaren im Training, indem von der größeren Paare entfernt werden
    eval_strategy="no", #wenn nicht ="no", dann automatisch self.do_eval=True. Dadurch evaluiert sich Modell nach jedem Step und das kostet extrem Rechenzeit        
    save_strategy="no", #wenn nicht ="no", dann automatisch self.do_eval=True. Dadurch evaluiert sich Modell nach jedem Step und das kostet extrem Rechenzeit. Quelle: https://huggingface.co/docs/transformers/main_classes/trainer
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,        
    eval_dataset=test_dataset, #Evaluationsdatensatz ist der test_dataset
    metric="accuracy", #Evaulation an der Accuracy (Trefferquote) messen        
)

# Trainieren
trainer.train()

# 6.  Evaluieren
eval_results = trainer.evaluate()
print("\nFinal evaluation results:", eval_results)

##Modell speichern
model.save_pretrained("mein_modernbert_studien_modell")

#if __name__ == "__main__": #schützt Computer davor, das Skript ungewollt auf mehreren Kernen auszuführen, was zu Abstürzen führen kann.
#    main()