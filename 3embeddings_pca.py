#load libraries
from pyexpat import model
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd



#Texte mit Daten einladen
df = pd.read_json("studytext_part1.jsonl", lines=True) #erstellt ein Pandas DataFrame

df = df.rename(columns={ #Spalten umbenennen
    "text": "text",
    "Replication_Success_Sig_Dir": "label",
    "Number_Original": "Number_Original",
    "setfit_split": "split"
})

# --------------------------
# 7️⃣ Fine-Tuned Embeddings für alle Texte erzeugen
# --------------------------
embeddings = model.embed(df["text"])  # shape: (Anzahl_Texte, Embedding-Dimension)
print("Original Embedding-Shape:", embeddings.shape)

##SIND DIE DIE FINE-TUNED EMBEDDINGS???

# --------------------------
# 8️⃣ PCA auf 128 Dimensionen
# --------------------------
##IST DAS DIE PCA-FUNKTION, DIE DIE SETFIT-AUTOREN EMPFEHLEN UND DIE ICH IN MEINER PRÄREGISTRIERUNG GENANNT HABE?

pca = PCA(n_components=128, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings)
print("Reduced Embedding-Shape:", embeddings_reduced.shape)

# --------------------------
# 9️⃣ Speichern
# --------------------------
np.save("embeddings_168.npy", embeddings_reduced)
print("Embeddings gespeichert in embeddings_168.npy")