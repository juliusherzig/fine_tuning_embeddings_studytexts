#!/usr/bin/env python3
"""HuggingFace Hub Modell-Verwaltung für SetFit.

Ermöglicht das Hochladen und Herunterladen von trainierten SetFit Modellen
zu/von HuggingFace Hub.

Um die Textlänge zu begrenzen, wurden die Studientexte zuvor in 4 gleich große Segmente (Parts) aufgeteilt. Es wird für jeden Part ein eigenes Modell trainiert und separat hochgeladen/ heruntergeladen. 

Verwendung:
    uv run hf_model_j.py push              # Lokales Modell hochladen
    uv run hf_model_j.py load              # Modell von Hub herunterladen
    uv run hf_model_j.py push --part 1     # Nur Part 1 hochladen
    uv run hf_model_j.py load --part 3     # Nur Part 3 herunterladen"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from setfit import SetFitModel

# Konstanten - hier anpassen
DEFAULT_REPO_BASE = "juliusherzig/setfit-modernbert-studien"
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell"

# Spezifizieren, ob Modell für Textpart 1,2,3 oder 4 oder alle gepullt/ gepusht werden soll
def process_part(command, part_number):
    """Verarbeitet einen einzelnen Part (1-4) für push oder load."""
    repo_id = f"{DEFAULT_REPO_BASE}{part_number}"
    local_path = f"{DEFAULT_LOCAL_PATH}_{part_number}"
    
    if command == "push":
        if not Path(local_path).exists():
            print(f"⚠️  Part {part_number} übersprungen: Ordner {local_path} nicht gefunden.")
            return
        
        print(f"Pushe Modell für Part {part_number} zu {repo_id}.")
        push_model(repo_id, local_path)


    elif command == "load":    
        print(f"Lade Modell für Part {part_number} von {repo_id}.")
        download_model(repo_id, local_path)
    
    else:
        print(f"Unbekannter Befehl: {command}")
        sys.exit(1)

#Funktion zum Hochladen von Modellen zu HuggingFace Hub
def push_model(repo_id: str, local_path: str) -> None:
    """Lädt ein lokales Modell zu HuggingFace Hub hoch.
    
    Args:
        repo_id: HuggingFace Repository ID (z.B. "user/model-name")
        local_path: Pfad zum lokalen Modell-Verzeichnis
    """
    # Token laden und authentifizieren
    token = os.getenv("HF_TOKEN_WRITE") #Token ist in .env Datei hinterlegt, damit er nicht im Code steht
    if not token:
        print("Fehler: HF_TOKEN_WRITE nicht in .env gefunden")
        print("Erstelle eine .env Datei mit deinem HuggingFace Token (write access)")
        sys.exit(1)
    
    login(token=token)
    
    # Prüfen ob lokales Modell existiert
    if not Path(local_path).exists():
        print(f"Fehler: Lokales Modell nicht gefunden: {local_path}")
        print("Trainiere zuerst ein Modell mit dem Training-Skript")
        sys.exit(1)
    
    print(f"Lade Modell von {local_path}...")
    model = SetFitModel.from_pretrained(local_path)
    
    print(f"Pushe Modell zu {repo_id}...")
    model.push_to_hub(repo_id)
    
    print(f"Erfolgreich hochgeladen: https://huggingface.co/{repo_id}")


def download_model (repo_id: str, local_path: str) -> None:
    """Lädt ein Modell von HuggingFace Hub herunter.
    
    Args:
        repo_id: HuggingFace Repository ID (z.B. "user/model-name")
        local_path: Pfad zum Speichern des Modells
    """
    # Token laden und authentifizieren (für private Repos)
    token = os.getenv("HF_TOKEN_READ") #Token ist in .env Datei hinterlegt, damit er nicht im Code steht
    if token:
        login(token=token)
        print("Authentifiziert mit HF_TOKEN_READ")
    else:
        print("Kein HF_TOKEN_READ gefunden - versuche ohne Authentifizierung")
        print("(Funktioniert nur für öffentliche Modelle)")
    
    print(f"Lade Modell von {repo_id}...")
    model = SetFitModel.from_pretrained(repo_id, trust_remote_code=True)
    
    print(f"Speichere Modell nach {local_path}...")
    model.save_pretrained(local_path)
    
    print(f"Erfolgreich heruntergeladen: {local_path}")


def main():
    """Hauptfunktion mit Argument-Parsing."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub Modell-Verwaltung für SetFit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

    
Verwendung:
  uv run hf_model_j.py push           # Lädt alle 4 lokalen Modelle hoch
  uv run hf_model_j.py load           # Lädt alle 4 Modelle vom Hub herunter
  uv run hf_model_j.py push --part 1  # Lädt nur das Modell von Part 1 hoch
  uv run hf_model_j.py load --part 3  # Lädt nur das Modell von Part 3 herunter
"""
    )
    
    parser.add_argument("command", choices=["push", "load"], help="Aktion: Modell hochladen oder herunterladen")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], help="Spezifischer Part (1-4). Ohne Angabe werden alle verarbeitet.")

    args = parser.parse_args()

    if args.part:
        process_part(args.command, args.part)
    else:
        print("Verarbeite alle Modelle (1-4)...")
        for i in range(1, 5):
            process_part(args.command, i)


if __name__ == "__main__":
    main()
