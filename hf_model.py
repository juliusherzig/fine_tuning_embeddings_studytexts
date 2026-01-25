#!/usr/bin/env python3
"""HuggingFace Hub Modell-Verwaltung für SetFit.

Ermöglicht das Hochladen und Herunterladen von trainierten SetFit Modellen
zu/von HuggingFace Hub.

Verwendung:
    uv run hf_model.py push              # Lokales Modell hochladen
    uv run hf_model.py load              # Modell von Hub herunterladen
    uv run hf_model.py push --repo-id "andere/repo"  # Anderes Repo verwenden
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from setfit import SetFitModel

# Konstanten - hier anpassen
DEFAULT_REPO_ID = "ivozilkenat/setfit-modernbert-studien"
DEFAULT_LOCAL_PATH = "mein_modernbert_studien_modell"


def push_model(repo_id: str, local_path: str) -> None:
    """Lädt ein lokales Modell zu HuggingFace Hub hoch.
    
    Args:
        repo_id: HuggingFace Repository ID (z.B. "user/model-name")
        local_path: Pfad zum lokalen Modell-Verzeichnis
    """
    # Token laden und authentifizieren
    token = os.getenv("HF_TOKEN_WRITE")
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


def load_model(repo_id: str, local_path: str) -> None:
    """Lädt ein Modell von HuggingFace Hub herunter.
    
    Args:
        repo_id: HuggingFace Repository ID (z.B. "user/model-name")
        local_path: Pfad zum Speichern des Modells
    """
    # Token laden und authentifizieren (für private Repos)
    token = os.getenv("HF_TOKEN_READ")
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
Beispiele:
  uv run hf_model.py push                    # Push mit Standard-Repo
  uv run hf_model.py load                    # Load mit Standard-Repo
  uv run hf_model.py push --repo-id user/x   # Push zu anderem Repo
  uv run hf_model.py load --local-path ./m   # Load zu anderem Pfad
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Push Subcommand
    push_parser = subparsers.add_parser(
        "push",
        help="Lokales Modell zu HuggingFace Hub hochladen"
    )
    push_parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace Repository ID (default: {DEFAULT_REPO_ID})"
    )
    push_parser.add_argument(
        "--local-path",
        default=DEFAULT_LOCAL_PATH,
        help=f"Pfad zum lokalen Modell (default: {DEFAULT_LOCAL_PATH})"
    )
    
    # Load Subcommand
    load_parser = subparsers.add_parser(
        "load",
        help="Modell von HuggingFace Hub herunterladen"
    )
    load_parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace Repository ID (default: {DEFAULT_REPO_ID})"
    )
    load_parser.add_argument(
        "--local-path",
        default=DEFAULT_LOCAL_PATH,
        help=f"Pfad zum Speichern des Modells (default: {DEFAULT_LOCAL_PATH})"
    )
    
    args = parser.parse_args()
    
    if args.command == "push":
        push_model(args.repo_id, args.local_path)
    elif args.command == "load":
        load_model(args.repo_id, args.local_path)


if __name__ == "__main__":
    main()
