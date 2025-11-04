# main.py
import pandas as pd
import os
import sys
import argparse
from preprocess import clean_text

def clean_csv(input_csv: str, output_csv: str, text_col: str = "text") -> None:
    if not os.path.exists(input_csv):
        print(f"[ERREUR] Fichier introuvable : {input_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        print(f"[ERREUR] Colonne '{text_col}' absente. Colonnes dispo : {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print("🧹 Nettoyage en cours...")
    df["cleaned_text"] = df[text_col].astype(str).apply(clean_text)

    # Supprimer les lignes vides après nettoyage
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].copy()
    after = len(df)

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f" Nettoyage terminé : {after}/{before} lignes gardées")
    print(f" Fichier sauvegardé : {output_csv}")
    print(" Aperçu :")
    print(df[[text_col, "cleaned_text"]].head().to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Nettoyer un corpus texte anglais avec NLTK")
    parser.add_argument("--input", "-i", default="text.csv", help="Fichier CSV d'entrée (défaut: text.csv)")
    parser.add_argument("--output", "-o", default="cleaned_text.csv", help="Fichier CSV de sortie (défaut: cleaned_text.csv)")
    parser.add_argument("--text-col", "-c", default="text", help="Nom de la colonne texte (défaut: text)")
    args = parser.parse_args()

    clean_csv(args.input, args.output, args.text_col)

if __name__ == "__main__":
    main()
