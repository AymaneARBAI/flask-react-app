import sys
import numpy as np
import pandas as pd
import spacy


class VectoriseurSpacy:
    def __init__(self, modele="fr_core_news_md"):
        # Charge le modèle français de spaCy avec vecteurs
        self.nlp = spacy.load(modele)

    def vectoriser_phrase(self, phrase: str) -> np.ndarray:
        """
        Retourne le vecteur de la phrase (vecteur du Doc spaCy).
        """
        doc = self.nlp(str(phrase))
        return doc.vector  # vecteur moyen des tokens

    @staticmethod
    def similarite_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        """cos(v1, v2), [0..1]"""
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return (num / den) if den > 0 else 0.0

    def similarite_entre_phrases(self, p1: str, p2: str) -> float:
        """Cosinus entre deux phrases vectorisées par spaCy."""
        v1 = self.vectoriser_phrase(p1)
        v2 = self.vectoriser_phrase(p2)
        return self.similarite_cosine(v1, v2)

    def compute_label_vectors(self, input_csv: str, output_csv: str):
        """
        Calcule les vecteurs moyens de chaque label à partir de la colonne 'cleaned_text'
        du fichier CSV d'entrée, et écrit le résultat dans un .csv
        """
        df = pd.read_csv(input_csv)

        if "cleaned_text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                "Le CSV doit contenir au moins les colonnes 'cleaned_text' et 'label'."
            )

        label_vectors = {}
        for label, group in df.groupby("label"):
            vectors = [self.vectoriser_phrase(t) for t in group["cleaned_text"]]
            label_vectors[label] = np.mean(vectors, axis=0)

        output_df = pd.DataFrame.from_dict(label_vectors, orient="index")
        output_df.reset_index(inplace=True)
        output_df.rename(columns={"index": "label"}, inplace=True)
        output_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    vec = VectoriseurSpacy()
    vec.compute_label_vectors(input_csv, output_csv)
