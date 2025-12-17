import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter
import ast


class Vecteur:
    def __init__(self, modele="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(modele)
        self.vecteurs_train = []
        self.labels_train = []

    def vectoriser_phrase(self, phrase: str) -> np.ndarray:
        return self.model.encode(str(phrase))

    @staticmethod
    def similarite_cosine(v1: np.ndarray, v2: np.ndarray):
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return (num / den) if den > 0 else 0.0

    def similarite_entre_phrases(self, p1: str, p2: str):
        v1 = self.vectoriser_phrase(p1)
        v2 = self.vectoriser_phrase(p2)
        return self.similarite_cosine(v1, v2)

    def precomputer_vecteurs(self, input_csv: str, output_csv: str):
        """
        Permet de pré-computer les vecteurs pour chaque phrase dans un 
        CSV et de sauvegarder le résultat dans un nouveau CSV.
        
        auteur: Lila
        """
        df = pd.read_csv(input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Le CSV doit contenir 'text' et 'label'.")

        vecteurs = []
        for texte in df["text"]:
            vec = self.vectoriser_phrase(texte)
            vecteurs.append(vec.tolist())

        df["vecteur"] = vecteurs

        vecteur_array = np.array(vecteurs)
        for i in range(vecteur_array.shape[1]):
            df[f"dim_{i}"] = vecteur_array[:, i]

        df.to_csv(output_csv, index=False)

    def compute_label_vectors(self, input_csv: str, output_csv: str):
        """
        Permet de calculer le vecteur moyen pour chaque label

        auteur: Lila
        """
        df = pd.read_csv(input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Le CSV doit contenir 'text' et 'label'.")

        label_vectors = {}
        for label, group in df.groupby("label"):
            vectors = [self.vectoriser_phrase(t) for t in group["text"]]
            mean_vector = np.mean(vectors, axis=0)
            label_vectors[label] = mean_vector

        output_data = []
        for label, vector in label_vectors.items():
            row = {"label": label, "vecteur": vector.tolist()}
            for i, val in enumerate(vector):
                row[f"dim_{i}"] = val
            output_data.append(row)

        output_df = pd.DataFrame(output_data)
        cols = ["label", "vecteur"] + [c for c in output_df.columns if c.startswith("dim_")]
        output_df = output_df[cols]
        output_df.to_csv(output_csv, index=False)

    def predire_label_moyenne(self, texte: str, label_vectors_csv: str) -> str:
        """
        Prediction de l'émotion avec le vecteur moyen de chaque label
        auteur: Aymane
        """
        df = pd.read_csv(label_vectors_csv)
        v = self.vectoriser_phrase(texte)

        meilleures_sim = -1
        meilleur_label = None

        for _, row in df.iterrows():
            vec_label = row[[c for c in df.columns if c.startswith("dim_")]].to_numpy(dtype=float)
            sim = self.similarite_cosine(v, vec_label)
            if sim > meilleures_sim:
                meilleures_sim = sim
                meilleur_label = row["label"]

        return meilleur_label

    def entrainer_knn(self, input_csv: str):
        df = pd.read_csv(input_csv)
        if "vecteur" not in df.columns or "label" not in df.columns:
            raise ValueError("Le CSV doit contenir 'vecteur' et 'label'.")

        self.vecteurs_train = []
        self.labels_train = []

        for _, row in df.iterrows():
            vec = np.array(ast.literal_eval(row["vecteur"]))
            self.vecteurs_train.append(vec)
            self.labels_train.append(row["label"])

    def predire_label_knn(self, texte: str, k=5):
        """
        Prédiction de l'emotion avec la methode KNN
        
        auteur: Lila
        """
        if not self.vecteurs_train:
            raise ValueError("Vous devez d'abord appeler entrainer_knn() !")

        v = self.vectoriser_phrase(texte)

        similarites = []
        for i, vec_train in enumerate(self.vecteurs_train):
            sim = self.similarite_cosine(v, vec_train)
            similarites.append((sim, self.labels_train[i]))

        similarites.sort(reverse=True, key=lambda x: x[0])
        top_k = similarites[:k]

        votes = Counter([label for _, label in top_k])
        return votes.most_common(1)[0][0]

    def trouver_texte_le_plus_proche(self, texte: str, vectors_csv: str) -> tuple:
        """
        Recherche du texte du corpus ayant le vecteur le plus proche de texte d'entré
        
        auteur: Aymane
        """
        df = pd.read_csv(vectors_csv)

        v_query = self.vectoriser_phrase(texte)

        similarite_max = -1
        texte_le_plus_proche = None
        label_le_plus_proche = None

        for _, row in df.iterrows():
            vec_bdd = np.array(ast.literal_eval(row["vecteur"]))
            sim = self.similarite_cosine(v_query, vec_bdd)

            if sim > similarite_max:
                similarite_max = sim
                texte_le_plus_proche = row["text"]
                label_le_plus_proche = row["label"] if "label" in df.columns else None

        return texte_le_plus_proche, similarite_max


if __name__ == "__main__":
    vec = Vecteur()

    texte_test = "I’m so angry I could cry, everything is just too much right now."

    print(vec.trouver_texte_le_plus_proche(
            texte_test,
            "cleaned_text_with_vectors.csv"
        )
    )
