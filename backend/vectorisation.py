import sys
import numpy as np
import pandas as pd
import spacy
from collections import Counter

class Vecteur:
    def _init_(self, modele="en_core_web_md"):
        self.nlp = spacy.load(modele)
        self.vecteurs_train = []
        self.labels_train = []
    
    def vectoriser_phrase(self, phrase: str) -> np.ndarray:
        doc = self.nlp(str(phrase))
        return doc.vector
    
    @staticmethod
    def similarite_cosine(v1: np.ndarray, v2: np.ndarray):
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return (num / den) if den > 0 else 0.0
    
    def similarite_entre_phrases(self, p1: str, p2: str):
        v1 = self.vectoriser_phrase(p1)
        v2 = self.vectoriser_phrase(p2)
        return self.similarite_cosine(v1, v2)
    
    def compute_label_vectors(self, input_csv: str, output_csv: str):
        df = pd.read_csv(input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                "Le CSV doit contenir au moins les colonnes 'text' et 'label'."
            )
        
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
    
    def predire_label(self, texte: str, label_vectors_csv: str) -> str:
        df = pd.read_csv(label_vectors_csv)
        v = self.vectoriser_phrase(texte)
        
        meilleures_sim = -1
        meilleur_label = None
        
        for _, row in df.iterrows():
            label = row["label"]
            vec_label = row[[c for c in df.columns if c.startswith("dim_")]].to_numpy(dtype=float)
            sim = self.similarite_cosine(v, vec_label)
            if sim > meilleures_sim:
                meilleures_sim = sim
                meilleur_label = label
        
        return meilleur_label
    
    def entrainer_knn(self, input_csv: str):
        df = pd.read_csv(input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                "Le CSV doit contenir au moins les colonnes 'text' et 'label'."
            )
        
        self.vecteurs_train = []
        self.labels_train = []
        
        for _, row in df.iterrows():
            vec = self.vectoriser_phrase(row["text"])
            self.vecteurs_train.append(vec)
            self.labels_train.append(row["label"])
    
    def predire_label_knn(self, texte: str, k=5):
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
        label_predit = votes.most_common(1)[0][0]
        
        return label_predit
    
    def predire_label_knn_detaille(self, texte: str, k=5):
        if not self.vecteurs_train:
            raise ValueError("Vous devez d'abord appeler entrainer_knn() !")
        
        v = self.vectoriser_phrase(texte)
        
        similarites = []
        for i, vec_train in enumerate(self.vecteurs_train):
            sim = self.similarite_cosine(v, vec_train)
            similarites.append((sim, self.labels_train[i], i))
        
        similarites.sort(reverse=True, key=lambda x: x[0])
        top_k = similarites[:k]
        
        votes = Counter([label for _, label, _ in top_k])
        label_predit = votes.most_common(1)[0][0]
        
        return label_predit


if _name_ == "_main_":
    vec = Vecteur()
    
    input_csv = "/Users/aymanearbai/Documents/cours/S7/ProcessusDev/flask-react-app/backend/cleaned_text.csv"
    output_csv = "/Users/aymanearbai/Documents/cours/S7/ProcessusDev/flask-react-app/backend/output.csv"
    vec.compute_label_vectors(input_csv, output_csv)
    
    texte_test = "Oh my god, I can't believe this is happening, this is so unexpected and amazing, I'm absolutely shocked"
    
    prediction_moyenne = vec.predire_label(texte_test, output_csv)
    
    vec.entrainer_knn(input_csv)
    vec.predire_label_knn_detaille(texte_test, k=5)