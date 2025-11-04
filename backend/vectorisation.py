from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk, numpy as np
import pandas as pd
import spacy
import numpy as np

nltk.download('punkt')

class VectoriseurWord2Vec:
    def __init__(self, taille_vecteur=100):
        self.taille = taille_vecteur
        self.model = None

    def vectoriser_phrase(self, phrase):
        """
        Retourne le vecteur de la phrase ,moyenne des vecteurs de mots 
        """
        mots = word_tokenize(phrase.lower())
        vects = []
        for mot in mots:
            if mot in self.model.wv:
                vects.append(self.model.wv[mot])
        if not vects:
            return [0.0] * self.taille
        # moyenne des vecterus
        s = sum(vects)
        return s / len(vects)


    def similarite_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        """cos(v1, v2), [0..1] """
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return (num / den) if den > 0 else 0.0

    def similarite_entre_phrases(self, p1: str, p2: str) -> float:
        """Cosinus entre deux phrases, moyenne des mots"""
        v1 = self.vectoriser_phrase(p1)
        v2 = self.vectoriser_phrase(p2)
        return self.similarite_cosine(v1, v2)

    def compute_label_vectors(self, input_csv, output_csv):
        df = pd.read_csv(input_csv)
        label_vectors = {}

        for label, group in df.groupby("label"):
            vectors = []
            for text in group["cleaned_text"]:
                v = self.vectoriser_phrase(str(text))
                vectors.append(v)
            mean_vec = np.mean(vectors, axis=0)
            label_vectors[label] = mean_vec

        output_df = pd.DataFrame.from_dict(label_vectors, orient="index")
        output_df.reset_index(inplace=True)
        output_df.rename(columns={"index": "label"}, inplace=True)
        output_df.to_csv(output_csv, index=False)

