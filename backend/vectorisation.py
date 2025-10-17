from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Si pas déjà téléchargé
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
        # moyenne
        s = sum(vects)
        return s / len(vects)
