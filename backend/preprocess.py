# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

# Assure-toi que ces ressources sont téléchargées une fois
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS_EN = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Nettoyage minimal en anglais avec NLTK :
    - minuscules
    - enlève accents
    - enlève ponctuation et chiffres
    - tokenisation nltk
    - supprime les stopwords anglais
    """
    if not isinstance(text, str):
        return ""

    # 1. minuscules
    t = text.lower()
    # 2. enlever les accents
    t = unidecode(t)
    # 3. enlever ponctuation et chiffres
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", " ", t)
    # 4. tokenisation
    tokens = word_tokenize(t)
    # 5. suppression des stopwords
    tokens = [w for w in tokens if w not in STOPWORDS_EN]
    # 6. suppression des espaces vides et jointure
    cleaned = " ".join(tokens).strip()

    return cleaned
