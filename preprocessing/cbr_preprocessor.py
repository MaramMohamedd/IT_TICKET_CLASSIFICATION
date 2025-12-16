from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

class CBRPreprocessor:
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1,2)
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, path):
        dump(self.vectorizer, path)

    def load(self, path):
        self.vectorizer = load(path)
