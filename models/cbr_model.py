# CBR Model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class CBRModel:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.nn = None
        self.case_base = None
        self.X = None

    def fit(self, df):
        self.case_base = df.reset_index(drop=True)
        corpus = df["description"].astype(str).tolist()
        self.X = self.vectorizer.fit_transform(corpus)
        self.nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="cosine"
        ).fit(self.X)

    def retrieve(self, query, top_k=3):
        qv = self.vectorizer.transform([query])
        dists, idxs = self.nn.kneighbors(qv, n_neighbors=top_k)

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            row = self.case_base.iloc[idx]
            sim = 1 - dist
            results.append({
                "case_id": row["case_id"],
                "description": row["description"],
                "solution": row["solution"],
                "similarity": float(sim)
            })
        return results

    def add_case(self, case_dict):
        self.case_base = pd.concat(
            [self.case_base, pd.DataFrame([case_dict])],
            ignore_index=True
        )
        self.fit(self.case_base)

    def save(self, path):
        joblib.dump(self.vectorizer, path + "_tfidf.joblib")
        self.case_base.to_csv(path + "_casebase.csv", index=False)

    def load(self, path):
        self.vectorizer = joblib.load(path + "_tfidf.joblib")
        self.case_base = pd.read_csv(path + "_casebase.csv")
        corpus = self.case_base["description"].astype(str).tolist()
        self.X = self.vectorizer.transform(corpus)
        self.nn = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric="cosine"
        ).fit(self.X)

