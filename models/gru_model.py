# GRU Model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class GRUModel:
    def __init__(self, vocab_size, max_len, num_classes, embed_dim=128, gru_units=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_classes = num_classes

        self.model = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            GRU(gru_units),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    def predict(self, seq):
        pred = self.model.predict(np.array([seq]))
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))
        return class_id, confidence

    def save(self, path):
        self.model.save(path + "_model.h5")

    def load(self, path):
        self.model = tf.keras.models.load_model(path + "_model.h5")
