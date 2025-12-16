from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import dump, load


class SequencePreprocessor:
    def __init__(self, config):
        self.config = config
        self.max_len = config.get("max_seq_len", 120)
        self.tokenizer = Tokenizer(oov_token="<UNK>")

    def fit_transform(self, texts):
        self.tokenizer.fit_on_texts(texts)
        seqs = self.tokenizer.texts_to_sequences(texts)
        seqs = pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")
        return seqs

    def transform(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")

    def save(self, path):
        dump(self.tokenizer, path)

    def load(self, path):
        self.tokenizer = load(path)
