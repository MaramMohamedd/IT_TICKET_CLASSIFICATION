import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class BasePreprocessor:
    def __init__(self, config):
        self.config = config
        self.stopwords = ENGLISH_STOP_WORDS if config.get("stopwords", True) else set()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        if self.config.get("lowercase", True):
            text = text.lower()

        if self.config.get("remove_urls", True):
            text = re.sub(r"http\S+|www\S+", " ", text)

        if self.config.get("remove_emails", True):
            text = re.sub(r"\S+@\S+", " ", text)

        if self.config.get("remove_numbers", False):
            text = re.sub(r"\d+", " ", text)

        if self.config.get("remove_punctuation", True):
            text = re.sub(r"[^\w\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.stopwords:
            text = " ".join([w for w in text.split() if w not in self.stopwords])

        return text

    def analyze_outliers(self, lengths):
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 - 1.5 * iqr
        print(f"[INFO] Length outlier range: {lower:.1f} – {upper:.1f}")
        return lower, upper

    def base_pipeline(self, df, text_col, label_col=None):
        if text_col not in df.columns:
            raise ValueError(f"text_col '{text_col}' not found in DataFrame")

        df = df.copy()

        if self.config.get("drop_nulls", True):
            nulls = df[text_col].isnull().sum()
            print(f"[INFO] Null texts removed: {nulls}")
            df = df.dropna(subset=[text_col])

        if self.config.get("remove_duplicates", True):
            dups = df.duplicated(subset=[text_col]).sum()
            print(f"[INFO] Duplicate texts removed: {dups}")
            df = df.drop_duplicates(subset=[text_col])

        df["cleaned_text"] = df[text_col].astype(str).apply(self.clean_text)

        lengths = df["cleaned_text"].apply(lambda t: len(t.split())).values
        if self.config.get("enable_outlier_analysis", True):
            self.analyze_outliers(lengths)

        texts = df["cleaned_text"].tolist()

        labels = df[label_col].tolist() if label_col else None

        return texts, labels
