
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        texts = list(map(str, X))
        embs = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        return np.array(embs)

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = TfidfVectorizer(max_features=self.max_features,
                                   ngram_range=self.ngram_range)

    def fit(self, X, y=None):
        self.vec.fit(X)
        return self

    def transform(self, X):
        return self.vec.transform(X)

    def save(self, path):
        joblib.dump(self.vec, path)

    def load(self, path):
        self.vec = joblib.load(path)
        return self
