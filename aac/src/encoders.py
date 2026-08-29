"""
src/encoders.py  —  Block 2 (semantic layer, pluggable)
Two interchangeable encoders so the project runs anywhere:
  - TfidfEncoder: offline, instant, great for quick runs / CI / the sandbox.
  - STEncoder: sentence-transformers embeddings (semantic). Default on Colab;
    catches novel attacks phrased in unseen words (intent, not keywords).
Both expose .fit(texts) and .transform(texts) -> 2D numpy array.
"""
import numpy as np


class TfidfEncoder:
    name = "tfidf"

    def __init__(self, max_features=4000, ngram=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram,
                                   sublinear_tf=True, min_df=1)

    def fit(self, texts):
        self.vec.fit(texts); return self

    def transform(self, texts):
        return self.vec.transform(texts).toarray().astype(np.float32)


class STEncoder:
    name = "sentence-transformer"

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self  # pretrained; nothing to fit

    def transform(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


def get_encoder(kind="auto"):
    """kind: 'tfidf' | 'st' | 'auto' (try ST, fall back to TF-IDF)."""
    if kind == "tfidf":
        return TfidfEncoder()
    if kind == "st":
        return STEncoder()
    try:
        return STEncoder()
    except Exception as e:
        print(f"[encoders] sentence-transformers unavailable ({e}); using TF-IDF.")
        return TfidfEncoder()
