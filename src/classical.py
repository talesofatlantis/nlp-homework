"""TF-IDF + Logistic Regression baseline."""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def run_classical(
    texts_train: np.ndarray,
    y_train: np.ndarray,
    texts_test: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), time.time() - t0
